"""
flare/research/exp_ensemble_rebalance_frequency.py
- 새로 튜닝된 RADE 표준 (TF 1.0% x MR 8.0%) x FLARE 5x
  주요 배분 비율(9:1, 85:15, 8:2, 7:3)에 대해 리밸런싱 주기(1M, 2M, 3M, 4M, 6M, 12M, No-Rebalance) 전수 스캔
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from rade.backtest.simulator import BacktestSimulator
from rade.config.presets import get_preset
from flare.backtest.test_multi_position_equal_weight import run_equal_weight_multi_position


def run_frequency_scan():
    sys.stdout.reconfigure(encoding='utf-8')
    data_dir = PROJECT_ROOT / "data"
    
    print("\n" + "=" * 105)
    print(" 🚀 [신규 RADE 표준 x FLARE 5x] 리밸런싱 주기(1개월 ~ 12개월) 전수 그리드 스캔 (4개년)")
    print("=" * 105)

    # 1. RADE 시뮬레이션
    f_is = data_dir / "BTCUSDT_1h_2021_2024.csv"
    f_oos = data_dir / "BTCUSDT_1h_2024_OOS.csv"
    df_raw = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
    df_indicators = add_all_indicators(df_raw)
    
    preset = get_preset("STANDARD_GOLDEN")
    regime_manager = RegimeManager(
        hmm_window=preset.hmm_window,
        retrain_interval=preset.retrain_interval,
        trans_threshold=preset.hmm_base_threshold,
        cooldown_bars=0
    )
    df_processed = regime_manager.calculate_regime_probabilities(df_indicators)
    
    base_cap = 10_000.0
    sim = BacktestSimulator(
        initial_capital=base_cap,
        trend_risk_pct=preset.trend_risk_pct,
        mr_risk_pct=preset.mr_risk_pct,
        leverage=preset.leverage,
        bear_mode=preset.bear_mode,
        maker_fee_pct=0.0002,
        taker_fee_pct=0.0005,
        slippage_pct=0.0002,
    )
    rade_res = sim.run(df_processed)

    # 2. FLARE 5x 시뮬레이션
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    flare_res = run_equal_weight_multi_position(
        symbols=symbols,
        data_dir=data_dir,
        initial_capital=base_cap,
        leverage=5.0,
        allocation_ratio=0.80
    )

    df_rade = pd.DataFrame({
        "datetime": pd.to_datetime(rade_res["timestamps"], utc=True),
        "rade_eq": rade_res["equity_curve"]
    }).drop_duplicates("datetime")

    df_flare = pd.DataFrame({
        "datetime": pd.to_datetime(flare_res["timestamps"], utc=True),
        "flare_eq": flare_res["equity_curve"]
    }).drop_duplicates("datetime")

    merged = pd.merge(df_rade, df_flare, on="datetime", how="outer").sort_values("datetime").ffill().bfill().reset_index(drop=True)
    merged["rade_norm"] = merged["rade_eq"] / base_cap
    merged["flare_norm"] = merged["flare_eq"] / base_cap

    # 3. 리밸런싱 주기 정의 (월 단위)
    freq_dict = {
        "1M (매월 리밸런싱)": "1M",
        "2M (격월 리밸런싱)": "2M",
        "3M (분기 리밸런싱 ⭐)": "3M",
        "4M (4개월 리밸런싱)": "4M",
        "6M (반기 리밸런싱)": "6M",
        "12M (연간 리밸런싱)": "12M",
        "No-Rebal (정적 방치)": None,
    }

    target_ratios = [
        (0.90, 0.10, "90:10 (안정형)"),
        (0.85, 0.15, "85:15 (중도형)"),
        (0.80, 0.20, "80:20 (황금 표준)"),
        (0.70, 0.30, "70:30 (공격형)")
    ]

    for r_w, f_w, ratio_title in target_ratios:
        print(f"\n=========================================================================================================")
        print(f" 📊 [{ratio_title}] 리밸런싱 주기별 성적 비교")
        print(f"=========================================================================================================")
        print(f" {'리밸런싱 주기':<22} | {'최종 자산 ($)':<16} | {'4년 총수익률 (%)':<18} | {'통합 MDD (%)':<15} | {'칼마 비율 (Calmar)':<18}")
        print("-" * 105)

        for freq_name, freq_code in freq_dict.items():
            if freq_code is None:
                # 정적 방치
                s_curve = (base_cap * r_w * merged["rade_norm"]) + (base_cap * f_w * merged["flare_norm"])
            else:
                # N개월 단위 리밸런싱
                # 주기별 그룹핑
                merged_temp = merged.copy()
                if freq_code == "1M":
                    merged_temp["group"] = merged_temp["datetime"].dt.to_period("M")
                elif freq_code == "2M":
                    # 2달 단위
                    merged_temp["group"] = (merged_temp["datetime"].dt.year * 12 + merged_temp["datetime"].dt.month - 1) // 2
                elif freq_code == "3M":
                    merged_temp["group"] = merged_temp["datetime"].dt.to_period("Q")
                elif freq_code == "4M":
                    merged_temp["group"] = (merged_temp["datetime"].dt.year * 12 + merged_temp["datetime"].dt.month - 1) // 4
                elif freq_code == "6M":
                    merged_temp["group"] = (merged_temp["datetime"].dt.year * 12 + merged_temp["datetime"].dt.month - 1) // 6
                elif freq_code == "12M":
                    merged_temp["group"] = merged_temp["datetime"].dt.to_period("Y")

                curr_total = base_cap
                rebal_curve = []
                for g in merged_temp["group"].unique():
                    g_df = merged_temp[merged_temp["group"] == g]
                    if g_df.empty:
                        continue
                    q_rade_cap = curr_total * r_w
                    q_flare_cap = curr_total * f_w
                    rade_start = g_df["rade_norm"].iloc[0]
                    flare_start = g_df["flare_norm"].iloc[0]
                    q_rade = q_rade_cap * (g_df["rade_norm"] / rade_start)
                    q_flare = q_flare_cap * (g_df["flare_norm"] / flare_start)
                    q_tot = q_rade + q_flare
                    rebal_curve.extend(q_tot.tolist())
                    curr_total = q_tot.iloc[-1]
                s_curve = pd.Series(rebal_curve)

            final_eq = s_curve.iloc[-1]
            tot_ret = ((final_eq - base_cap) / base_cap) * 100.0
            peak = s_curve.cummax()
            dd = (peak - s_curve) / peak
            mdd = dd.max() * 100.0
            calmar = tot_ret / mdd if mdd > 0 else 0.0

            is_star = " 🏆" if freq_code == "3M" else ""
            print(f" {freq_name:<22} | ${final_eq:>14,.2f} | {tot_ret:>+16.2f}% | {mdd:>13.2f}% | {calmar:>16.2f}{is_star}")

    print("\n" + "=" * 105 + "\n")


if __name__ == "__main__":
    run_frequency_scan()
