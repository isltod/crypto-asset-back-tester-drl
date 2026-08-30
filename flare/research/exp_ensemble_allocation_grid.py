"""
flare/research/exp_ensemble_allocation_grid.py
- 새로 튜닝된 RADE 공식 표준 (STANDARD_GOLDEN: TF 1.0% x MR 8.0%)과 FLARE 5x의
  자산 배분 비율(10:0 ~ 0:10) 전수 그리드 탐색 및 분기 리밸런싱 성과 분석
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


def run_allocation_grid_analysis():
    sys.stdout.reconfigure(encoding='utf-8')
    data_dir = PROJECT_ROOT / "data"
    
    print("\n" + "=" * 100)
    print(" 🚀 [새로운 RADE 표준 (TF 1% x MR 8%) x FLARE 5x] 자산 배분 비율 전수 그리드 탐색 (4개년)")
    print("=" * 100)

    # 1. 새로운 RADE 표준(STANDARD_GOLDEN) 시뮬레이션 실행 (초기 자본 $1,000 기준 정규화)
    print("\n[1/3] 🌟 신규 RADE 공식 표준 (STANDARD_GOLDEN: TF 1.0% x MR 8.0%, MDD 14.18%) 시뮬레이션 중...")
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

    # 2. FLARE 5x 멀티코인 시뮬레이션 실행
    print("[2/3] ⚡ FLARE 5x 스윙 (4대 코인: BTC, ETH, SOL, XRP) 시뮬레이션 중...")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    flare_res = run_equal_weight_multi_position(
        symbols=symbols,
        data_dir=data_dir,
        initial_capital=base_cap,
        leverage=5.0,
        allocation_ratio=0.80
    )

    # 3. 시계열 데이터 결합 및 시간 축 정렬
    df_rade = pd.DataFrame({
        "datetime": pd.to_datetime(rade_res["timestamps"], utc=True),
        "rade_eq": rade_res["equity_curve"]
    }).drop_duplicates("datetime")

    df_flare = pd.DataFrame({
        "datetime": pd.to_datetime(flare_res["timestamps"], utc=True),
        "flare_eq": flare_res["equity_curve"]
    }).drop_duplicates("datetime")

    merged = pd.merge(df_rade, df_flare, on="datetime", how="outer").sort_values("datetime").ffill().bfill().reset_index(drop=True)
    merged["quarter"] = merged["datetime"].dt.to_period("Q")

    # RADE 및 FLARE 정규화 수익률 (1.0 기준)
    merged["rade_norm"] = merged["rade_eq"] / base_cap
    merged["flare_norm"] = merged["flare_eq"] / base_cap

    # 4. 배분 비율별 그리드 탐색 (10:0 ~ 0:10)
    ratios = [
        (1.0, 0.0),
        (0.9, 0.1),
        (0.85, 0.15),
        (0.8, 0.2),
        (0.75, 0.25),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
        (0.4, 0.6),
        (0.3, 0.7),
        (0.2, 0.8),
        (0.0, 1.0)
    ]

    print("\n[3/3] 📊 자산 배분 비율별 [정적 배분 vs 분기 리밸런싱] 전수 비교 분석 중...\n")

    results_static = []
    results_rebal = []

    total_init_cap = 10_000.0

    for r_w, f_w in ratios:
        name = f"{int(r_w*100)}:{int(f_w*100)}"
        
        # A. 정적 배분 (Static: 리밸런싱 없이 방치)
        static_series = (total_init_cap * r_w * merged["rade_norm"]) + (total_init_cap * f_w * merged["flare_norm"])
        final_static = static_series.iloc[-1]
        ret_static = ((final_static - total_init_cap) / total_init_cap) * 100.0
        peak_s = static_series.cummax()
        dd_s = (peak_s - static_series) / peak_s
        mdd_static = dd_s.max() * 100.0
        calmar_static = ret_static / mdd_static if mdd_static > 0 else 0.0

        results_static.append({
            "ratio": name,
            "rade_w": r_w,
            "flare_w": f_w,
            "final_equity": final_static,
            "total_return": ret_static,
            "mdd": mdd_static,
            "calmar": calmar_static
        })

        # B. 분기(3개월) 리밸런싱 (Quarterly Rebalancing)
        # 분기마다 총자산을 r_w : f_w 로 재분배
        quarters = merged["quarter"].unique()
        curr_total = total_init_cap
        rebal_curve = []

        for q in quarters:
            q_df = merged[merged["quarter"] == q].copy()
            if q_df.empty:
                continue
            
            # 분기 시작 시 배분된 자본
            q_rade_cap = curr_total * r_w
            q_flare_cap = curr_total * f_w

            # 분기 내 RADE 및 FLARE 상대 수익률
            rade_q_start = q_df["rade_norm"].iloc[0]
            flare_q_start = q_df["flare_norm"].iloc[0]

            q_rade_curve = q_rade_cap * (q_df["rade_norm"] / rade_q_start)
            q_flare_curve = q_flare_cap * (q_df["flare_norm"] / flare_q_start)
            q_total_curve = q_rade_curve + q_flare_curve

            rebal_curve.extend(q_total_curve.tolist())
            curr_total = q_total_curve.iloc[-1]

        rebal_series = pd.Series(rebal_curve)
        final_rebal = rebal_series.iloc[-1]
        ret_rebal = ((final_rebal - total_init_cap) / total_init_cap) * 100.0
        peak_r = rebal_series.cummax()
        dd_r = (peak_r - rebal_series) / peak_r
        mdd_rebal = dd_r.max() * 100.0
        calmar_rebal = ret_rebal / mdd_rebal if mdd_rebal > 0 else 0.0

        results_rebal.append({
            "ratio": name,
            "rade_w": r_w,
            "flare_w": f_w,
            "final_equity": final_rebal,
            "total_return": ret_rebal,
            "mdd": mdd_rebal,
            "calmar": calmar_rebal
        })

    df_res_static = pd.DataFrame(results_static)
    df_res_rebal = pd.DataFrame(results_rebal)

    print("=" * 105)
    print(" 📋 [1. 정적 배분 (Static: 리밸런싱 없음) 비율별 성적표]")
    print("=" * 105)
    print(f" {'배분 비율 (R:F)':<15} | {'최종 자산 ($)':<16} | {'4년 총수익률 (%)':<18} | {'통합 MDD (%)':<15} | {'칼마 비율 (Calmar)':<18}")
    print("-" * 105)
    for _, row in df_res_static.iterrows():
        print(f" {row['ratio']:<15} | ${row['final_equity']:>14,.2f} | {row['total_return']:>+16.2f}% | {row['mdd']:>13.2f}% | {row['calmar']:>16.2f}")

    print("\n" + "=" * 105)
    print(" 📋 [2. 분기 리밸런싱 (Quarterly Rebalanced) 비율별 성적표 ⭐]")
    print("=" * 105)
    print(f" {'배분 비율 (R:F)':<15} | {'최종 자산 ($)':<16} | {'4년 총수익률 (%)':<18} | {'통합 MDD (%)':<15} | {'칼마 비율 (Calmar)':<18}")
    print("-" * 105)
    for _, row in df_res_rebal.iterrows():
        is_best = " 🏆 (칼마 1위)" if row['calmar'] == df_res_rebal['calmar'].max() else ""
        print(f" {row['ratio']:<15} | ${row['final_equity']:>14,.2f} | {row['total_return']:>+16.2f}% | {row['mdd']:>13.2f}% | {row['calmar']:>16.2f}{is_best}")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    run_allocation_grid_analysis()
