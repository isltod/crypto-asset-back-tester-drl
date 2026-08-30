"""
flare/research/exp_consecutive_losses_comparison.py
- 9:1 vs 8:2 vs RADE단독 vs FLARE단독 연속 손실(Streak) 및 회복 기간 정밀 분석
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


def run_streak_analysis():
    sys.stdout.reconfigure(encoding='utf-8')
    data_dir = PROJECT_ROOT / "data"
    
    # 1. RADE
    f_is = data_dir / "BTCUSDT_1h_2021_2024.csv"
    f_oos = data_dir / "BTCUSDT_1h_2024_OOS.csv"
    df_raw = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
    df_indicators = add_all_indicators(df_raw)
    
    preset = get_preset("STANDARD_GOLDEN")
    regime_manager = RegimeManager(hmm_window=preset.hmm_window, retrain_interval=preset.retrain_interval, trans_threshold=preset.hmm_base_threshold, cooldown_bars=0)
    df_processed = regime_manager.calculate_regime_probabilities(df_indicators)
    
    base_cap = 10_000.0
    sim = BacktestSimulator(initial_capital=base_cap, trend_risk_pct=preset.trend_risk_pct, mr_risk_pct=preset.mr_risk_pct, leverage=preset.leverage, bear_mode=preset.bear_mode, maker_fee_pct=0.0002, taker_fee_pct=0.0005, slippage_pct=0.0002)
    rade_res = sim.run(df_processed)

    # 2. FLARE
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    flare_res = run_equal_weight_multi_position(symbols=symbols, data_dir=data_dir, initial_capital=base_cap, leverage=5.0, allocation_ratio=0.80)

    # RADE 거래별 승패
    rade_trades = pd.DataFrame(rade_res["trades_detail"]) if "trades_detail" in rade_res else pd.DataFrame()
    
    # 시간축 결합
    df_rade = pd.DataFrame({"datetime": pd.to_datetime(rade_res["timestamps"], utc=True), "rade_eq": rade_res["equity_curve"]}).drop_duplicates("datetime")
    df_flare = pd.DataFrame({"datetime": pd.to_datetime(flare_res["timestamps"], utc=True), "flare_eq": flare_res["equity_curve"]}).drop_duplicates("datetime")
    merged = pd.merge(df_rade, df_flare, on="datetime", how="outer").sort_values("datetime").ffill().bfill().reset_index(drop=True)
    merged["quarter"] = merged["datetime"].dt.to_period("Q")
    merged["month"] = merged["datetime"].dt.to_period("M")
    merged["rade_norm"] = merged["rade_eq"] / base_cap
    merged["flare_norm"] = merged["flare_eq"] / base_cap

    def calc_rebal_series(r_w, f_w):
        curr_total = base_cap
        rebal_curve = []
        for q in merged["quarter"].unique():
            g_df = merged[merged["quarter"] == q]
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
        return pd.Series(rebal_curve)

    s_91 = calc_rebal_series(0.9, 0.1)
    s_82 = calc_rebal_series(0.8, 0.2)

    # 월별 수익률 및 연속 적자 개월 수 분석
    merged_m = merged.copy()
    merged_m["eq_91"] = s_91.values
    merged_m["eq_82"] = s_82.values

    df_monthly = merged_m.groupby("month").last().reset_index()
    df_monthly["ret_91"] = df_monthly["eq_91"].pct_change() * 100
    df_monthly["ret_82"] = df_monthly["eq_82"].pct_change() * 100
    df_monthly["ret_rade"] = df_monthly["rade_eq"].pct_change() * 100
    df_monthly["ret_flare"] = df_monthly["flare_eq"].pct_change() * 100

    def get_max_losing_streak(returns):
        max_streak = 0
        current_streak = 0
        for r in returns.dropna():
            if r < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def get_loss_months_count(returns):
        valid = returns.dropna()
        loss_cnt = (valid < 0).sum()
        total_cnt = len(valid)
        return loss_cnt, total_cnt, (loss_cnt / total_cnt) * 100

    streak_91 = get_max_losing_streak(df_monthly["ret_91"])
    streak_82 = get_max_losing_streak(df_monthly["ret_82"])
    streak_rade = get_max_losing_streak(df_monthly["ret_rade"])
    streak_flare = get_max_losing_streak(df_monthly["ret_flare"])

    loss_91, tot, pct_91 = get_loss_months_count(df_monthly["ret_91"])
    loss_82, _, pct_82 = get_loss_months_count(df_monthly["ret_82"])
    loss_rade, _, pct_rade = get_loss_months_count(df_monthly["ret_rade"])
    loss_flare, _, pct_flare = get_loss_months_count(df_monthly["ret_flare"])

    print("\n" + "=" * 85)
    print(" 📉 [연속 손실 및 월별 적자 확률 정밀 비교 분석 (48개월 표본)]")
    print("=" * 85)
    print(f" {'구분':<20} | {'최대 연속 적자 개월':<16} | {'총 적자 개월 수':<15} | {'월간 적자 확률 (%)':<15}")
    print("-" * 85)
    print(f" {'🌟 RADE 표준 단독':<20} | {streak_rade:<16} 개월 | {loss_rade:>2} / {tot} 개월 | {pct_rade:>13.1f}%")
    print(f" {'🛡️ 9:1 앙상블':<20} | {streak_91:<16} 개월 | {loss_91:>2} / {tot} 개월 | {pct_91:>13.1f}%")
    print(f" {'⭐ 8:2 앙상블':<20} | {streak_82:<16} 개월 | {loss_82:>2} / {tot} 개월 | {pct_82:>13.1f}%")
    print(f" {'⚡ FLARE 5x 단독':<20} | {streak_flare:<16} 개월 | {loss_flare:>2} / {tot} 개월 | {pct_flare:>13.1f}%")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_streak_analysis()
