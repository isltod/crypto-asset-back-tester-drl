"""
[실험 34] 2022년 크립토 윈터 구간 한정 TH=0.45 vs TH=0.74 정밀 비교 분석
- 목적:
  - 2022년 폭락장에서 TH=0.45(+$1,289) 대비 TH=0.74(+$194)의 성과 차이 원인 규명
  - 2022년 내 MR vs TF 거래수, 승률, 손익 기여금 및 개별 거래 내역 대조
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from rade.backtest.simulator import BacktestSimulator
from rade.engines.trend_following import TrendFollowingEngine
from rade.engines.mean_reversion import MeanReversionEngine


def analyze_2022_difference():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    # 1. TH = 0.45
    reg_045 = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.45, cooldown_bars=0)
    df_045 = reg_045.calculate_regime_probabilities(df_ind)
    test_045 = df_045.iloc[720:].reset_index(drop=True)
    sim_045 = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        bear_mode="CASH",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_045 = sim_045.run(test_045)

    # 2. TH = 0.74
    reg_074 = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.74, cooldown_bars=0)
    df_074 = reg_074.calculate_regime_probabilities(df_ind)
    test_074 = df_074.iloc[720:].reset_index(drop=True)
    sim_074 = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        bear_mode="CASH",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_074 = sim_074.run(test_074)

    # 3. 2022년 거래 필터링
    t_045 = res_045["trades_df"].copy()
    t_074 = res_074["trades_df"].copy()
    t_045["entry_dt"] = pd.to_datetime(t_045["entry_time"])
    t_074["entry_dt"] = pd.to_datetime(t_074["entry_time"])

    t_045_2022 = t_045[t_045["entry_dt"].dt.year == 2022].reset_index(drop=True)
    t_074_2022 = t_074[t_074["entry_dt"].dt.year == 2022].reset_index(drop=True)

    print("=" * 85)
    print("      [실험 34] 2022년 크립토 윈터 구간 한정 TH=0.45 vs TH=0.74 정밀 대조 리포트")
    print("=" * 85)

    print("\n[1. 2022년 전체 성과 1:1 비교]")
    pnl_045 = t_045_2022["pnl"].sum()
    pnl_074 = t_074_2022["pnl"].sum()
    wr_045 = len(t_045_2022[t_045_2022["pnl"] > 0]) / len(t_045_2022) * 100.0 if len(t_045_2022) > 0 else 0.0
    wr_074 = len(t_074_2022[t_074_2022["pnl"] > 0]) / len(t_074_2022) * 100.0 if len(t_074_2022) > 0 else 0.0

    print(f" * TH=0.45: 총 손익 {pnl_045:+9.2f}$ | 총 거래 {len(t_045_2022):2d}회 | 승률 {wr_045:5.1f}%")
    print(f" * TH=0.74: 총 손익 {pnl_074:+9.2f}$ | 총 거래 {len(t_074_2022):2d}회 | 승률 {wr_074:5.1f}%")
    print(f" * 손익 차이: {pnl_074 - pnl_045:+9.2f}$")

    print("\n[2. 2022년 엔진별(MR vs TF) 세부 비교]")
    for eng in ["MEAN_REVERSION", "TREND_FOLLOWING"]:
        sub_045 = t_045_2022[t_045_2022["engine"] == eng]
        sub_074 = t_074_2022[t_074_2022["engine"] == eng]
        
        p_045 = sub_045["pnl"].sum()
        p_074 = sub_074["pnl"].sum()
        
        w_045 = len(sub_045[sub_045["pnl"] > 0]) / len(sub_045) * 100.0 if len(sub_045) > 0 else 0.0
        w_074 = len(sub_074[sub_074["pnl"] > 0]) / len(sub_074) * 100.0 if len(sub_074) > 0 else 0.0

        print(f" * [{eng}]")
        print(f"   - TH=0.45: {len(sub_045):2d}회 거래 | 승률 {w_045:5.1f}% | 총 PnL: {p_045:+9.2f}$")
        print(f"   - TH=0.74: {len(sub_074):2d}회 거래 | 승률 {w_074:5.1f}% | 총 PnL: {p_074:+9.2f}$")

    print("\n[3. 2022년 개별 거래 내역 대조 (TH=0.74 vs TH=0.45)]")
    print("--- TH=0.45 2022년 거래 ---")
    for idx, row in t_045_2022.iterrows():
        pnl_val = row["pnl"]
        rsn = row.get("reason", row.get("exit_reason", "N/A"))
        print(f"   {idx+1:2d}) [{row['engine'][:2]}] {str(row['entry_time'])[:16]} ~ {str(row['exit_time'])[:16]} | PnL: {pnl_val:+7.2f}$ | 사유: {rsn}")

    print("\n--- TH=0.74 2022년 거래 ---")
    for idx, row in t_074_2022.iterrows():
        pnl_val = row["pnl"]
        rsn = row.get("reason", row.get("exit_reason", "N/A"))
        print(f"   {idx+1:2d}) [{row['engine'][:2]}] {str(row['entry_time'])[:16]} ~ {str(row['exit_time'])[:16]} | PnL: {pnl_val:+7.2f}$ | 사유: {rsn}")

    print("=" * 85)


if __name__ == "__main__":
    analyze_2022_difference()
