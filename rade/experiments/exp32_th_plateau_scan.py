"""
[실험 32] HMM 임계값(TH) 0.60 ~ 0.80 정밀 핀포인트 스캔 (Plateau 분석)
- 목적:
  - 0.70 주변의 Bi-modal 2차 피크가 단일 과적합 스파이크인지, 넓은 안정 고원(Plateau)인지 검증
  - 0.60 ~ 0.80 구간을 0.02 단위로 정밀 스캔하여 최적의 안정점 탐색
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


def run_pinpoint_scan():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    fine_thresholds = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80]
    
    print("=" * 95)
    print("      [실험 32] HMM 임계값 0.60 ~ 0.80 정밀 핀포인트 스캔 (Plateau 분석)")
    print("=" * 95)
    print(f"{'HMM 임계값 (TH)':<16} | {'4개년 수익률':<12} | {'최대 낙폭 (MDD)':<14} | {'손익비 (PF)':<10} | {'거래 횟수':<10} | {'승률'}")
    print("-" * 95)

    results = []
    for th in fine_thresholds:
        reg = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=th, cooldown_bars=0)
        df_proc = reg.calculate_regime_probabilities(df_ind)
        test_df = df_proc.iloc[720:].reset_index(drop=True)
        
        sim = BacktestSimulator(
            initial_capital=10000.0,
            risk_per_trade_pct=0.02,
            leverage=3.0,
            bear_mode="CASH",
            use_regime_transition_cut=False,
            trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
            mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
        )
        res = sim.run(test_df)
        
        ret_str = f"{res['total_return_pct']:+.2f}%"
        mdd_str = f"{res['mdd_pct']:.2f}%"
        pf_str = f"{res['profit_factor']:.2f}"
        tr_str = f"{res['total_trades']}회"
        wr_str = f"{res['win_rate_pct']:.1f}%"
        
        results.append((th, res))
        print(f"TH = {th:<11.2f} | {ret_str:<12} | {mdd_str:<14} | {pf_str:<10} | {tr_str:<10} | {wr_str}")

    print("=" * 95)


if __name__ == "__main__":
    run_pinpoint_scan()
