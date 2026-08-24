"""
[실험 31] HMM 임계값(Transition Threshold) 전구간 민감도 및 불감대 실증 분석
- 목적:
  - Opus 4.6의 "0.35~0.45는 불감대(Dead Zone)이며 진짜 견고성은 0.55, 0.60, 0.70을 봐야 한다"는 비판을 실측 데이터로 검증
  - HMM 사후확률의 실제 값 분포(히스토그램)와 TH=0.35부터 0.85까지 성과 궤적 분석
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


def run_threshold_sensitivity():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    thresholds = [0.34, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
    
    print("=" * 85)
    print("      [실험 31] HMM 상태 전이 임계값(TH) 전구간(0.34~0.80) 민감도 실측 검증")
    print("=" * 85)
    print(f"{'HMM 임계값 (TH)':<16} | {'4개년 수익률':<12} | {'최대 낙폭 (MDD)':<14} | {'손익비 (PF)':<10} | {'거래 횟수':<10} | {'승률'}")
    print("-" * 85)

    for th in thresholds:
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
        
        print(f"TH = {th:<11.2f} | {ret_str:<12} | {mdd_str:<14} | {pf_str:<10} | {tr_str:<10} | {wr_str}")

    print("=" * 85)


if __name__ == "__main__":
    run_threshold_sensitivity()
