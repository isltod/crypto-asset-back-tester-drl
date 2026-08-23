"""
[실험 33] 기존 베이스라인(TH=0.45) vs 신규 최적 안정 고원(TH=0.74) 1:1 정밀 비교
- 엔진별(MR vs TF) 거래 횟수, 승률, 손익 기여금
- 연도별 PnL 및 MDD 상세 분해
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


def compare_th_045_vs_074():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    # 1. TH = 0.45 (기존 베이스라인)
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

    # 2. TH = 0.74 (신규 최적 고원 정중앙)
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

    # 3. 데이터 분석 및 출력
    t_045 = res_045["trades_df"]
    t_074 = res_074["trades_df"]

    print("=" * 85)
    print("      [실험 33] 기존 TH=0.45 vs 신규 TH=0.74 1:1 정밀 분해 리포트")
    print("=" * 85)

    print("\n[1. 전체 핵심 지표 비교]")
    print(f"{'지표 항목':<22} | {'기존 베이스라인 (TH=0.45)':<25} | {'신규 고원 최적화 (TH=0.74)':<25}")
    print("-" * 80)
    pnl_045_str = f"+${res_045['final_equity']-10000:,.2f} (+{res_045['total_return_pct']:.2f}%)"
    pnl_074_str = f"+${res_074['final_equity']-10000:,.2f} (+{res_074['total_return_pct']:.2f}%)"
    mdd_045_str = f"{res_045['mdd_pct']:.2f}%"
    mdd_074_str = f"{res_074['mdd_pct']:.2f}% (개선!)"
    pf_045_str = f"{res_045['profit_factor']:.2f}"
    pf_074_str = f"{res_074['profit_factor']:.2f} (개선!)"
    wr_045_str = f"{res_045['win_rate_pct']:.1f}%"
    wr_074_str = f"{res_074['win_rate_pct']:.1f}%"
    tr_045_str = f"{res_045['total_trades']}회 (연 {res_045['total_trades']/3.92:.1f}회)"
    tr_074_str = f"{res_074['total_trades']}회 (연 {res_074['total_trades']/3.92:.1f}회)"
    avg_045_str = f"+${t_045['pnl'].mean():.2f}"
    avg_074_str = f"+${t_074['pnl'].mean():.2f} (개선!)"

    print(f"{'4개년 총 수익금':<22} | {pnl_045_str:<25} | {pnl_074_str:<25}")
    print(f"{'최대 낙폭 (MDD)':<22} | {mdd_045_str:<25} | {mdd_074_str:<25}")
    print(f"{'손익비 (PF)':<22} | {pf_045_str:<25} | {pf_074_str:<25}")
    print(f"{'전체 승률 (Win Rate)':<22} | {wr_045_str:<25} | {wr_074_str:<25}")
    print(f"{'총 거래 횟수':<22} | {tr_045_str:<25} | {tr_074_str:<25}")
    print(f"{'건당 평균 이익':<22} | {avg_045_str:<25} | {avg_074_str:<25}")
    print("-" * 80)

    print("\n[2. 엔진별 세부 실적 (MR vs TF) 1:1 비교]")
    for eng in ["MEAN_REVERSION", "TREND_FOLLOWING"]:
        sub_045 = t_045[t_045["engine"] == eng]
        sub_074 = t_074[t_074["engine"] == eng]
        
        pnl_045 = sub_045["pnl"].sum()
        pnl_074 = sub_074["pnl"].sum()
        
        wr_045 = len(sub_045[sub_045["pnl"] > 0]) / len(sub_045) * 100.0 if len(sub_045) > 0 else 0.0
        wr_074 = len(sub_074[sub_074["pnl"] > 0]) / len(sub_074) * 100.0 if len(sub_074) > 0 else 0.0

        print(f" * [{eng}]")
        print(f"   - TH=0.45: {len(sub_045):3d}회 거래 | 승률 {wr_045:5.1f}% | 수익 기여: {pnl_045:+10.2f}$ (건당 {pnl_045/len(sub_045):+.2f}$)")
        print(f"   - TH=0.74: {len(sub_074):3d}회 거래 | 승률 {wr_074:5.1f}% | 수익 기여: {pnl_074:+10.2f}$ (건당 {pnl_074/len(sub_074):+.2f}$)")

    print("\n[3. 연도별 PnL 비교]")
    t_045["year"] = pd.to_datetime(t_045["entry_time"]).dt.year
    t_074["year"] = pd.to_datetime(t_074["entry_time"]).dt.year
    for yr in [2021, 2022, 2023, 2024]:
        y_045 = t_045[t_045["year"] == yr]["pnl"].sum()
        y_074 = t_074[t_074["year"] == yr]["pnl"].sum()
        print(f" * {yr}년: TH=0.45 -> {y_045:+10.2f}$  |  TH=0.74 -> {y_074:+10.2f}$ ({y_074 - y_045:+8.2f}$ 차이)")
    print("=" * 85)


if __name__ == "__main__":
    compare_th_045_vs_074()
