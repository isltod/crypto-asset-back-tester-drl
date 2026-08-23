"""
[실험 36] 국면별 순수 방향성 분리 (BULL=Long Only, BEAR=Short Only, RANGE=MR) 백테스트
- 규칙:
  1. 상승 국면 (BULL_TREND): 오직 추세 롱 (Trend Long Only - 숏 절대 금지)
  2. 하락 국면 (BEAR_PANIC): 오직 추세 숏 (Trend Short Only - 롱 절대 금지)
  3. 횡보 국면 (RANGE): 기존 평균회귀 (MR 양방향 롱/숏)
- 검증:
  - HMM TH=0.74 및 TH=0.45 기준 실행
  - [1] 기존 CASH 모드 (롱 전용) vs [2] exp35 SHORT 모드 vs [3] exp36 순수 국면 방향성 모드 1:1 비교
  - 연도별, 국면별, 포지션 방향별 상세 집계
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager, RegimeState
from rade.backtest.simulator import BacktestSimulator
from rade.engines.trend_following import TrendFollowingEngine
from rade.engines.mean_reversion import MeanReversionEngine


def run_experiment_36():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    print("=" * 95)
    print("      [실험 36] 국면별 엄격한 방향성 분리 (BULL=Long, BEAR=Short, RANGE=MR) 백테스트")
    print("=" * 95)

    for th in [0.74, 0.45]:
        reg_mgr = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=th, cooldown_bars=0)
        df_proc = reg_mgr.calculate_regime_probabilities(df_ind)
        test_df = df_proc.iloc[720:].reset_index(drop=True)

        # 1. CASH 모드 (롱 전용)
        sim_cash = BacktestSimulator(
            initial_capital=10000.0,
            risk_per_trade_pct=0.02,
            leverage=3.0,
            bear_mode="CASH",
            use_regime_transition_cut=False,
            trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
            mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
        )
        res_cash = sim_cash.run(test_df)

        # 2. 순수 국면 분리 모드 (BULL=Trend Long, BEAR=Trend Short, RANGE=MR)
        sim_pure = BacktestSimulator(
            initial_capital=10000.0,
            risk_per_trade_pct=0.02,
            leverage=3.0,
            bear_mode="SHORT", # BEAR_PANIC에서 숏 실행
            use_regime_transition_cut=False,
            trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
            mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
        )
        res_pure = sim_pure.run(test_df)

        trades_pure = res_pure["trades_df"].copy()
        trades_pure["entry_dt"] = pd.to_datetime(trades_pure["entry_time"])
        trades_pure["year"] = trades_pure["entry_dt"].dt.year

        print(f"\n====================== [ HMM TH = {th} 종합 비교 ] ======================")
        print(f"{'구분':<20} | {'① 현금관망 (CASH)':<22} | {'② 순수 방향성 분리 (SHORT)':<25}")
        print("-" * 80)
        pnl_c = f"+${res_cash['final_equity']-10000:,.2f} (+{res_cash['total_return_pct']:.2f}%)"
        pnl_p = f"+${res_pure['final_equity']-10000:,.2f} (+{res_pure['total_return_pct']:.2f}%)"
        mdd_c = f"{res_cash['mdd_pct']:.2f}%"
        mdd_p = f"{res_pure['mdd_pct']:.2f}%"
        pf_c = f"{res_cash['profit_factor']:.2f}"
        pf_p = f"{res_pure['profit_factor']:.2f}"
        wr_c = f"{res_cash['win_rate_pct']:.1f}%"
        wr_p = f"{res_pure['win_rate_pct']:.1f}%"
        tr_c = f"{res_cash['total_trades']}회 (연 {res_cash['total_trades']/3.92:.1f}회)"
        tr_p = f"{res_pure['total_trades']}회 (연 {res_pure['total_trades']/3.92:.1f}회)"

        print(f"{'4개년 총 수익금':<20} | {pnl_c:<22} | {pnl_p:<25}")
        print(f"{'최대 낙폭 (MDD)':<20} | {mdd_c:<22} | {mdd_p:<25}")
        print(f"{'손익비 (PF)':<20} | {pf_c:<22} | {pf_p:<25}")
        print(f"{'전체 승률 (Win Rate)':<20} | {wr_c:<22} | {wr_p:<25}")
        print(f"{'총 거래 횟수':<20} | {tr_c:<22} | {tr_p:<25}")
        print("-" * 80)

        # 국면별 x 포지션 세부 분해
        print(f"\n[국면별 세부 실적 분해 (TH={th})]")
        # 1) BULL 국면 (Trend Long)
        bull_trades = trades_pure[trades_pure["engine"] == "TREND_FOLLOWING"]
        bull_long = bull_trades[bull_trades["side"].astype(str).str.contains("LONG")]
        bull_short = bull_trades[bull_trades["side"].astype(str).str.contains("SHORT")]

        print(f" * [상승 국면 (BULL_TREND) - Trend Long] : {bull_long['pnl'].sum():+9.2f}$ | {len(bull_long):2d}회 | 승률 {len(bull_long[bull_long['pnl']>0])/len(bull_long)*100 if len(bull_long)>0 else 0:5.1f}%")
        print(f" * [하락 국면 (BEAR_PANIC) - Trend Short]: {bull_short['pnl'].sum():+9.2f}$ | {len(bull_short):2d}회 | 승률 {len(bull_short[bull_short['pnl']>0])/len(bull_short)*100 if len(bull_short)>0 else 0:5.1f}%")
        
        # 2) RANGE 국면 (MR)
        mr_trades = trades_pure[trades_pure["engine"] == "MEAN_REVERSION"]
        mr_long = mr_trades[mr_trades["side"].astype(str).str.contains("LONG")]
        mr_short = mr_trades[mr_trades["side"].astype(str).str.contains("SHORT")]
        print(f" * [횡보 국면 (RANGE)      - MR Long]    : {mr_long['pnl'].sum():+9.2f}$ | {len(mr_long):2d}회 | 승률 {len(mr_long[mr_long['pnl']>0])/len(mr_long)*100 if len(mr_long)>0 else 0:5.1f}%")
        print(f" * [횡보 국면 (RANGE)      - MR Short]   : {mr_short['pnl'].sum():+9.2f}$ | {len(mr_short):2d}회 | 승률 {len(mr_short[mr_short['pnl']>0])/len(mr_short)*100 if len(mr_short)>0 else 0:5.1f}%")

        print(f"\n[연도별 PnL 분해 (TH={th})]")
        t_cash = res_cash["trades_df"].copy()
        t_cash["entry_dt"] = pd.to_datetime(t_cash["entry_time"])
        t_cash["year"] = t_cash["entry_dt"].dt.year

        for yr in [2021, 2022, 2023, 2024]:
            c_yr = t_cash[t_cash["year"] == yr]["pnl"].sum()
            p_yr = trades_pure[trades_pure["year"] == yr]["pnl"].sum()
            diff = p_yr - c_yr
            print(f" * {yr}년: 현금관망(CASH) -> {c_yr:+9.2f}$  |  순수방향성(SHORT) -> {p_yr:+9.2f}$ ({diff:+8.2f}$ 차이)")

    print("=" * 95)


if __name__ == "__main__":
    run_experiment_36()
