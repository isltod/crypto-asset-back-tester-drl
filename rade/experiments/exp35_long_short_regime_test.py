"""
[실험 35] 3-State 국면 롱/숏 양방향 배팅 정밀 백테스트
- 조건:
  1. 3개 국면 분리 (RANGE, BULL_TREND, BEAR_PANIC)
  2. 횡보 국면 (RANGE): 평균회귀(MR) 롱/숏 거래
  3. 상승 국면 (BULL_TREND): 추세 롱 (Trend Long)
  4. 하락 국면 (BEAR_PANIC): 추세 숏 (Trend Short - bear_mode="SHORT")
- 검증 대상:
  - TH=0.45 vs TH=0.74에서 숏 배팅의 유효성 검증
  - 연도별(2021~2024), 국면별(RANGE, BULL, BEAR), 포지션 방향별(LONG, SHORT) 세부 성과 집계
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


def run_experiment_35():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    for th in [0.45, 0.74]:
        print("\n" + "=" * 95)
        print(f"      [실험 35] 3-State 롱/숏 양방향 가동 백테스트 (HMM TH = {th})")
        print("=" * 95)

        reg_mgr = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=th, cooldown_bars=0)
        df_proc = reg_mgr.calculate_regime_probabilities(df_ind)
        test_df = df_proc.iloc[720:].reset_index(drop=True)

        # 1. CASH 모드 (기존 롱 전용)
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

        # 2. SHORT 모드 (하락장 추세 숏 배팅 활성화)
        sim_short = BacktestSimulator(
            initial_capital=10000.0,
            risk_per_trade_pct=0.02,
            leverage=3.0,
            bear_mode="SHORT",
            use_regime_transition_cut=False,
            trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
            mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
        )
        res_short = sim_short.run(test_df)

        # 요약 비교
        print("\n[1. 하락장 현금관망(CASH) vs 하락장 숏배팅(SHORT) 1:1 비교]")
        print(f"{'구분':<20} | {'하락장 현금관망 (기존 CASH)':<25} | {'하락장 숏배팅 (신규 SHORT)':<25}")
        print("-" * 80)
        c_pnl = f"+${res_cash['final_equity']-10000:,.2f} (+{res_cash['total_return_pct']:.2f}%)"
        s_pnl = f"+${res_short['final_equity']-10000:,.2f} (+{res_short['total_return_pct']:.2f}%)"
        c_mdd = f"{res_cash['mdd_pct']:.2f}%"
        s_mdd = f"{res_short['mdd_pct']:.2f}%"
        c_pf = f"{res_cash['profit_factor']:.2f}"
        s_pf = f"{res_short['profit_factor']:.2f}"
        c_wr = f"{res_cash['win_rate_pct']:.1f}%"
        s_wr = f"{res_short['win_rate_pct']:.1f}%"
        c_tr = f"{res_cash['total_trades']}회"
        s_tr = f"{res_short['total_trades']}회"

        print(f"{'4개년 총 수익금':<20} | {c_pnl:<25} | {s_pnl:<25}")
        print(f"{'최대 낙폭 (MDD)':<20} | {c_mdd:<25} | {s_mdd:<25}")
        print(f"{'손익비 (PF)':<20} | {c_pf:<25} | {s_pf:<25}")
        print(f"{'승률 (Win Rate)':<20} | {c_wr:<25} | {s_wr:<25}")
        print(f"{'총 거래 횟수':<20} | {c_tr:<25} | {s_tr:<25}")
        print("-" * 80)

        # 3. SHORT 모드 세부 분해 분석
        trades = res_short["trades_df"].copy()
        trades["entry_dt"] = pd.to_datetime(trades["entry_time"])
        trades["year"] = trades["entry_dt"].dt.year

        print(f"\n[2. SHORT 모드 포지션 방향별(LONG vs SHORT) 세부 실적 (TH={th})]")
        for side in ["LONG", "SHORT"]:
            sub = trades[trades["side"].astype(str).str.contains(side)]
            p = sub["pnl"].sum()
            w = len(sub[sub["pnl"] > 0]) / len(sub) * 100.0 if len(sub) > 0 else 0.0
            print(f" * [{side:<5}] 거래: {len(sub):3d}회 | 승률: {w:5.1f}% | 총 PnL: {p:+10.2f}$ (건당 {p/len(sub) if len(sub)>0 else 0:+.2f}$)")

        print(f"\n[3. SHORT 모드 국면별(RANGE vs BULL vs BEAR) 세부 실적 (TH={th})]")
        for eng in trades["engine"].unique():
            sub = trades[trades["engine"] == eng]
            p = sub["pnl"].sum()
            w = len(sub[sub["pnl"] > 0]) / len(sub) * 100.0 if len(sub) > 0 else 0.0
            print(f" * [{eng:<16}] 거래: {len(sub):3d}회 | 승률: {w:5.1f}% | 총 PnL: {p:+10.2f}$")

        print(f"\n[4. SHORT 모드 연도별 성과 분해 (TH={th})]")
        for yr in [2021, 2022, 2023, 2024]:
            sub = trades[trades["year"] == yr]
            p = sub["pnl"].sum()
            w = len(sub[sub["pnl"] > 0]) / len(sub) * 100.0 if len(sub) > 0 else 0.0
            sub_l = sub[sub["side"].astype(str).str.contains("LONG")]
            sub_s = sub[sub["side"].astype(str).str.contains("SHORT")]
            print(f" * {yr}년: 총 {p:+10.2f}$ ({len(sub):2d}회, 승률 {w:5.1f}%) | 롱: {sub_l['pnl'].sum():+9.2f}$ ({len(sub_l):2d}회) | 숏: {sub_s['pnl'].sum():+9.2f}$ ({len(sub_s):2d}회)")


if __name__ == "__main__":
    run_experiment_35()
