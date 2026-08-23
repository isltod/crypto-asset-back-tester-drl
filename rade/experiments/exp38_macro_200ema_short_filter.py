"""
[실험 38] 거시 200일선(Macro 200 EMA) 기반 하락 추세 숏(TF Short) 필터링 백테스트
- 규칙:
  1. BULL_TREND: 추세 롱 (Trend Long)
  2. RANGE: 기존 평균회귀 (MR Long / Short)
  3. BEAR_PANIC:
     - Close < Macro 200 EMA (200일선 아래 대세 약세장): 추세 숏(TF Short) 실행
     - Close >= Macro 200 EMA (200일선 위 대세 강세장): 추세 숏 금지 (현금 관망)
- 비교 대상:
  - [1] 현금 관망 (CASH 모드)
  - [2] 무조건 숏 배팅 (exp36 SHORT 모드)
  - [3] 200일선 거시 필터 숏 배팅 (exp38 Macro SHORT 모드)
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


def run_experiment_38():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    # 200일선 (1시간봉 기준 200 * 24 = 4800봉 EMA) 및 200봉 EMA(8.3일선) 추가
    df_ind["ema_200d"] = df_ind["close"].ewm(span=4800, adjust=False).mean()

    reg_mgr = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.74, cooldown_bars=0)
    df_proc = reg_mgr.calculate_regime_probabilities(df_ind)
    test_df = df_proc.iloc[720:].reset_index(drop=True)

    print("=" * 95)
    print("      [실험 38] 200일선 거시 필터 적용 숏 배팅 정밀 백테스트 (TH=0.74)")
    print("=" * 95)

    # 1. CASH 모드
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

    # 2. 무조건 숏 (SHORT 모드)
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

    # 3. 200일선 거시 필터 숏 모드 (Close < 200일선 일 때만 BEAR_PANIC 유지, 200일선 위면 관망)
    records_macro = test_df.to_dict("records")
    for r in records_macro:
        if r["regime_state"] == RegimeState.BEAR_PANIC and r["close"] >= r["ema_200d"]:
            r["regime_state"] = "BEAR_CASH" # 200일선 위 하락 국면은 관망 처리

    df_macro = pd.DataFrame(records_macro)
    sim_macro = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        bear_mode="SHORT",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_macro = sim_macro.run(df_macro)

    # 결과 비교 출력
    print("\n[1. 세 가지 운용 모델 1:1 비교표]")
    print(f"{'지표 항목':<20} | {'① 현금관망 (CASH)':<20} | {'② 무조건 숏 (SHORT)':<20} | {'③ 200일선 필터 숏 (Macro SHORT)':<25}")
    print("-" * 95)
    
    p1 = f"+${res_cash['final_equity']-10000:,.2f} (+{res_cash['total_return_pct']:.2f}%)"
    p2 = f"+${res_short['final_equity']-10000:,.2f} (+{res_short['total_return_pct']:.2f}%)"
    p3 = f"+${res_macro['final_equity']-10000:,.2f} (+{res_macro['total_return_pct']:.2f}%)"

    m1 = f"{res_cash['mdd_pct']:.2f}%"
    m2 = f"{res_short['mdd_pct']:.2f}%"
    m3 = f"{res_macro['mdd_pct']:.2f}%"

    pf1 = f"{res_cash['profit_factor']:.2f}"
    pf2 = f"{res_short['profit_factor']:.2f}"
    pf3 = f"{res_macro['profit_factor']:.2f}"

    wr1 = f"{res_cash['win_rate_pct']:.1f}%"
    wr2 = f"{res_short['win_rate_pct']:.1f}%"
    wr3 = f"{res_macro['win_rate_pct']:.1f}%"

    tr1 = f"{res_cash['total_trades']}회"
    tr2 = f"{res_short['total_trades']}회"
    tr3 = f"{res_macro['total_trades']}회"

    print(f"{'4개년 총 수익금':<20} | {p1:<20} | {p2:<20} | {p3:<25}")
    print(f"{'최대 낙폭 (MDD)':<20} | {m1:<20} | {m2:<20} | {m3:<25}")
    print(f"{'손익비 (PF)':<20} | {pf1:<20} | {pf2:<20} | {pf3:<25}")
    print(f"{'전체 승률 (Win Rate)':<20} | {wr1:<20} | {wr2:<20} | {wr3:<25}")
    print(f"{'총 거래 횟수':<20} | {tr1:<20} | {tr2:<20} | {tr3:<25}")
    print("-" * 95)

    # 연도별 PnL 분해
    print("\n[2. 연도별 PnL 분해 비교]")
    t_c = res_cash["trades_df"].copy()
    t_s = res_short["trades_df"].copy()
    t_m = res_macro["trades_df"].copy()
    
    t_c["year"] = pd.to_datetime(t_c["entry_time"]).dt.year
    t_s["year"] = pd.to_datetime(t_s["entry_time"]).dt.year
    t_m["year"] = pd.to_datetime(t_m["entry_time"]).dt.year

    for yr in [2021, 2022, 2023, 2024]:
        yc = t_c[t_c["year"] == yr]["pnl"].sum()
        ys = t_s[t_s["year"] == yr]["pnl"].sum()
        ym = t_m[t_m["year"] == yr]["pnl"].sum()
        print(f" * {yr}년: 현금관망(CASH) -> {yc:+9.2f}$ | 무조건숏 -> {ys:+9.2f}$ | 200일선필터 -> {ym:+9.2f}$")

    print("=" * 95)


if __name__ == "__main__":
    run_experiment_38()
