"""
[실험 46] 신규 공식 표준(STANDARD_GOLDEN: 추세 2.0% x 횡보 4.0%) + 비대칭 80% 숏 결합 정밀 백테스트
- 모델 1: STANDARD_GOLDEN (공식 기본값: 추세 2.0% x 횡보 4.0%, CASH 모드, TH=0.74)
- 모델 2: GOLDEN_SHORT_2.0 (신규 결합: 추세 2.0% x 횡보 4.0% + 비대칭 BEAR TH=0.80 숏)
- 모델 3: 기존 결합 비교군 (2.5% x 4.0% + 비대칭 80% 숏)
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


def get_asymmetric_df(df_ind: pd.DataFrame, base_th: float = 0.74, bear_th: float = 0.80) -> pd.DataFrame:
    reg_raw = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.30, cooldown_bars=0)
    df_raw = reg_raw.calculate_regime_probabilities(df_ind)
    
    curr = RegimeState.RANGE
    asym_states = []
    for idx, row in df_raw.iterrows():
        p_r = row["p_range"]
        p_u = row["p_bull"]
        p_d = row["p_bear"]
        if p_d >= bear_th and p_d >= p_u and p_d >= p_r:
            curr = RegimeState.BEAR_PANIC
        elif p_u >= base_th and p_u >= p_r and p_u >= p_d:
            curr = RegimeState.BULL_TREND
        elif p_r >= base_th and p_r >= p_u and p_r >= p_d:
            curr = RegimeState.RANGE
        asym_states.append(curr)
    df_raw["regime_state"] = asym_states
    return df_raw


def run_experiment_46():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    print("=" * 105)
    print("      [실험 46] 신규 공식 표준(추세 2.0% x 횡보 4.0%) + 비대칭 80% 숏 결합 백테스트")
    print("=" * 105)

    # 1. 모델 1: STANDARD_GOLDEN (CASH 모드, 2.0% x 4.0%)
    reg_mgr_74 = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.74, cooldown_bars=0)
    df_proc_74 = reg_mgr_74.calculate_regime_probabilities(df_ind)
    test_df_m1 = df_proc_74.iloc[720:].reset_index(drop=True)

    sim_m1 = BacktestSimulator(
        initial_capital=10000.0,
        trend_risk_pct=0.020,
        mr_risk_pct=0.040,
        leverage=3.0,
        bear_mode="CASH",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_m1 = sim_m1.run(test_df_m1)

    # 2. 비대칭 국면 데이터 생성 (BEAR TH = 0.80)
    df_asym = get_asymmetric_df(df_ind, base_th=0.74, bear_th=0.80)
    test_df_asym = df_asym.iloc[720:].reset_index(drop=True)

    # 모델 2: 신규 결합 GOLDEN_SHORT_2.0 (추세 2.0% x 횡보 4.0% + 비대칭 80% 숏)
    sim_m2 = BacktestSimulator(
        initial_capital=10000.0,
        trend_risk_pct=0.020,
        mr_risk_pct=0.040,
        leverage=3.0,
        bear_mode="SHORT",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_m2 = sim_m2.run(test_df_asym)

    # 모델 3: 비교군 (추세 2.5% x 횡보 4.0% + 비대칭 80% 숏)
    sim_m3 = BacktestSimulator(
        initial_capital=10000.0,
        trend_risk_pct=0.025,
        mr_risk_pct=0.040,
        leverage=3.0,
        bear_mode="SHORT",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_m3 = sim_m3.run(test_df_asym)

    # 1:1 비교표 출력
    print("\n[1. 세 가지 모델 1:1 종합 비교표]")
    print(f"{'지표 항목':<22} | {'① STANDARD_GOLDEN (현금관망)':<30} | {'② 신규결합: GOLDEN_SHORT_2.0':<32} | {'③ 기존결합: GOLDEN_SHORT_2.5':<32}")
    print("-" * 125)

    def fmt_pnl(res): return f"+${res['final_equity']-10000:,.2f} (+{res['total_return_pct']:.2f}%)"
    def fmt_mdd(res): return f"{res['mdd_pct']:.2f}%"
    def fmt_pf(res): return f"{res['profit_factor']:.2f}"
    def fmt_wr(res): return f"{res['win_rate_pct']:.1f}%"
    def fmt_tr(res): return f"{res['total_trades']}회 (연 {res['total_trades']/3.92:.1f}회)"
    def fmt_cal(res): return f"{res['total_return_pct']/res['mdd_pct']:.2f}"

    print(f"{'4개년 총 수익금':<22} | {fmt_pnl(res_m1):<30} | {fmt_pnl(res_m2):<32} | {fmt_pnl(res_m3):<32}")
    print(f"{'최대 낙폭 (MDD)':<22} | {fmt_mdd(res_m1):<30} | {fmt_mdd(res_m2):<32} | {fmt_mdd(res_m3):<32}")
    print(f"{'칼마 비율 (수익÷MDD)':<22} | {fmt_cal(res_m1):<30} | {fmt_cal(res_m2):<32} | {fmt_cal(res_m3):<32}")
    print(f"{'손익비 (PF)':<22} | {fmt_pf(res_m1):<30} | {fmt_pf(res_m2):<32} | {fmt_pf(res_m3):<32}")
    print(f"{'전체 승률 (Win Rate)':<22} | {fmt_wr(res_m1):<30} | {fmt_wr(res_m2):<32} | {fmt_wr(res_m3):<32}")
    print(f"{'총 거래 횟수':<22} | {fmt_tr(res_m1):<30} | {fmt_tr(res_m2):<32} | {fmt_tr(res_m3):<32}")
    print("-" * 125)

    # 2. 연도별 PnL 비교
    print("\n[2. 연도별 PnL 분해 비교]")
    t1 = res_m1["trades_df"].copy()
    t2 = res_m2["trades_df"].copy()
    t3 = res_m3["trades_df"].copy()

    for t_df in [t1, t2, t3]:
        t_df["year"] = pd.to_datetime(t_df["entry_time"]).dt.year

    for yr in [2021, 2022, 2023, 2024]:
        p1 = t1[t1["year"] == yr]["pnl"].sum() if len(t1[t1["year"] == yr]) > 0 else 0.0
        p2 = t2[t2["year"] == yr]["pnl"].sum() if len(t2[t2["year"] == yr]) > 0 else 0.0
        p3 = t3[t3["year"] == yr]["pnl"].sum() if len(t3[t3["year"] == yr]) > 0 else 0.0
        print(f" * {yr}년: ① STANDARD_GOLDEN -> {p1:>+10.2f}$ | ② GOLDEN_SHORT_2.0 -> {p2:>+10.2f}$ | ③ GOLDEN_SHORT_2.5 -> {p3:>+10.2f}$")


if __name__ == "__main__":
    run_experiment_46()
