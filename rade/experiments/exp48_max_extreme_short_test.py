"""
[실험 48] 극한의 풀베팅 (추세 4.0% x 횡보 4.0%) + 비대칭 80% 숏 결합 백테스트
- 모델 1: EXTREME_CASH (추세 4.0% x 횡보 4.0%, 현금 관망)
- 모델 2: EXTREME_SHORT (추세 4.0% x 횡보 4.0% + 비대칭 BEAR TH=0.80 숏)
- 모델 3: STANDARD_GOLDEN (공식 표준: 추세 2.0% x 횡보 4.0%, 현금 관망)
- 목적: RADE 시스템의 이론상 한계 수익률 및 최대 낙폭 극한치 실측
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


def run_experiment_48():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    print("=" * 115)
    print("      [실험 48] 극한의 풀베팅 (추세 4.0% x 횡보 4.0%) + 비대칭 80% 숏 결합 백테스트")
    print("=" * 115)

    # 1. 국면 데이터 생성
    reg_mgr_74 = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.74, cooldown_bars=0)
    df_proc_74 = reg_mgr_74.calculate_regime_probabilities(df_ind)
    test_df_cash = df_proc_74.iloc[720:].reset_index(drop=True)

    df_asym = get_asymmetric_df(df_ind, base_th=0.74, bear_th=0.80)
    test_df_asym = df_asym.iloc[720:].reset_index(drop=True)

    # 모델 1: STANDARD_GOLDEN (공식 표준: 2.0% x 4.0%, CASH)
    sim_m1 = BacktestSimulator(
        initial_capital=10000.0, trend_risk_pct=0.020, mr_risk_pct=0.040, leverage=3.0, bear_mode="CASH",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_m1 = sim_m1.run(test_df_cash)

    # 모델 2: EXTREME_CASH (풀베팅 4.0% x 4.0%, CASH)
    sim_m2 = BacktestSimulator(
        initial_capital=10000.0, trend_risk_pct=0.040, mr_risk_pct=0.040, leverage=3.0, bear_mode="CASH",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_m2 = sim_m2.run(test_df_cash)

    # 모델 3: EXTREME_SHORT (풀베팅 4.0% x 4.0% + 비대칭 80% 숏)
    sim_m3 = BacktestSimulator(
        initial_capital=10000.0, trend_risk_pct=0.040, mr_risk_pct=0.040, leverage=3.0, bear_mode="SHORT",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_m3 = sim_m3.run(test_df_asym)

    # 1:1 비교표 출력
    models = [
        ("① STANDARD_GOLDEN (공식표준)", res_m1),
        ("② EXTREME_CASH (4%x4% 관망)", res_m2),
        ("③ EXTREME_SHORT (4%x4% 숏결합)", res_m3)
    ]

    print("\n[1. 세 가지 모델 1:1 종합 비교표]")
    print(f"{'지표 항목':<22} | {'① STANDARD_GOLDEN':<28} | {'② EXTREME_CASH (관망)':<28} | {'③ EXTREME_SHORT (숏결합)':<28}")
    print("-" * 115)

    def fmt_pnl(res): return f"+${res['final_equity']-10000:,.2f} (+{res['total_return_pct']:.2f}%)"
    def fmt_mdd(res): return f"{res['mdd_pct']:.2f}%"
    def fmt_pf(res): return f"{res['profit_factor']:.2f}"
    def fmt_wr(res): return f"{res['win_rate_pct']:.1f}%"
    def fmt_tr(res): return f"{res['total_trades']}회 (연 {res['total_trades']/3.92:.1f}회)"
    def fmt_cal(res): return f"{res['total_return_pct']/res['mdd_pct']:.2f}"

    print(f"{'4개년 총 수익금':<22} | {fmt_pnl(res_m1):<28} | {fmt_pnl(res_m2):<28} | {fmt_pnl(res_m3):<28}")
    print(f"{'최대 낙폭 (MDD)':<22} | {fmt_mdd(res_m1):<28} | {fmt_mdd(res_m2):<28} | {fmt_mdd(res_m3):<28}")
    print(f"{'칼마 비율 (수익÷MDD)':<22} | {fmt_cal(res_m1):<28} | {fmt_cal(res_m2):<28} | {fmt_cal(res_m3):<28}")
    print(f"{'손익비 (PF)':<22} | {fmt_pf(res_m1):<28} | {fmt_pf(res_m2):<28} | {fmt_pf(res_m3):<28}")
    print(f"{'전체 승률 (Win Rate)':<22} | {fmt_wr(res_m1):<28} | {fmt_wr(res_m2):<28} | {fmt_wr(res_m3):<28}")
    print(f"{'총 거래 횟수':<22} | {fmt_tr(res_m1):<28} | {fmt_tr(res_m2):<28} | {fmt_tr(res_m3):<28}")
    print("-" * 115)

    # 2. 연도별 PnL 비교
    print("\n[2. 연도별 PnL 분해 비교]")
    for name, r in models:
        t_df = r["trades_df"].copy()
        t_df["year"] = pd.to_datetime(t_df["entry_time"]).dt.year
        p21 = t_df[t_df["year"] == 2021]["pnl"].sum() if len(t_df[t_df["year"] == 2021]) > 0 else 0.0
        p22 = t_df[t_df["year"] == 2022]["pnl"].sum() if len(t_df[t_df["year"] == 2022]) > 0 else 0.0
        p23 = t_df[t_df["year"] == 2023]["pnl"].sum() if len(t_df[t_df["year"] == 2023]) > 0 else 0.0
        p24 = t_df[t_df["year"] == 2024]["pnl"].sum() if len(t_df[t_df["year"] == 2024]) > 0 else 0.0
        print(f" * {name:<26} -> 2021: {p21:>+10.2f}$ | 2022(하락): {p22:>+10.2f}$ | 2023: {p23:>+10.2f}$ | 2024(불장): {p24:>+10.2f}$")


if __name__ == "__main__":
    run_experiment_48()
