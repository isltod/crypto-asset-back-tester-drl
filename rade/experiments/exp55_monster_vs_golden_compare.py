"""
[실험 55] RADE 최신 표준(v3: TF 1.0% x MR 8.0%) vs 몬스터 극한(TF 4.0% x MR 16.0% + 80% 숏, 100x) 정밀 비교
"""
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

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


def run_compare():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    # 1. 표준 모델 (STANDARD_GOLDEN_v3: 1.0% x 8.0%, CASH, 3.0x)
    reg_mgr_74 = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.74, cooldown_bars=0)
    df_proc_74 = reg_mgr_74.calculate_regime_probabilities(df_ind)
    test_df_cash = df_proc_74.iloc[720:].reset_index(drop=True)

    sim_m1 = BacktestSimulator(
        initial_capital=10000.0, trend_risk_pct=0.010, mr_risk_pct=0.080, leverage=3.0, bear_mode="CASH",
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_m1 = sim_m1.run(test_df_cash)

    # 2. 극한 몬스터 모델 (TF 4.0% x MR 16.0% + 80% 숏, 100.0x)
    df_asym = get_asymmetric_df(df_ind, base_th=0.74, bear_th=0.80)
    test_df_asym = df_asym.iloc[720:].reset_index(drop=True)

    sim_m2 = BacktestSimulator(
        initial_capital=10000.0, trend_risk_pct=0.040, mr_risk_pct=0.160, leverage=100.0, bear_mode="SHORT",
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    sim_m2.pos_manager.max_leverage = 100.0
    sim_m2.pos_manager.default_leverage = 100.0
    res_m2 = sim_m2.run(test_df_asym)

    def get_yearly_stats(res):
        t = res["trades_df"].copy()
        t["entry_dt"] = pd.to_datetime(t["entry_time"])
        t["year"] = t["entry_dt"].dt.year
        yearly_pnl = {}
        for y in [2021, 2022, 2023, 2024]:
            sub = t[t["year"] == y]
            yearly_pnl[y] = sub["pnl"].sum() if len(sub) > 0 else 0.0
        return yearly_pnl

    y_m1 = get_yearly_stats(res_m1)
    y_m2 = get_yearly_stats(res_m2)

    print("\n" + "=" * 105)
    print("      [ STANDARD_GOLDEN_v3 (공식 표준) vs MONSTER_EXTREME_100X (수익률 1위 몬스터) 1:1 비교 ]")
    print("=" * 105)

    def fmt_pnl(res): return f"+${res['final_equity']-10000:,.2f} (+{res['total_return_pct']:.2f}%)"
    def fmt_cagr(res): 
        years = 3.92
        cagr = ((res['final_equity'] / 10000.0) ** (1.0 / years) - 1.0) * 100.0
        return f"연 {cagr:.1f}%"
    def fmt_mdd(res): return f"{res['mdd_pct']:.2f}%"
    def fmt_cal(res): return f"{res['total_return_pct']/res['mdd_pct']:.2f}"
    def fmt_pf(res): return f"{res['profit_factor']:.2f}"
    def fmt_wr(res): return f"{res['win_rate_pct']:.1f}%"
    def fmt_tr(res): return f"{res['total_trades']}회 (연 {res['total_trades']/3.92:.1f}회)"

    print(f"{'지표 항목':<25} | {'🌟 STANDARD_GOLDEN_v3 (3.0x)':<35} | {'💥 MONSTER_EXTREME (100.0x)':<35}")
    print("-" * 105)
    print(f"{'4개년 총 수익금 (수익률)':<25} | {fmt_pnl(res_m1):<35} | {fmt_pnl(res_m2):<35}")
    print(f"{'연평균 복리 수익률 (CAGR)':<25} | {fmt_cagr(res_m1):<35} | {fmt_cagr(res_m2):<35}")
    print(f"{'실전 최대 낙폭 (MDD)':<25} | {fmt_mdd(res_m1):<35} | {fmt_mdd(res_m2):<35}")
    print(f"{'칼마 비율 (수익÷MDD)':<25} | {fmt_cal(res_m1):<35} | {fmt_cal(res_m2):<35}")
    print(f"{'손익비 (Profit Factor)':<25} | {fmt_pf(res_m1):<35} | {fmt_pf(res_m2):<35}")
    print(f"{'전체 계좌 승률':<25} | {fmt_wr(res_m1):<35} | {fmt_wr(res_m2):<35}")
    print(f"{'총 거래 횟수':<25} | {fmt_tr(res_m1):<35} | {fmt_tr(res_m2):<35}")
    print("-" * 105)
    print(" [ 연도별 손익 분해 ($) ]")
    for y in [2021, 2022, 2023, 2024]:
        print(f" * {y}년 손익: 표준 {y_m1[y]:+10.2f}$ | 몬스터 {y_m2[y]:+12.2f}$")


if __name__ == "__main__":
    run_compare()
