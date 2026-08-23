"""
[실험 39-2] 하락 국면 비대칭 고확신 임계값 (정밀 수정 버전)
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


def apply_asymmetric_thresholds(df_prob: pd.DataFrame, base_th: float = 0.74, bear_th: float = 0.85) -> pd.DataFrame:
    df = df_prob.copy()
    curr_state = RegimeState.RANGE
    states = []

    for idx, row in df.iterrows():
        p_r = row["p_range"]
        p_u = row["p_bull"]
        p_d = row["p_bear"]

        # 비대칭 전이 조건:
        # 하락장(BEAR_PANIC)으로 진입하려면 p_d >= bear_th (엄격한 고확신)
        # 상승장(BULL_TREND)으로 진입하려면 p_u >= base_th (0.74)
        # 횡보장(RANGE)으로 진입하려면 p_r >= base_th (0.74)
        if p_d >= bear_th and p_d >= p_u and p_d >= p_r:
            curr_state = RegimeState.BEAR_PANIC
        elif p_u >= base_th and p_u >= p_r and p_u >= p_d:
            curr_state = RegimeState.BULL_TREND
        elif p_r >= base_th and p_r >= p_u and p_r >= p_d:
            curr_state = RegimeState.RANGE
        # 문턱을 못 넘으면 이전 상태(curr_state) 유지

        states.append(curr_state)

    df["regime_state"] = states
    return df


def run_experiment_39_2():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    # 1. HMM 사후확률 기본 산출
    reg_mgr = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.30, cooldown_bars=0)
    df_raw_prob = reg_mgr.calculate_regime_probabilities(df_ind)

    print("=" * 95)
    print("      [실험 39] 하락 국면 비대칭 고확신 임계값 (BEAR TH = 0.74 ~ 0.95) 정밀 검증")
    print("=" * 95)

    bear_thresholds = [0.74, 0.80, 0.85, 0.90, 0.95]

    for b_th in bear_thresholds:
        df_asym = apply_asymmetric_thresholds(df_raw_prob, base_th=0.74, bear_th=b_th)
        test_df = df_asym.iloc[720:].reset_index(drop=True)

        sim = BacktestSimulator(
            initial_capital=10000.0,
            risk_per_trade_pct=0.02,
            leverage=3.0,
            bear_mode="SHORT",
            use_regime_transition_cut=False,
            trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
            mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
        )
        res = sim.run(test_df)
        t = res["trades_df"].copy()
        t["entry_dt"] = pd.to_datetime(t["entry_time"])
        t["year"] = t["entry_dt"].dt.year

        t_2022 = t[t["year"] == 2022]
        t_2024 = t[t["year"] == 2024]
        
        s_2022 = t_2022[t_2022["side"].astype(str).str.contains("SHORT")]
        s_2024 = t_2024[t_2024["side"].astype(str).str.contains("SHORT")]

        tf_s_2022 = s_2022[s_2022["engine"] == "TREND_FOLLOWING"]
        mr_s_2022 = s_2022[s_2022["engine"] == "MEAN_REVERSION"]
        tf_s_2024 = s_2024[s_2024["engine"] == "TREND_FOLLOWING"]
        mr_s_2024 = s_2024[s_2024["engine"] == "MEAN_REVERSION"]

        print(f"\n====================== [ BEAR TH = {b_th:.2f} (기본 TH=0.74) ] ======================")
        print(f" * 4개년 총 수익금: +${res['final_equity']-10000:,.2f} (+{res['total_return_pct']:.2f}%) | MDD: {res['mdd_pct']:.2f}% | PF: {res['profit_factor']:.2f} | 총 거래: {res['total_trades']}회")
        print(f" * 2022년(하락장) 숏 손익: {s_2022['pnl'].sum():+9.2f}$ | TF숏: {tf_s_2022['pnl'].sum():+8.2f}$ ({len(tf_s_2022)}회) | MR숏: {mr_s_2022['pnl'].sum():+8.2f}$ ({len(mr_s_2022)}회)")
        print(f" * 2024년(불장  ) 숏 손익: {s_2024['pnl'].sum():+9.2f}$ | TF숏: {tf_s_2024['pnl'].sum():+8.2f}$ ({len(tf_s_2024)}회) | MR숏: {mr_s_2024['pnl'].sum():+8.2f}$ ({len(mr_s_2024)}회)")


if __name__ == "__main__":
    run_experiment_39_2()
