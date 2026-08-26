"""
[실험 51] 극한의 횡보장 MR 리스크 [4%, 8%, 12%, 16%, 20%] x TF [2%, 4%, 6%, 8%, 10%] 25개 2D 전수 스캔
- 조건: HMM BEAR TH = 0.80, bear_mode = "SHORT"
- 추세장(TF) 리스크: [2.0%, 4.0%, 6.0%, 8.0%, 10.0%]
- 횡보장(MR) 리스크: [4.0%, 8.0%, 12.0%, 16.0%, 20.0%]
- 목적: 횡보장 고승률 엔진의 이론상 수익 한계점 및 레버리지 클램핑 임계선 규명
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


def run_experiment_51():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    # 비대칭 국면 데이터 생성 (BEAR TH = 0.80)
    df_asym = get_asymmetric_df(df_ind, base_th=0.74, bear_th=0.80)
    test_df = df_asym.iloc[720:].reset_index(drop=True)

    print("=" * 105)
    print("      [실험 51] 극한의 횡보장 MR [4%~20%] x TF [2%~10%] 25개 조합 2D 전수 스캔 (80% 숏)")
    print("=" * 105)

    tf_levels = [0.02, 0.04, 0.06, 0.08, 0.10]
    mr_levels = [0.04, 0.08, 0.12, 0.16, 0.20]
    results = []

    for t_risk in tf_levels:
        for m_risk in mr_levels:
            sim = BacktestSimulator(
                initial_capital=10000.0,
                trend_risk_pct=t_risk,
                mr_risk_pct=m_risk,
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

            calmar = res["total_return_pct"] / res["mdd_pct"] if res["mdd_pct"] > 0 else 0
            y2022_pnl = t[t["year"] == 2022]["pnl"].sum() if len(t[t["year"] == 2022]) > 0 else 0.0

            results.append({
                "trend_risk": round(t_risk * 100, 1),
                "mr_risk": round(m_risk * 100, 1),
                "final_equity": res["final_equity"],
                "total_return_pct": res["total_return_pct"],
                "profit_dollars": res["final_equity"] - 10000.0,
                "mdd_pct": res["mdd_pct"],
                "calmar_ratio": calmar,
                "profit_factor": res["profit_factor"],
                "win_rate_pct": res["win_rate_pct"],
                "total_trades": res["total_trades"],
                "y2022_pnl": y2022_pnl,
            })

    df_res = pd.DataFrame(results)

    # 1. 2D 매트릭스: 4개년 총 수익률 (%)
    pnl_matrix = df_res.pivot(index="trend_risk", columns="mr_risk", values="total_return_pct")
    print("\n[1. 4개년 총 수익률 매트릭스 (%) - 행: 추세장(TF) / 열: 횡보장(MR)]")
    print(pnl_matrix.round(1).to_string())

    # 2. 2D 매트릭스: 최대 낙폭 MDD (%)
    mdd_matrix = df_res.pivot(index="trend_risk", columns="mr_risk", values="mdd_pct")
    print("\n[2. 최대 낙폭 MDD 매트릭스 (%) - 행: 추세장(TF) / 열: 횡보장(MR)]")
    print(mdd_matrix.round(2).to_string())

    # 3. 2D 매트릭스: 칼마 비율 (수익률 / MDD)
    calmar_matrix = df_res.pivot(index="trend_risk", columns="mr_risk", values="calmar_ratio")
    print("\n[3. 칼마 비율 매트릭스 (수익률 ÷ MDD) - 행: 추세장(TF) / 열: 횡보장(MR)]")
    print(calmar_matrix.round(2).to_string())

    # 4. 2022년 하락장 손익 매트릭스 ($)
    y22_matrix = df_res.pivot(index="trend_risk", columns="mr_risk", values="y2022_pnl")
    print("\n[4. 2022년 크립토 윈터 PnL ($) - 행: 추세장(TF) / 열: 횡보장(MR)]")
    print(y22_matrix.round(1).to_string())

    # 5. 수익률 TOP 3
    top_pnl = df_res.sort_values(by="total_return_pct", ascending=False).head(3)
    print("\n[5. 4개년 총수익률 최고 TOP 3]")
    for idx, r in top_pnl.iterrows():
        print(f" * TF {r['trend_risk']}% x MR {r['mr_risk']}% -> 수익: +${r['profit_dollars']:,.2f} (+{r['total_return_pct']:.1f}%) | MDD: {r['mdd_pct']:.2f}% | Calmar: {r['calmar_ratio']:.2f} | 2022년: {r['y2022_pnl']:+8.1f}$")

    # 6. 칼마 비율 TOP 3
    top_cal = df_res.sort_values(by="calmar_ratio", ascending=False).head(3)
    print("\n[6. 칼마 비율(가성비) 최고 TOP 3]")
    for idx, r in top_cal.iterrows():
        print(f" * TF {r['trend_risk']}% x MR {r['mr_risk']}% -> 수익: +${r['profit_dollars']:,.2f} (+{r['total_return_pct']:.1f}%) | MDD: {r['mdd_pct']:.2f}% | Calmar: {r['calmar_ratio']:.2f} | 2022년: {r['y2022_pnl']:+8.1f}$")


if __name__ == "__main__":
    run_experiment_51()
