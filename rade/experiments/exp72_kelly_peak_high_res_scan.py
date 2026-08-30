"""
[실험 72] 100배 레버리지 켈리 기준 피크 초정밀 고해상도 스캔
- 탐색 범위:
  - TF (추세장): [3.0%, 3.5%, 4.0%, 4.5%, 5.0%]
  - MR (횡보장): [12.0%, 14.0%, 16.0%, 18.0%, 20.0%]
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


def run_experiment_72():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    df_asym = get_asymmetric_df(df_ind, base_th=0.74, bear_th=0.80)
    test_df = df_asym.iloc[720:].reset_index(drop=True)

    print("=" * 105)
    print("      [실험 72] 100배 레버리지 켈리 기준 피크 고해상도 스캔 (TF 3.0~5.0% x MR 12~20%)")
    print("=" * 105)

    tf_levels = [0.030, 0.035, 0.040, 0.045, 0.050]
    mr_levels = [0.120, 0.140, 0.160, 0.180, 0.200]
    results = []

    for t_risk in tf_levels:
        for m_risk in mr_levels:
            sim = BacktestSimulator(
                initial_capital=10000.0,
                trend_risk_pct=t_risk,
                mr_risk_pct=m_risk,
                leverage=100.0,
                bear_mode="SHORT",
                use_regime_transition_cut=False,
                trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
                mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
            )
            sim.pos_manager.max_leverage = 100.0
            sim.pos_manager.default_leverage = 100.0

            res = sim.run(test_df)
            calmar = res["total_return_pct"] / res["mdd_pct"] if res["mdd_pct"] > 0 else 0

            results.append({
                "trend_risk": round(t_risk * 100, 1),
                "mr_risk": round(m_risk * 100, 1),
                "final_equity": res["final_equity"],
                "total_return_pct": res["total_return_pct"],
                "profit_dollars": res["final_equity"] - 10000.0,
                "mdd_pct": res["mdd_pct"],
                "calmar_ratio": calmar,
            })

    df_res = pd.DataFrame(results)

    # 1. 2D 총 수익률 (%)
    pnl_matrix = df_res.pivot(index="trend_risk", columns="mr_risk", values="total_return_pct")
    print("\n[1. 4개년 총 수익률 매트릭스 (%) - 행: 추세장(TF) / 열: 횡보장(MR)]")
    print(pnl_matrix.round(1).to_string())

    # 2. 2D 최대 낙폭 MDD (%)
    mdd_matrix = df_res.pivot(index="trend_risk", columns="mr_risk", values="mdd_pct")
    print("\n[2. 최대 낙폭 MDD 매트릭스 (%) - 행: 추세장(TF) / 열: 횡보장(MR)]")
    print(mdd_matrix.round(2).to_string())

    # 3. 2D 칼마 비율
    calmar_matrix = df_res.pivot(index="trend_risk", columns="mr_risk", values="calmar_ratio")
    print("\n[3. 칼마 비율 매트릭스 (수익률 ÷ MDD) - 행: 추세장(TF) / 열: 횡보장(MR)]")
    print(calmar_matrix.round(2).to_string())

    # TOP 3 출력
    top_pnl = df_res.sort_values(by="total_return_pct", ascending=False).head(5)
    print("\n[4. 4개년 총수익률 최고 TOP 5 피크]")
    for idx, r in top_pnl.iterrows():
        print(f" * TF {r['trend_risk']}% x MR {r['mr_risk']}% -> 수익: +${r['profit_dollars']:,.2f} (+{r['total_return_pct']:.1f}%) | MDD: {r['mdd_pct']:.2f}% | Calmar: {r['calmar_ratio']:.2f}")


if __name__ == "__main__":
    run_experiment_72()
