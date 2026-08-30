"""
[실험 70] MR(횡보장) 리스크 극대화 2D 정밀 그리드 스캔 (0.5% ~ 15.0%)
- 엔진: rade.backtest.simulator.BacktestSimulator (무결성 보수적 체결 모델링)
- TF 리스크: [0.5%, 1.0%, 1.5%, 2.0%, 2.5%, 3.0%, 3.5%, 4.0%]
- MR 리스크: [0.5%, 3.0%, 6.0%, 9.0%, 12.0%, 15.0%]
- 목적: MR 리스크가 10%를 넘어 12%, 15%로 갈 때 완벽한 포화(Plateau)인지, 파산/MDD 급증이 발생하는지 한계선 규명
"""
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from rade.backtest.simulator import BacktestSimulator
from rade.engines.trend_following import TrendFollowingEngine
from rade.engines.mean_reversion import MeanReversionEngine


def run_experiment_70():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    print("HMM 국면 확률 계산 중...")
    reg_mgr = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.74, cooldown_bars=0)
    df_proc = reg_mgr.calculate_regime_probabilities(df_ind)
    test_df = df_proc.iloc[720:].reset_index(drop=True)

    print("=" * 105)
    print("      [실험 70] MR 극대화 2D 정밀 그리드 스캔 (TF 0.5~4.0% x MR 0.5~15.0%)")
    print("=" * 105)

    tf_risks = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
    mr_risks = [0.005, 0.030, 0.060, 0.090, 0.120, 0.150]

    results = []

    for t_risk in tf_risks:
        for m_risk in mr_risks:
            sim = BacktestSimulator(
                initial_capital=10000.0,
                trend_risk_pct=t_risk,
                mr_risk_pct=m_risk,
                leverage=3.0,
                bear_mode="CASH",
                use_regime_transition_cut=False,
                trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
                mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
            )
            res = sim.run(test_df)
            t = res["trades_df"].copy()
            if not t.empty:
                t["entry_dt"] = pd.to_datetime(t["entry_time"])
                t["year"] = t["entry_dt"].dt.year
                y2022_pnl = t[t["year"] == 2022]["pnl"].sum() if len(t[t["year"] == 2022]) > 0 else 0.0
            else:
                y2022_pnl = 0.0

            calmar = res["total_return_pct"] / res["mdd_pct"] if res["mdd_pct"] > 0 else 0.0

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
    print("\n[1. 실전 정밀 4개년 총 수익률 매트릭스 (%) - 행: 추세장(TF) / 열: 횡보장(MR)]")
    print(pnl_matrix.round(1).to_string())

    # 2. 2D 매트릭스: 최대 낙폭 MDD (%)
    mdd_matrix = df_res.pivot(index="trend_risk", columns="mr_risk", values="mdd_pct")
    print("\n[2. 실전 정밀 최대 낙폭 MDD 매트릭스 (%) - 행: 추세장(TF) / 열: 횡보장(MR)]")
    print(mdd_matrix.round(2).to_string())

    # 3. 2D 매트릭스: 칼마 비율 (수익률 ÷ MDD)
    calmar_matrix = df_res.pivot(index="trend_risk", columns="mr_risk", values="calmar_ratio")
    print("\n[3. 실전 정밀 칼마 비율 매트릭스 (수익률 ÷ MDD) - 행: 추세장(TF) / 열: 횡보장(MR)]")
    print(calmar_matrix.round(2).to_string())

    # 4. 2022년 하락장 손익 매트릭스 ($)
    y22_matrix = df_res.pivot(index="trend_risk", columns="mr_risk", values="y2022_pnl")
    print("\n[4. 실전 정밀 2022년 크립토 윈터 PnL ($) - 행: 추세장(TF) / 열: 횡보장(MR)]")
    print(y22_matrix.round(1).to_string())

    # 5. 칼마 비율(가성비) 기준 TOP 5
    top_cal = df_res.sort_values(by="calmar_ratio", ascending=False).head(5)
    print("\n[5. 칼마 비율(가성비) 기준 전체 TOP 5]")
    for idx, r in top_cal.iterrows():
        print(f" * TF {r['trend_risk']}% x MR {r['mr_risk']}% -> 수익: +${r['profit_dollars']:,.2f} (+{r['total_return_pct']:.1f}%) | MDD: {r['mdd_pct']:.2f}% | Calmar: {r['calmar_ratio']:.2f} | 2022년: {r['y2022_pnl']:+8.1f}$")

    # 6. 최대 수익률 기준 TOP 5
    top_pnl = df_res.sort_values(by="total_return_pct", ascending=False).head(5)
    print("\n[6. 최대 수익률 기준 전체 TOP 5]")
    for idx, r in top_pnl.iterrows():
        print(f" * TF {r['trend_risk']}% x MR {r['mr_risk']}% -> 수익: +${r['profit_dollars']:,.2f} (+{r['total_return_pct']:.1f}%) | MDD: {r['mdd_pct']:.2f}% | Calmar: {r['calmar_ratio']:.2f} | 2022년: {r['y2022_pnl']:+8.1f}$")


if __name__ == "__main__":
    run_experiment_70()
