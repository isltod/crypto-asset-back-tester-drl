"""
[실험 41] 단일 고정 리스크 비율 (risk_per_trade_pct = 0.5% ~ 4.0%) 전구간 민감도 백테스트
- 대상: RADE 공식 표준 프로파일 (CONSERVATIVE_CASH, TH=0.74)
- 범위: 0.5%, 1.0%, 1.5%, 2.0%, 2.5%, 3.0%, 3.5%, 4.0%
- 목적: 리스크 비율 증가에 따른 복리 수익률 vs MDD 증가 곡선 및 칼마 비율(Calmar Ratio = 수익률 / MDD) 분석
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


def run_experiment_41():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    reg_mgr = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.74, cooldown_bars=0)
    df_proc = reg_mgr.calculate_regime_probabilities(df_ind)
    test_df = df_proc.iloc[720:].reset_index(drop=True)

    print("=" * 105)
    print("      [실험 41] 단일 고정 리스크 비율 (0.5% ~ 4.0%) 전구간 민감도 스캔 (TH=0.74, CASH 모드)")
    print("=" * 105)

    risk_candidates = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
    results = []

    for r_pct in risk_candidates:
        sim = BacktestSimulator(
            initial_capital=10000.0,
            risk_per_trade_pct=r_pct,
            leverage=3.0,
            bear_mode="CASH",
            use_regime_transition_cut=False,
            trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
            mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
        )
        res = sim.run(test_df)
        t = res["trades_df"].copy()
        t["entry_dt"] = pd.to_datetime(t["entry_time"])
        t["year"] = t["entry_dt"].dt.year

        calmar = res["total_return_pct"] / res["mdd_pct"] if res["mdd_pct"] > 0 else 0

        # 연도별 PnL
        y_pnls = {}
        for yr in [2021, 2022, 2023, 2024]:
            y_pnls[yr] = t[t["year"] == yr]["pnl"].sum() if len(t[t["year"] == yr]) > 0 else 0.0

        results.append({
            "risk_pct": r_pct * 100,
            "final_equity": res["final_equity"],
            "total_return_pct": res["total_return_pct"],
            "profit_dollars": res["final_equity"] - 10000.0,
            "mdd_pct": res["mdd_pct"],
            "profit_factor": res["profit_factor"],
            "win_rate_pct": res["win_rate_pct"],
            "total_trades": res["total_trades"],
            "calmar_ratio": calmar,
            "y2021": y_pnls[2021],
            "y2022": y_pnls[2022],
            "y2023": y_pnls[2023],
            "y2024": y_pnls[2024],
        })

    # 종합 비교 출력
    print(f"\n{'리스크(%)':<8} | {'4개년 총 수익금':<20} | {'최대 낙폭(MDD)':<14} | {'칼마비율(Calmar)':<16} | {'손익비(PF)':<10} | {'승률':<8} | {'2022(하락장)'}")
    print("-" * 105)
    for r in results:
        p_str = f"+${r['profit_dollars']:,.2f} (+{r['total_return_pct']:.1f}%)"
        m_str = f"{r['mdd_pct']:.2f}%"
        c_str = f"{r['calmar_ratio']:.2f}"
        pf_str = f"{r['profit_factor']:.2f}"
        wr_str = f"{r['win_rate_pct']:.1f}%"
        y22_str = f"{r['y2022']:+8.1f}$"
        mark = " (현재 기본값)" if abs(r["risk_pct"] - 2.0) < 1e-4 else ""
        print(f"{r['risk_pct']:>5.1f}%   | {p_str:<20} | {m_str:<14} | {c_str:<16} | {pf_str:<10} | {wr_str:<8} | {y22_str}{mark}")
    print("-" * 105)

    print("\n[연도별 PnL 분해 상세]")
    print(f"{'리스크(%)':<8} | {'2021년 (불장)':<15} | {'2022년 (하락장)':<15} | {'2023년 (횡보장)':<15} | {'2024년 (불장)':<15}")
    print("-" * 75)
    for r in results:
        print(f"{r['risk_pct']:>5.1f}%   | {r['y2021']:>+12.2f}$ | {r['y2022']:>+12.2f}$ | {r['y2023']:>+12.2f}$ | {r['y2024']:>+12.2f}$")
    print("=" * 105)


if __name__ == "__main__":
    run_experiment_41()
