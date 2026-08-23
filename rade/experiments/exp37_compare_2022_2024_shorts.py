"""
[실험 37] 2022년 vs 2024년 국면 시간 점유율 및 숏 거래 성과 1:1 비교
- 1. 국면 시간 점유율 (RANGE vs BULL vs BEAR)
- 2. 숏 거래 유형별(MR Short vs TF Short) 거래수, 승률, PnL
"""
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from rade.backtest.simulator import BacktestSimulator
from rade.engines.trend_following import TrendFollowingEngine
from rade.engines.mean_reversion import MeanReversionEngine


def compare_2022_vs_2024():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    reg_mgr = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.74, cooldown_bars=0)
    df_proc = reg_mgr.calculate_regime_probabilities(df_ind)
    test_df = df_proc.iloc[720:].reset_index(drop=True)

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

    print("=" * 85)
    print("      [실험 37] 2022년(하락장) vs 2024년(불장) 숏 배팅 메커니즘 1:1 비교")
    print("=" * 85)

    for yr in [2022, 2024]:
        df_yr = test_df[test_df["datetime"].dt.year == yr]
        t_yr = t[t["year"] == yr]
        s_yr = t_yr[t_yr["side"].astype(str).str.contains("SHORT")]

        print(f"\n=================== [ {yr}년 정밀 분석 ] ===================")
        reg_dist = df_yr["regime_state"].value_counts(normalize=True) * 100
        print(f"[1. 국면 시간 점유율]")
        for r_name in ["RANGE", "BULL_TREND", "BEAR_PANIC"]:
            val = reg_dist.get(r_name, 0.0)
            print(f" * {r_name:<12}: {val:5.1f}%")

        print(f"\n[2. 숏 거래 세부 실적]")
        for eng in ["MEAN_REVERSION", "TREND_FOLLOWING"]:
            sub = s_yr[s_yr["engine"] == eng]
            pnl_sum = sub["pnl"].sum() if len(sub) > 0 else 0.0
            wr = len(sub[sub["pnl"] > 0]) / len(sub) * 100 if len(sub) > 0 else 0.0
            print(f" * {eng:<16}: {len(sub):2d}회 거래 | 승률 {wr:5.1f}% | 총 PnL: {pnl_sum:+9.2f}$ (건당 {pnl_sum/len(sub) if len(sub)>0 else 0:+.2f}$)")

        tot_pnl = s_yr["pnl"].sum()
        tot_wr = len(s_yr[s_yr["pnl"] > 0]) / len(s_yr) * 100 if len(s_yr) > 0 else 0.0
        print(f" * [숏 전체 합계  ]: {len(s_yr):2d}회 거래 | 승률 {tot_wr:5.1f}% | 총 PnL: {tot_pnl:+9.2f}$")

    print("=" * 85)


if __name__ == "__main__":
    compare_2022_vs_2024()
