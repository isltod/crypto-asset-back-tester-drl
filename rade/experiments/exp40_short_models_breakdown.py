"""
[실험 40] 3가지 숏 배팅 모델 정밀 대조 및 세부 집계 리포트 생성 스크립트
1. 모델 1: exp35 (대칭 TH=0.74, 숏 활성화)
2. 모델 2: exp36 (대칭 TH=0.74, 순수 방향성 분리)
3. 모델 3: exp39 (비대칭 BEAR TH=0.80, 기본 TH=0.74)
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


def get_trades_for_model(df_proc: pd.DataFrame):
    sim = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        bear_mode="SHORT",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    test_df = df_proc.iloc[720:].reset_index(drop=True)
    res = sim.run(test_df)
    t = res["trades_df"].copy()
    t["entry_dt"] = pd.to_datetime(t["entry_time"])
    t["year"] = t["entry_dt"].dt.year
    return res, t


def run_experiment_40():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    # 1. 모델 1 & 2: 대칭 TH=0.74
    reg_mgr_74 = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.74, cooldown_bars=0)
    df_proc_74 = reg_mgr_74.calculate_regime_probabilities(df_ind)
    res_m1, t_m1 = get_trades_for_model(df_proc_74)

    # 2. 모델 3: 비대칭 BEAR TH=0.80
    reg_raw = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.30, cooldown_bars=0)
    df_raw = reg_raw.calculate_regime_probabilities(df_ind)
    
    curr = RegimeState.RANGE
    asym_states = []
    for idx, row in df_raw.iterrows():
        p_r = row["p_range"]
        p_u = row["p_bull"]
        p_d = row["p_bear"]
        if p_d >= 0.80 and p_d >= p_u and p_d >= p_r:
            curr = RegimeState.BEAR_PANIC
        elif p_u >= 0.74 and p_u >= p_r and p_u >= p_d:
            curr = RegimeState.BULL_TREND
        elif p_r >= 0.74 and p_r >= p_u and p_r >= p_d:
            curr = RegimeState.RANGE
        asym_states.append(curr)
    df_raw["regime_state"] = asym_states
    res_m3, t_m3 = get_trades_for_model(df_raw)

    print("=" * 115)
    print("      [실험 40] 3가지 숏 배팅 모델 정밀 대조 및 세부 성과 집계 리포트")
    print("=" * 115)

    # 총괄표 출력
    print("\n[1. 총괄 성과 비교표 (Overview Table)]")
    print(f"{'지표 항목':<22} | {'① [exp35] 숏 배팅 활성화 (TH=0.74)':<35} | {'② [exp36] 국면별 순수 방향성 (TH=0.74)':<35} | {'③ [exp39] 비대칭 BEAR TH=0.80':<30}")
    print("-" * 130)
    
    r1_pnl = f"+${res_m1['final_equity']-10000:,.2f} (+{res_m1['total_return_pct']:.2f}%)"
    r2_pnl = f"+${res_m1['final_equity']-10000:,.2f} (+{res_m1['total_return_pct']:.2f}%)"
    r3_pnl = f"+${res_m3['final_equity']-10000:,.2f} (+{res_m3['total_return_pct']:.2f}%)"

    r1_mdd = f"{res_m1['mdd_pct']:.2f}%"
    r2_mdd = f"{res_m1['mdd_pct']:.2f}%"
    r3_mdd = f"{res_m3['mdd_pct']:.2f}%"

    r1_pf = f"{res_m1['profit_factor']:.2f}"
    r2_pf = f"{res_m1['profit_factor']:.2f}"
    r3_pf = f"{res_m3['profit_factor']:.2f}"

    r1_wr = f"{res_m1['win_rate_pct']:.1f}%"
    r2_wr = f"{res_m1['win_rate_pct']:.1f}%"
    r3_wr = f"{res_m3['win_rate_pct']:.1f}%"

    r1_tr = f"{res_m1['total_trades']}회 (연 {res_m1['total_trades']/3.92:.1f}회)"
    r2_tr = f"{res_m1['total_trades']}회 (연 {res_m1['total_trades']/3.92:.1f}회)"
    r3_tr = f"{res_m3['total_trades']}회 (연 {res_m3['total_trades']/3.92:.1f}회)"

    print(f"{'4개년 총 수익금':<22} | {r1_pnl:<35} | {r2_pnl:<35} | {r3_pnl:<30}")
    print(f"{'최대 낙폭 (MDD)':<22} | {r1_mdd:<35} | {r2_mdd:<35} | {r3_mdd:<30}")
    print(f"{'손익비 (PF)':<22} | {r1_pf:<35} | {r2_pf:<35} | {r3_pf:<30}")
    print(f"{'전체 승률 (Win Rate)':<22} | {r1_wr:<35} | {r2_wr:<35} | {r3_wr:<30}")
    print(f"{'총 거래 횟수':<22} | {r1_tr:<35} | {r2_tr:<35} | {r3_tr:<30}")
    print("-" * 130)

    # 연도별 x 국면별 x 롱/숏 상세 분해
    def print_breakdown(t_df, label):
        print(f"\n====================== [ {label} 세부 성과 분해 ] ======================")
        print(f"{'연도':<6} | {'국면 (Regime)':<16} | {'방향':<6} | {'거래수':<6} | {'승률':<8} | {'총 손익 ($)':<14} | {'건당 평균 ($)'}")
        print("-" * 85)
        for yr in [2021, 2022, 2023, 2024]:
            t_yr = t_df[t_df["year"] == yr]
            for eng, r_name in [("TREND_FOLLOWING", "BULL (Trend Long)"), ("TREND_FOLLOWING_SHORT", "BEAR (Trend Short)"), ("MEAN_REVERSION_LONG", "RANGE (MR Long)"), ("MEAN_REVERSION_SHORT", "RANGE (MR Short)")]:
                if eng == "BULL (Trend Long)" or eng == "TREND_FOLLOWING":
                    sub = t_yr[(t_yr["engine"] == "TREND_FOLLOWING") & (t_yr["side"].astype(str).str.contains("LONG"))]
                    r_lbl = "BULL_TREND"
                    s_lbl = "LONG"
                elif eng == "TREND_FOLLOWING_SHORT":
                    sub = t_yr[(t_yr["engine"] == "TREND_FOLLOWING") & (t_yr["side"].astype(str).str.contains("SHORT"))]
                    r_lbl = "BEAR_PANIC"
                    s_lbl = "SHORT"
                elif eng == "MEAN_REVERSION_LONG":
                    sub = t_yr[(t_yr["engine"] == "MEAN_REVERSION") & (t_yr["side"].astype(str).str.contains("LONG"))]
                    r_lbl = "RANGE"
                    s_lbl = "LONG"
                elif eng == "MEAN_REVERSION_SHORT":
                    sub = t_yr[(t_yr["engine"] == "MEAN_REVERSION") & (t_yr["side"].astype(str).str.contains("SHORT"))]
                    r_lbl = "RANGE"
                    s_lbl = "SHORT"
                
                cnt = len(sub)
                if cnt > 0:
                    pnl_sum = sub["pnl"].sum()
                    wr = len(sub[sub["pnl"] > 0]) / cnt * 100
                    avg_p = pnl_sum / cnt
                    print(f"{yr:<6} | {r_lbl:<16} | {s_lbl:<6} | {cnt:>4d}회 | {wr:>6.1f}% | {pnl_sum:>+12.2f}$ | {avg_p:>+10.2f}$")
            
            # 연도별 소계
            yr_pnl = t_yr["pnl"].sum()
            yr_wr = len(t_yr[t_yr["pnl"] > 0]) / len(t_yr) * 100 if len(t_yr) > 0 else 0
            print(f"[{yr}년 소계] 총 {len(t_yr):2d}회 거래 | 전체 승률 {yr_wr:5.1f}% | 총 PnL: {yr_pnl:+12.2f}$")
            print("-" * 85)

    print_breakdown(t_m1, "①/② [exp35 & exp36] 대칭 TH=0.74 모델")
    print_breakdown(t_m3, "③ [exp39] 비대칭 BEAR TH=0.80 모델")


if __name__ == "__main__":
    run_experiment_40()
