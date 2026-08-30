"""
[실험 68] 국면 판정 소거 실험 (Ablation Study)
- 표준 RADE 시스템 (HMM 3-State: BULL 2.0% / MR 4.0% / BEAR CASH)
vs
- 100% 횡보장 가상 (HMM 제외, 전 구간 MEAN_REVERSION 엔진 단독 매매)
  - Case 1: MR 4.0% 리스크 (표준 RADE의 MR 리스크 동일 적용)
  - Case 2: MR 2.0% 리스크 (동일 2.0% 기준)
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


def run_ablation_experiment():
    print("=" * 105)
    print("      [실험 68] 국면 판정 소거(Ablation) 백테스트: 표준 RADE vs 100% MR(횡보) 단독 매매")
    print("=" * 105)

    # 1. 데이터 로드
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_raw = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
    df_indicators = add_all_indicators(df_raw)

    # 2. 표준 HMM 국면 계산
    print("[1/3] HMM 국면 확률 계산 중...")
    regime_manager = RegimeManager(
        hmm_window=720,
        retrain_interval=168,
        trans_threshold=0.74,
        cooldown_bars=0
    )
    df_processed = regime_manager.calculate_regime_probabilities(df_indicators)
    test_df_standard = df_processed.iloc[720:].copy().reset_index(drop=True)

    # 3. 100% RANGE 가상 데이터 준비 (HMM 제외, 전 구간을 RANGE로 덮어씀)
    test_df_all_range = test_df_standard.copy()
    test_df_all_range["regime_state"] = "RANGE"
    test_df_all_range["p_range"] = 1.0
    test_df_all_range["p_bull"] = 0.0
    test_df_all_range["p_bear"] = 0.0

    # 4. 표준 RADE 백테스트 (BULL 2.0% / MR 4.0%)
    print("[2/3] 표준 RADE 백테스트 시뮬레이션 중...")
    sim_standard = BacktestSimulator(
        initial_capital=10000.0,
        trend_risk_pct=0.020,
        mr_risk_pct=0.040,
        leverage=3.0,
        bear_mode="CASH",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24),
    )
    res_standard = sim_standard.run(test_df_standard)

    # 5. 100% MR 단독 백테스트 (MR 4.0% 리스크)
    print("[3/3] 100% MR(평균회귀) 단독 백테스트 시뮬레이션 중...")
    sim_all_mr_4pct = BacktestSimulator(
        initial_capital=10000.0,
        trend_risk_pct=0.020,
        mr_risk_pct=0.040,
        leverage=3.0,
        bear_mode="CASH",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24),
    )
    res_all_mr_4pct = sim_all_mr_4pct.run(test_df_all_range)

    # 6. 100% MR 단독 백테스트 (MR 2.0% 리스크)
    sim_all_mr_2pct = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.020,
        leverage=3.0,
        bear_mode="CASH",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24),
    )
    res_all_mr_2pct = sim_all_mr_2pct.run(test_df_all_range)

    # 연도별 성과 분해 함수
    def get_yearly_stats(trades_df: pd.DataFrame):
        if trades_df.empty:
            return {}
        df = trades_df.copy()
        df["entry_dt"] = pd.to_datetime(df["entry_time"])
        df["year"] = df["entry_dt"].dt.year
        yearly = {}
        for yr in [2021, 2022, 2023, 2024]:
            sub = df[df["year"] == yr]
            pnl = sub["pnl"].sum()
            wr = (sub["pnl"] > 0).mean() * 100.0 if len(sub) > 0 else 0.0
            yearly[yr] = (pnl, len(sub), wr)
        return yearly

    y_std = get_yearly_stats(res_standard["trades_df"])
    y_mr4 = get_yearly_stats(res_all_mr_4pct["trades_df"])
    y_mr2 = get_yearly_stats(res_all_mr_2pct["trades_df"])

    pnl_std = res_standard["final_equity"] - 10000.0
    pnl_mr4 = res_all_mr_4pct["final_equity"] - 10000.0
    pnl_mr2 = res_all_mr_2pct["final_equity"] - 10000.0

    ret_std = res_standard["total_return_pct"]
    ret_mr4 = res_all_mr_4pct["total_return_pct"]
    ret_mr2 = res_all_mr_2pct["total_return_pct"]

    mdd_std = res_standard["mdd_pct"]
    mdd_mr4 = res_all_mr_4pct["mdd_pct"]
    mdd_mr2 = res_all_mr_2pct["mdd_pct"]

    pf_std = res_standard["profit_factor"]
    pf_mr4 = res_all_mr_4pct["profit_factor"]
    pf_mr2 = res_all_mr_2pct["profit_factor"]

    wr_std = res_standard["win_rate_pct"]
    wr_mr4 = res_all_mr_4pct["win_rate_pct"]
    wr_mr2 = res_all_mr_2pct["win_rate_pct"]

    tr_std = res_standard["total_trades"]
    tr_mr4 = res_all_mr_4pct["total_trades"]
    tr_mr2 = res_all_mr_2pct["total_trades"]

    calmar_std = ret_std / mdd_std if mdd_std > 0 else 0.0
    calmar_mr4 = ret_mr4 / mdd_mr4 if mdd_mr4 > 0 else 0.0
    calmar_mr2 = ret_mr2 / mdd_mr2 if mdd_mr2 > 0 else 0.0

    str_ret_std = f"+${pnl_std:,.2f} (+{ret_std:.2f}%)"
    str_ret_mr4 = f"+${pnl_mr4:,.2f} (+{ret_mr4:.2f}%)"
    str_ret_mr2 = f"+${pnl_mr2:,.2f} (+{ret_mr2:.2f}%)"

    str_mdd_std = f"{mdd_std:.2f}%"
    str_mdd_mr4 = f"{mdd_mr4:.2f}%"
    str_mdd_mr2 = f"{mdd_mr2:.2f}%"

    str_cal_std = f"{calmar_std:.2f}"
    str_cal_mr4 = f"{calmar_mr4:.2f}"
    str_cal_mr2 = f"{calmar_mr2:.2f}"

    str_pf_std = f"{pf_std:.2f}"
    str_pf_mr4 = f"{pf_mr4:.2f}"
    str_pf_mr2 = f"{pf_mr2:.2f}"

    str_wr_std = f"{wr_std:.1f}%"
    str_wr_mr4 = f"{wr_mr4:.1f}%"
    str_wr_mr2 = f"{wr_mr2:.1f}%"

    str_tr_std = f"{tr_std}회 (연 {tr_std/3.92:.1f}회)"
    str_tr_mr4 = f"{tr_mr4}회 (연 {tr_mr4/3.92:.1f}회)"
    str_tr_mr2 = f"{tr_mr2}회 (연 {tr_mr2/3.92:.1f}회)"

    # 7. 종합 결과 출력
    print("\n" + "=" * 105)
    print("                  [ 표준 RADE vs 100% MR(횡보전략) 단독 1:1 비교표 ]")
    print("=" * 105)
    print(f"{'지표 항목':<22} | {'🌟 표준 RADE (HMM 활성)':<25} | {'❌ 100% MR 단독 (Risk 4%)':<25} | {'❌ 100% MR 단독 (Risk 2%)':<25}")
    print("-" * 105)
    print(f"{'4개년 총 수익금':<22} | {str_ret_std:<25} | {str_ret_mr4:<25} | {str_ret_mr2:<25}")
    print(f"{'최대 낙폭 (MDD)':<22} | {str_mdd_std:<25} | {str_mdd_mr4:<25} | {str_mdd_mr2:<25}")
    print(f"{'칼마 비율 (수익/MDD)':<22} | {str_cal_std:<25} | {str_cal_mr4:<25} | {str_cal_mr2:<25}")
    print(f"{'손익비 (PF)':<22} | {str_pf_std:<25} | {str_pf_mr4:<25} | {str_pf_mr2:<25}")
    print(f"{'승률 (Win Rate)':<22} | {str_wr_std:<25} | {str_wr_mr4:<25} | {str_wr_mr2:<25}")
    print(f"{'총 거래 횟수':<22} | {str_tr_std:<25} | {str_tr_mr4:<25} | {str_tr_mr2:<25}")
    print("-" * 105)

    print("\n[연도별 PnL 분해 비교]")
    print("-" * 105)
    for yr in [2021, 2022, 2023, 2024]:
        s_pnl = y_std.get(yr, (0, 0, 0))[0]
        m4_pnl = y_mr4.get(yr, (0, 0, 0))[0]
        m2_pnl = y_mr2.get(yr, (0, 0, 0))[0]
        print(f" * {yr}년 : 표준 RADE {s_pnl:+9.2f}$  vs  MR 단독(4%) {m4_pnl:+9.2f}$  vs  MR 단독(2%) {m2_pnl:+9.2f}$")
    print("=" * 105)


if __name__ == "__main__":
    run_ablation_experiment()
