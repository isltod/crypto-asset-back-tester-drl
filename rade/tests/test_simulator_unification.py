"""
[통합 회귀 테스트] 1단계 시뮬레이터 단일화 무결성 검증 테스트 스위트
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
from rade.experiments.exp16_3state_hmm import Regime3StateManager, HMM3StateSimulator
from rade.live.paper_trader import PaperTrader


def run_unification_test():
    print("=" * 80)
    print("    [RADE SYSTEM] 1단계 시뮬레이터 단일화 3중 자동 회귀 테스트 스위트")
    print("=" * 80)

    # 1. 데이터 로드
    print("\n[Step 1] 4개년(2021~2024) 풀데이터 로드 및 지표 생성...")
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_all = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_all["datetime"] = pd.to_datetime(df_all["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_all)

    # 2. HMM 3-State 국면 계산 (프로덕션 RegimeManager 직접 사용)
    print("[Step 2] 3-State HMM 주간 앵커링 국면 계산 (TH=0.45)...")
    reg_mgr = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.45, cooldown_bars=0)
    df_processed = reg_mgr.calculate_regime_probabilities(df_ind)
    test_df = df_processed.iloc[720:].reset_index(drop=True)

    # 3. 프로덕션 BacktestSimulator 실행 (최적 조합 C: ATR 4.5x, TS 24h)
    print("[Step 3] 프로덕션 BacktestSimulator (Single Source of Truth) 실행...")
    tf_eng = TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5)
    mr_eng = MeanReversionEngine(max_holding_bars=24)

    prod_sim = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        bear_mode="CASH",
        use_regime_transition_cut=False,
        trend_engine=tf_eng,
        mean_revert_engine=mr_eng
    )
    prod_res = prod_sim.run(test_df)

    # 4. 검증 및 단언 (Assertion Check)
    print("\n" + "-" * 80)
    print("                      [ 1:1 정밀 검증 결과 판정표 ]")
    print("-" * 80)

    checks = []

    # 검증 1: 거래 횟수 157회 일치
    t_cnt = prod_res["total_trades"]
    pass_cnt = (t_cnt == 157)
    checks.append(("1. Total Trades Count (157)", f"{t_cnt} trades", "157 trades", pass_cnt))

    # 검증 2: 총수익률 +134.44% 일치
    tot_ret = prod_res["total_return_pct"]
    pass_ret = abs(tot_ret - 134.44) < 0.1
    checks.append(("2. Total Return Pct (+134.44%)", f"{tot_ret:+.2f}%", "+134.44%", pass_ret))

    # 검증 3: MDD 14.64% 일치
    mdd = prod_res["mdd_pct"]
    pass_mdd = abs(mdd - 14.64) < 0.1
    checks.append(("3. Max Drawdown MDD (14.64%)", f"{mdd:.2f}%", "14.64%", pass_mdd))

    # 검증 4: 손익비 PF 1.80 일치
    pf = prod_res["profit_factor"]
    pass_pf = abs(pf - 1.80) < 0.05
    checks.append(("4. Profit Factor PF (1.80)", f"{pf:.2f}", "1.80", pass_pf))

    # 검증 5: 승률 54.8% 일치
    wr = prod_res["win_rate_pct"]
    pass_wr = abs(wr - 54.8) < 0.5
    checks.append(("5. Win Rate (54.8%)", f"{wr:.1f}%", "54.8%", pass_wr))

    # 검증 6: 실전 PaperTrader 초기화 파라미터 정합성
    pt = PaperTrader()
    pt_check = (
        pt.tf_engine.max_trailing_atr == 4.5 and
        pt.mr_engine.max_holding_bars == 24 and
        pt.regime_manager.cooldown_bars == 0 and
        pt.regime_manager.trans_threshold == 0.45
    )
    checks.append(("6. PaperTrader Live Param Injection", "4.5x / 24h / 0cd", "4.5x / 24h / 0cd", pt_check))

    # 출력
    all_passed = True
    for name, actual, expected, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f" * {name:<42} | Actual: {actual:<14} | Expected: {expected:<14} | {status}")

    print("-" * 80)

    # 5. 연도별 수익성 분해 확인
    trades_df = prod_res["trades_df"]
    trades_df["entry_dt"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["year"] = trades_df["entry_dt"].dt.year

    print("\n[Yearly PnL Breakdown]")
    for yr in [2021, 2022, 2023, 2024]:
        sub = trades_df[trades_df["year"] == yr]
        yr_pnl = sub["pnl"].sum()
        print(f" - {yr}: {yr_pnl:+9.2f}$ ({len(sub)} trades)")

    print("=" * 80)
    if all_passed:
        print(" [1단계 최종 판정]: 모든 시뮬레이터 및 프로덕션 코드가 단일 소스로 완벽 통합되었습니다! (ALL PASS)")
    else:
        print(" [1단계 최종 판정]: 일부 검증 항목이 불일치합니다. 확인이 필요합니다.")
    print("=" * 80)

    assert all_passed, "1단계 단일화 검증 실패!"


if __name__ == "__main__":
    run_unification_test()
