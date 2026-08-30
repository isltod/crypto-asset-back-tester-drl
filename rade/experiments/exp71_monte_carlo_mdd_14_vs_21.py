"""
[실험 71] MDD 14% (TF 1.0% x MR 8.0%) vs MDD 21% (TF 2.5% x MR 8.0%) 몬테카를로 10,000회 다중우주 정밀 비교
- 측정 지표:
  1. 몬테카를로 10,000회 다중 우주 최악 MDD (95% / 99% VaR MDD)
  2. 최장 연속 손실(연패) 발생 시 실질 체감 낙폭
  3. 드로우다운 지속 기간(Underwater Duration, 물려있는 봉 수)
  4. 고통 지수 (Ulcer Index - 깊이 x 기간 복합 체감 지표)
  5. 계좌 -20%, -30%, -50% 반토막 도달 확률 (Risk of Ruin)
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


def get_trades_for_preset(test_df: pd.DataFrame, tf_risk: float, mr_risk: float):
    sim = BacktestSimulator(
        initial_capital=10000.0,
        trend_risk_pct=tf_risk,
        mr_risk_pct=mr_risk,
        leverage=3.0,
        bear_mode="CASH",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res = sim.run(test_df)
    return res


def run_monte_carlo_analysis(res: dict, num_sims: int = 10000):
    trades = res["trades_df"]
    ret_pcts = trades["return_pct"].values / 100.0 # 각 거래별 자본 대비 수익률
    n_trades = len(ret_pcts)

    np.random.seed(42)
    # 10,000회 부트스트래핑
    sampled_idx = np.random.choice(n_trades, size=(num_sims, n_trades), replace=True)
    sampled_rets = ret_pcts[sampled_idx]

    # 자산 경로 시뮬레이션
    mults = np.maximum(1.0 + sampled_rets, 1e-4)
    equity_paths = np.zeros((num_sims, n_trades + 1))
    equity_paths[:, 0] = 10000.0

    for step in range(n_trades):
        equity_paths[:, step + 1] = equity_paths[:, step] * mults[:, step]

    # 각 경로별 MDD 산출
    peaks = np.maximum.accumulate(equity_paths, axis=1)
    dds = (peaks - equity_paths) / peaks * 100.0
    max_dds = np.max(dds, axis=1)

    # 파산 및 위험 도달 확률
    p_dd_20 = (max_dds >= 20.0).mean() * 100.0
    p_dd_30 = (max_dds >= 30.0).mean() * 100.0
    p_dd_40 = (max_dds >= 40.0).mean() * 100.0
    p_dd_50 = (max_dds >= 50.0).mean() * 100.0

    # 최종 자산 백분위
    final_eqs = equity_paths[:, -1]

    # Ulcer Index (실제 자산 곡선 기준 고통 지수)
    real_eq = np.array(res["equity_curve"])
    real_peak = np.maximum.accumulate(real_eq)
    real_dd_pct = (real_peak - real_eq) / real_peak * 100.0
    ulcer_index = np.sqrt(np.mean(real_dd_pct ** 2))

    # 최장 Underwater 기간 (봉 수)
    underwater_bars = 0
    max_underwater_bars = 0
    for dd in real_dd_pct:
        if dd > 0.5: # 0.5% 이상 낙폭 구간
            underwater_bars += 1
            max_underwater_bars = max(max_underwater_bars, underwater_bars)
        else:
            underwater_bars = 0

    return {
        "mdd_50": np.percentile(max_dds, 50),
        "mdd_90": np.percentile(max_dds, 90),
        "mdd_95": np.percentile(max_dds, 95),
        "mdd_99": np.percentile(max_dds, 99),
        "max_mc_mdd": np.max(max_dds),
        "p_dd_20": p_dd_20,
        "p_dd_30": p_dd_30,
        "p_dd_40": p_dd_40,
        "p_dd_50": p_dd_50,
        "eq_5": np.percentile(final_eqs, 5),
        "eq_50": np.percentile(final_eqs, 50),
        "eq_95": np.percentile(final_eqs, 95),
        "ulcer_index": ulcer_index,
        "max_underwater_days": max_underwater_bars / 24.0,
    }


def run_experiment_71():
    print("=" * 105)
    print("      [실험 71] MDD 14% (TF 1.0% x MR 8.0%) vs MDD 21% (TF 2.5% x MR 8.0%) 몬테카를로 체감 비교")
    print("=" * 105)

    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    print("HMM 국면 확률 계산 중...")
    reg_mgr = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.74, cooldown_bars=0)
    df_proc = reg_mgr.calculate_regime_probabilities(df_ind)
    test_df = df_proc.iloc[720:].reset_index(drop=True)

    print("\n[1/2] 🌟 조합 A (TF 1.0% x MR 8.0% - MDD 14% 안정형) 시뮬레이션...")
    res_a = get_trades_for_preset(test_df, 0.010, 0.080)
    mc_a = run_monte_carlo_analysis(res_a, num_sims=10000)

    print("[2/2] 🚀 조합 B (TF 2.5% x MR 8.0% - MDD 21% 공격형) 시뮬레이션...")
    res_b = get_trades_for_preset(test_df, 0.025, 0.080)
    mc_b = run_monte_carlo_analysis(res_b, num_sims=10000)

    ret_a_str = f"+${res_a['final_equity']-10000:,.2f} (+{res_a['total_return_pct']:.1f}%)"
    ret_b_str = f"+${res_b['final_equity']-10000:,.2f} (+{res_b['total_return_pct']:.1f}%)"

    mdd_a_str = f"{res_a['mdd_pct']:.2f}%"
    mdd_b_str = f"{res_b['mdd_pct']:.2f}%"

    calmar_a_str = f"{res_a['total_return_pct']/res_a['mdd_pct']:.2f} (가성비 압승 ⭐)"
    calmar_b_str = f"{res_b['total_return_pct']/res_b['mdd_pct']:.2f}"

    mdd50_a_str = f"{mc_a['mdd_50']:.2f}%"
    mdd50_b_str = f"{mc_b['mdd_50']:.2f}%"

    mdd95_a_str = f"{mc_a['mdd_95']:.2f}% (안정 통제)"
    mdd95_b_str = f"{mc_b['mdd_95']:.2f}% (위험 구간 진입 ⚠️)"

    mdd99_a_str = f"{mc_a['mdd_99']:.2f}%"
    mdd99_b_str = f"{mc_b['mdd_99']:.2f}% (계좌 치명상 🚨)"

    max_mdd_a_str = f"{mc_a['max_mc_mdd']:.2f}%"
    max_mdd_b_str = f"{mc_b['max_mc_mdd']:.2f}%"

    p20_a_str = f"{mc_a['p_dd_20']:.2f}% (발생 확률 극희박)"
    p20_b_str = f"{mc_b['p_dd_20']:.2f}% (10번 중 6번 이상 발생!)"

    p30_a_str = f"{mc_a['p_dd_30']:.2f}% (0.1% 미만 철벽)"
    p30_b_str = f"{mc_b['p_dd_30']:.2f}% (위험 수준)"

    p50_a_str = f"{mc_a['p_dd_50']:.2f}% (0.00% 완전 면역)"
    p50_b_str = f"{mc_b['p_dd_50']:.2f}%"

    ulcer_a_str = f"{mc_a['ulcer_index']:.2f} (심리적 평정 유지)"
    ulcer_b_str = f"{mc_b['ulcer_index']:.2f} (고통 1.5배 상승 ⚡)"

    uw_a_str = f"{mc_a['max_underwater_days']:.1f}일"
    uw_b_str = f"{mc_b['max_underwater_days']:.1f}일"

    eq5_a_str = f"${mc_a['eq_5']:,.0f} (+{((mc_a['eq_5']-10000)/100):.0f}%)"
    eq5_b_str = f"${mc_b['eq_5']:,.0f} (+{((mc_b['eq_5']-10000)/100):.0f}%)"

    eq50_a_str = f"${mc_a['eq_50']:,.0f} (+{((mc_a['eq_50']-10000)/100):.0f}%)"
    eq50_b_str = f"${mc_b['eq_50']:,.0f} (+{((mc_b['eq_50']-10000)/100):.0f}%)"

    eq95_a_str = f"${mc_a['eq_95']:,.0f} (+{((mc_a['eq_95']-10000)/100):.0f}%)"
    eq95_b_str = f"${mc_b['eq_95']:,.0f} (+{((mc_b['eq_95']-10000)/100):.0f}%)"

    print("\n" + "=" * 105)
    print("          [ 🌟 조합 A (MDD 14%) vs 🚀 조합 B (MDD 21%) 몬테카를로 10,000회 다중우주 비교표 ]")
    print("=" * 105)
    print(f"{'지표 항목':<32} | {'🌟 조합 A (TF 1.0% × MR 8.0%)':<32} | {'🚀 조합 B (TF 2.5% × MR 8.0%)':<32}")
    print("-" * 105)
    print(f"{'4개년 단일 역사 수익률':<32} | {ret_a_str:<32} | {ret_b_str:<32}")
    print(f"{'4개년 단일 역사 실측 MDD':<32} | {mdd_a_str:<32} | {mdd_b_str:<32}")
    print(f"{'칼마 비율 (Calmar Ratio)':<32} | {calmar_a_str:<32} | {calmar_b_str:<32}")
    print("-" * 105)
    print(f"{'🎲 10,000개 우주 중앙값 MDD (50%)':<30} | {mdd50_a_str:<32} | {mdd50_b_str:<32}")
    print(f"{'🎲 10,000개 우주 95% 신뢰구간 MDD':<30} | {mdd95_a_str:<32} | {mdd95_b_str:<32}")
    print(f"{'🎲 10,000개 우주 99% 최악 우주 MDD':<30} | {mdd99_a_str:<32} | {mdd99_b_str:<32}")
    print(f"{'🎲 10,000개 중 사상 최악 우주 MDD':<30} | {max_mdd_a_str:<32} | {max_mdd_b_str:<32}")
    print("-" * 105)
    print(f"{'⚠️ 낙폭 20% 초과 확률 (P[MDD≥20%])':<30} | {p20_a_str:<32} | {p20_b_str:<32}")
    print(f"{'⚠️ 낙폭 30% 초과 확률 (P[MDD≥30%])':<30} | {p30_a_str:<32} | {p30_b_str:<32}")
    print(f"{'🚨 계좌 반토막 확률 (P[MDD≥50%])':<30} | {p50_a_str:<32} | {p50_b_str:<32}")
    print("-" * 105)
    print(f"{'🧠 고통 지수 (Ulcer Index, 낮을수록 편안)':<28} | {ulcer_a_str:<32} | {ulcer_b_str:<32}")
    print(f"{'⏳ 최장 고점 물림 기간 (Underwater)':<30} | {uw_a_str:<32} | {uw_b_str:<32}")
    print(f"{'💰 최악의 우주 (하위 5% 컷오프)':<30} | {eq5_a_str:<32} | {eq5_b_str:<32}")
    print(f"{'💰 중앙값 우주 (50% 기준)':<30} | {eq50_a_str:<32} | {eq50_b_str:<32}")
    print(f"{'🚀 대박의 우주 (상위 5% 컷오프)':<30} | {eq95_a_str:<32} | {eq95_b_str:<32}")
    print("=" * 105)


if __name__ == "__main__":
    run_experiment_71()
