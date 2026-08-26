"""
[실험 58] 몬테카를로 10,000회 시뮬레이션 기반 파산 확률(Risk of Ruin) 정밀 측정
- 비교 대상:
  1. STANDARD_GOLDEN (공식 표준: 추세 2.0% x 횡보 4.0%, 현금 관망, 3.0x)
  2. MONSTER_EXTREME_100X (수익률 1위 몬스터: 추세 4.0% x 횡보 20.0% + 80% 숏, 100.0x)
- 측정 지표:
  - 잔고 -50% 반토막 도달 확률 (초기 자본 대비)
  - 잔고 -70% 폭락 도달 확률
  - 잔고 -80% 치명상 도달 확률
  - 잔고 -90% 깡통/청산 도달 확률
  - 최종 자산 5% 백분위(최악), 50% 백분위(중앙값), 95% 백분위(최상)
  - 몬테카를로 MDD 95% 신뢰구간
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


def run_monte_carlo(trades_return_pct: np.ndarray, num_sims: int = 10000, num_trades: int = None):
    """몬테카를로 부트스트래핑 10,000회 실행"""
    if num_trades is None:
        num_trades = len(trades_return_pct)

    np.random.seed(42)
    # 복원 추출(Sampling with replacement) 10,000 x N
    sampled_indices = np.random.choice(len(trades_return_pct), size=(num_sims, num_trades), replace=True)
    sampled_returns = trades_return_pct[sampled_indices] / 100.0  # 비율로 변환

    # 각 시뮬레이션의 자산 경로 계산 (초기 10,000)
    # shape: (10000, num_trades + 1)
    multipliers = 1.0 + sampled_returns
    # 0 이하 클리핑 방지
    multipliers = np.maximum(multipliers, 1e-4)

    equity_paths = np.zeros((num_sims, num_trades + 1))
    equity_paths[:, 0] = 10000.0

    for step in range(num_trades):
        equity_paths[:, step + 1] = equity_paths[:, step] * multipliers[:, step]

    final_equities = equity_paths[:, -1]

    # 각 시뮬레이션별 최소 자본 및 MDD 계산
    min_equities = np.min(equity_paths, axis=1)

    peaks = np.maximum.accumulate(equity_paths, axis=1)
    dds = (peaks - equity_paths) / peaks
    max_dds = np.max(dds, axis=1) * 100.0

    # 파산 확률 계산
    # 1. 초기 원금 대비 -50% (5,000 이하 도달)
    ruin_50 = np.mean(min_equities <= 5000.0) * 100.0
    # 2. 초기 원금 대비 -70% (3,000 이하 도달)
    ruin_70 = np.mean(min_equities <= 3000.0) * 100.0
    # 3. 초기 원금 대비 -80% (2,000 이하 도달)
    ruin_80 = np.mean(min_equities <= 2000.0) * 100.0
    # 4. 초기 원금 대비 -90% (1,000 이하 도달 - 사실상 깡통)
    ruin_90 = np.mean(min_equities <= 1000.0) * 100.0

    return {
        "final_median": np.median(final_equities),
        "final_p5": np.percentile(final_equities, 5),
        "final_p95": np.percentile(final_equities, 95),
        "mdd_median": np.median(max_dds),
        "mdd_p95": np.percentile(max_dds, 95),
        "mdd_max": np.max(max_dds),
        "ruin_50": ruin_50,
        "ruin_70": ruin_70,
        "ruin_80": ruin_80,
        "ruin_90": ruin_90,
    }


def run_experiment_58():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    print("=" * 105)
    print("      [실험 58] 몬테카를로 10,000회 시뮬레이션 기반 파산 확률(Risk of Ruin) 정밀 측정")
    print("=" * 105)

    # 1. 표준 모델
    reg_mgr_74 = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.74, cooldown_bars=0)
    df_proc_74 = reg_mgr_74.calculate_regime_probabilities(df_ind)
    test_df_cash = df_proc_74.iloc[720:].reset_index(drop=True)

    sim_m1 = BacktestSimulator(
        initial_capital=10000.0, trend_risk_pct=0.020, mr_risk_pct=0.040, leverage=3.0, bear_mode="CASH",
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_m1 = sim_m1.run(test_df_cash)
    ret_m1 = res_m1["trades_df"]["return_pct"].values

    # 2. 몬스터 모델
    df_asym = get_asymmetric_df(df_ind, base_th=0.74, bear_th=0.80)
    test_df_asym = df_asym.iloc[720:].reset_index(drop=True)

    sim_m2 = BacktestSimulator(
        initial_capital=10000.0, trend_risk_pct=0.040, mr_risk_pct=0.200, leverage=100.0, bear_mode="SHORT",
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    sim_m2.pos_manager.max_leverage = 100.0
    sim_m2.pos_manager.default_leverage = 100.0
    res_m2 = sim_m2.run(test_df_asym)
    ret_m2 = res_m2["trades_df"]["return_pct"].values

    print("\n[몬테카를로 10,000개 평행우주 시뮬레이션 연산 중...]")
    mc_m1 = run_monte_carlo(ret_m1, num_sims=10000)
    mc_m2 = run_monte_carlo(ret_m2, num_sims=10000)

    print("\n" + "=" * 105)
    print(f"{'측정 지표 항목':<32} | {'STANDARD_GOLDEN (공식 표준)':<32} | {'MONSTER_EXTREME_100X (100x 몬스터)':<32}")
    print("-" * 105)
    print(f"{'단일 백테스트 최종 자산':<32} | ${res_m1['final_equity']:,.2f} (+{res_m1['total_return_pct']:.1f}%) | ${res_m2['final_equity']:,.2f} (+{res_m2['total_return_pct']:.1f}%)")
    print(f"{'단일 백테스트 MDD':<32} | {res_m1['mdd_pct']:.2f}% | {res_m2['mdd_pct']:.2f}%")
    print("-" * 105)
    print(f"{'10,000회 시뮬 중간값 자산 (Median)':<32} | ${mc_m1['final_median']:,.2f} | ${mc_m2['final_median']:,.2f}")
    print(f"{'10,000회 최악 5% 자산 (Worst 5%)':<32} | ${mc_m1['final_p5']:,.2f} | ${mc_m2['final_p5']:,.2f}")
    print(f"{'10,000회 최상 5% 자산 (Best 5%)':<32} | ${mc_m1['final_p95']:,.2f} | ${mc_m2['final_p95']:,.2f}")
    print("-" * 105)
    print(f"{'10,000회 시뮬 평균 MDD':<32} | {mc_m1['mdd_median']:.2f}% | {mc_m2['mdd_median']:.2f}%")
    print(f"{'10,000회 최악의 MDD (95% 신뢰구간)':<32} | {mc_m1['mdd_p95']:.2f}% | {mc_m2['mdd_p95']:.2f}%")
    print(f"{'10,000회 역사상 최악의 MDD (Worst)':<32} | {mc_m1['mdd_max']:.2f}% | {mc_m2['mdd_max']:.2f}%")
    print("-" * 105)
    print(f"{'원금 -50% 반토막 파산 확률':<32} | {mc_m1['ruin_50']:.2f}% | {mc_m2['ruin_50']:.2f}%")
    print(f"{'원금 -70% 폭락 파산 확률':<32} | {mc_m1['ruin_70']:.2f}% | {mc_m2['ruin_70']:.2f}%")
    print(f"{'원금 -80% 치명상 파산 확률':<32} | {mc_m1['ruin_80']:.2f}% | {mc_m2['ruin_80']:.2f}%")
    print(f"{'원금 -90% 전액 깡통 파산 확률':<32} | {mc_m1['ruin_90']:.2f}% | {mc_m2['ruin_90']:.2f}%")
    print("=" * 105)


if __name__ == "__main__":
    run_experiment_58()
