"""
RADE 시스템 통계적 유의성 검정 모듈 (Statistical Significance & Hypothesis Testing)
- Bootstrap Confidence Interval (1,000회 리샘플링을 통한 95% 신뢰구간 추정)
- Monte Carlo Permutation Test (무작위 셔플 가설 검정 및 엣지 p-value 산출)
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from python.utils.indicators import add_all_indicators
from python.regime.regime_manager import RegimeManager
from python.backtest.simulator import BacktestSimulator


def run_bootstrap_analysis(trades_df: pd.DataFrame, n_iterations: int = 1000, initial_capital: float = 10000.0) -> Dict[str, Any]:
    """거래별 PnL 부트스트랩 리샘플링을 통한 95% 신뢰구간 산출"""
    if trades_df.empty:
        return {}

    pnls = trades_df['pnl'].values
    n_trades = len(pnls)

    boot_returns = []
    boot_pfs = []
    boot_win_rates = []

    np.random.seed(42)

    for _ in range(n_iterations):
        sample_pnls = np.random.choice(pnls, size=n_trades, replace=True)
        tot_pnl = sample_pnls.sum()
        ret_pct = (tot_pnl / initial_capital) * 100.0

        wins = sample_pnls[sample_pnls > 0]
        losses = sample_pnls[sample_pnls < 0]

        gross_win = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-10
        pf = gross_win / gross_loss
        wr = (len(wins) / n_trades) * 100.0

        boot_returns.append(ret_pct)
        boot_pfs.append(pf)
        boot_win_rates.append(wr)

    return {
        "return_mean": float(np.mean(boot_returns)),
        "return_ci_95": (float(np.percentile(boot_returns, 2.5)), float(np.percentile(boot_returns, 97.5))),
        "pf_mean": float(np.mean(boot_pfs)),
        "pf_ci_95": (float(np.percentile(boot_pfs, 2.5)), float(np.percentile(boot_pfs, 97.5))),
        "win_rate_mean": float(np.mean(boot_win_rates)),
        "win_rate_ci_95": (float(np.percentile(boot_win_rates, 2.5)), float(np.percentile(boot_win_rates, 97.5))),
    }


def run_permutation_test(trades_df: pd.DataFrame, n_iterations: int = 1000) -> Dict[str, Any]:
    """
    Monte Carlo 순열 검정 (Permutation Test)
    - 귀무가설(H0): 이 전략의 손익은 무작위 동전 던지기(50:50) 결과와 다르지 않다 (엣지가 없다).
    - p-value: 무작위 부호 반전 시뮬레이션에서 실제 총 손익 이상의 수익이 발생할 확률
    """
    if trades_df.empty:
        return {"p_value": 1.0}

    pnls = trades_df['pnl'].values
    actual_total_pnl = pnls.sum()
    abs_pnls = np.abs(pnls)
    n_trades = len(pnls)

    np.random.seed(42)
    random_totals = []

    for _ in range(n_iterations):
        # 50% 확률로 손익 부호 무작위 반전 (+1 or -1)
        signs = np.random.choice([1, -1], size=n_trades)
        random_pnl = (abs_pnls * signs).sum()
        random_totals.append(random_pnl)

    random_totals = np.array(random_totals)
    # p-value = (무작위 결과 >= 실제 결과인 횟수) / 전체 반복 횟수
    p_val = float(np.mean(random_totals >= actual_total_pnl))

    return {
        "actual_pnl": float(actual_total_pnl),
        "random_pnl_mean": float(np.mean(random_totals)),
        "random_pnl_std": float(np.std(random_totals)),
        "p_value": p_val,
        "is_significant_5pct": (p_val < 0.05),
        "is_significant_1pct": (p_val < 0.01),
    }


def run_full_statistical_test():
    print("=== [P1 통계적 유의성 검정] 3.5년 백테스트 부트스트랩 및 몬테카를로 분석 시작 ===")

    cache_file = os.path.join("data", "BTCUSDT_1h_2021_2024.csv")
    df_raw = pd.read_csv(cache_file)
    df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)

    df_ind = add_all_indicators(df_raw)
    manager = RegimeManager(hmm_window=720, retrain_interval=168, hysteresis_upper=0.65, hysteresis_lower=0.35, cooldown_bars=3)
    df_proc = manager.calculate_regime_probabilities(df_ind)
    test_df = df_proc.dropna(subset=['regime_trend_prob']).reset_index(drop=True)

    sim = BacktestSimulator(initial_capital=10000.0, risk_per_trade_pct=0.02, leverage=3.0)
    res = sim.run(test_df)
    trades_df = res['trades_df']

    print(f"\n[기본 성과] 총 거래: {res['total_trades']}회 | 총 수익률: {res['total_return_pct']:+.2f}% | MDD: {res['mdd_pct']:.2f}% | PF: {res['profit_factor']:.2f}")

    # 1. Bootstrap 95% 신뢰구간
    print("\n1. Bootstrap 1,000회 리샘플링 분석 중...")
    boot_res = run_bootstrap_analysis(trades_df, n_iterations=1000, initial_capital=10000.0)
    print("=" * 65)
    print("         [ BOOTSTRAP 95% CONFIDENCE INTERVALS ]         ")
    print("=" * 65)
    print(f" * 3.5년 총수익률 평균:   {boot_res['return_mean']:+.2f}%")
    print(f" * 95% 신뢰구간 (Return): [{boot_res['return_ci_95'][0]:+.2f}%, {boot_res['return_ci_95'][1]:+.2f}%]")
    print(f" * Profit Factor 평균:    {boot_res['pf_mean']:.2f}")
    print(f" * 95% 신뢰구간 (PF):     [{boot_res['pf_ci_95'][0]:.2f}, {boot_res['pf_ci_95'][1]:.2f}]")
    print(f" * 승률(Win Rate) 평균:   {boot_res['win_rate_mean']:.1f}%")
    print(f" * 95% 신뢰구간 (WinRate):[{boot_res['win_rate_ci_95'][0]:.1f}%, {boot_res['win_rate_ci_95'][1]:.1f}%]")
    print("=" * 65)

    # 2. Monte Carlo 순열 검정 (p-value)
    print("\n2. Monte Carlo 순열 검정 (1,000회 무작위화 가설 검정) 중...")
    perm_res = run_permutation_test(trades_df, n_iterations=1000)
    print("=" * 65)
    print("          [ MONTE CARLO PERMUTATION TEST (p-value) ]          ")
    print("=" * 65)
    print(f" * 실제 전략 누적 손익 (Actual PnL):  ${perm_res['actual_pnl']:+,.2f}")
    print(f" * 무작위 무엣지 손익 평균:          ${perm_res['random_pnl_mean']:+,.2f}")
    print(f" * 무작위 손익 표준편차:              ${perm_res['random_pnl_std']:,.2f}")
    print(f" * 엣지 검정 p-value:                {perm_res['p_value']:.4f}")
    print(f" * 5% 유의수준 (p < 0.05):           {' 통과 (진짜 엣지 입증)' if perm_res['is_significant_5pct'] else 'X 미달 (운일 가능성)'}")
    print(f" * 1% 유의수준 (p < 0.01):           {' 통과 (매우 강력한 엣지)' if perm_res['is_significant_1pct'] else 'X 미달'}")
    print("=" * 65)


if __name__ == "__main__":
    run_full_statistical_test()
