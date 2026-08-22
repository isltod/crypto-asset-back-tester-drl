"""
[실험 20] 캘린더 앵커링(Calendar-Anchored Retraining) 기반 위상 동기화 및 4년 vs 1.5년 성과 일치성 정밀 검증
- 원리: 데이터 시작 시점과 무관하게 매주 월요일 00:00 UTC에만 HMM 재학습 수행
- 비교군:
  1) 2021.01.01 시작 4년 데이터 (2023.01~2024.06 평가 구간 추출)
  2) 2022.01.01 시작 선행 웜업 데이터 (2023.01~2024.06 평가 구간 추출)
- 목적: 캘린더 앵커링 적용 시 두 환경의 국면 판정 및 거래 성과가 100% 일치하는지 전수 검증
"""
import os
import sys
import time
import warnings
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# 경고 억제
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.backtest.simulator import BacktestSimulator


def fit_hmm_slice_fast(train_matrix: np.ndarray) -> Tuple[float, float, float, str]:
    """단일 720봉 슬라이스에 대한 고속 HMM 학습 및 확률 반환"""
    try:
        X = np.nan_to_num(train_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        model = GaussianHMM(n_components=3, covariance_type="full", min_covar=1e-3, n_iter=100, random_state=42)
        model.fit(X)
        
        mean_returns = model.means_[:, 0]
        mean_atrs = model.means_[:, 1]
        bull_candidate = int(np.argmax(mean_returns))
        remaining = [i for i in range(3) if i != bull_candidate]
        if mean_atrs[remaining[0]] < mean_atrs[remaining[1]]:
            range_candidate = remaining[0]
            bear_candidate = remaining[1]
        else:
            range_candidate = remaining[1]
            bear_candidate = remaining[0]
            
        posteriors = model.predict_proba(X[-1:])
        p_range = float(posteriors[0, range_candidate])
        p_bull = float(posteriors[0, bull_candidate])
        p_bear = float(posteriors[0, bear_candidate])
        
        if p_bull >= 0.45 and p_bull > p_bear:
            regime = "BULL_TREND"
        elif p_bear >= 0.45 and p_bear > p_bull:
            regime = "BEAR_PANIC"
        else:
            regime = "RANGE"
            
        return (p_range, p_bull, p_bear, regime)
    except Exception:
        return (0.34, 0.33, 0.33, "RANGE")


def calculate_calendar_anchored_regimes(df_ind: pd.DataFrame, hmm_window: int = 720) -> pd.DataFrame:
    """매주 월요일 00:00 UTC에만 재학습하는 캘린더 앵커링 국면 계산기"""
    df = df_ind.copy()
    feature_cols = ['return', 'vol_change', 'atr_ratio']
    feat_matrix = df[feature_cols].values
    dt_series = pd.to_datetime(df['datetime'], utc=True)
    n = len(df)
    
    p_range_arr = np.full(n, np.nan)
    p_bull_arr = np.full(n, np.nan)
    p_bear_arr = np.full(n, np.nan)
    regime_arr = np.full(n, "RANGE", dtype=object)
    
    last_val = (0.34, 0.33, 0.33, "RANGE")
    has_first_fit = False
    
    for i in range(hmm_window, n):
        curr_dt = dt_series.iloc[i]
        
        # 캘린더 앵커 조건: 매주 월요일 00:00 UTC (dayofweek==0, hour==0) 또는 첫 진입 시점
        is_monday_midnight = (curr_dt.dayofweek == 0 and curr_dt.hour == 0)
        
        if not has_first_fit or is_monday_midnight:
            train_slice = feat_matrix[i - hmm_window : i]
            last_val = fit_hmm_slice_fast(train_slice)
            has_first_fit = True
            
        p_range_arr[i] = last_val[0]
        p_bull_arr[i] = last_val[1]
        p_bear_arr[i] = last_val[2]
        regime_arr[i] = last_val[3]
        
    df['regime_state'] = regime_arr
    df['p_range'] = p_range_arr
    df['p_bull'] = p_bull_arr
    df['p_bear'] = p_bear_arr
    df['regime_trend_prob'] = p_bull_arr
    df['regime_mr_prob'] = p_range_arr
    
    return df


def run_experiment_20():
    print("==================================================================================")
    print(f"[{time.strftime('%X')}] === [실험 20] 캘린더 앵커링 기반 위상 동기화 및 4년 vs 1.5년 일치성 검증 ===")
    print("==================================================================================")

    # 1. 2021년 시작 4년 풀데이터 로드
    print(f"\n[{time.strftime('%X')}] [1/2] 2021년 시작 4년 풀데이터 로드 및 캘린더 앵커링 HMM 계산 중...")
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df21 = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values(by='timestamp').reset_index(drop=True)
    df21['datetime'] = pd.to_datetime(df21['timestamp'], unit='ms', utc=True)
    df21_ind = add_all_indicators(df21)
    df21_proc = calculate_calendar_anchored_regimes(df21_ind, hmm_window=720)

    # 2. 2022년 시작 웜업 데이터 로드
    print(f"[{time.strftime('%X')}] [2/2] 2022년 시작 웜업 데이터 로드 및 캘린더 앵커링 HMM 계산 중...")
    f22 = "data/BTCUSDT_1h_2022_2024.csv"
    df22 = pd.read_csv(f22)
    df22['datetime'] = pd.to_datetime(df22['timestamp'], unit='ms', utc=True)
    df22_ind = add_all_indicators(df22)
    df22_proc = calculate_calendar_anchored_regimes(df22_ind, hmm_window=720)

    # 3. 2023.01.01 ~ 2024.06.01 (1.5년) 동일 평가 구간 추출 및 국면 일치율 전수 검사
    print(f"\n[{time.strftime('%X')}] >> [2023.01.01 ~ 2024.06.01] 12,409개 캔들 1:1 국면 일치율 검사 중...")
    sub21 = df21_proc[(df21_proc['datetime'] >= '2023-01-01') & (df21_proc['datetime'] <= '2024-06-01')].reset_index(drop=True)
    sub22 = df22_proc[(df22_proc['datetime'] >= '2023-01-01') & (df22_proc['datetime'] <= '2024-06-01')].reset_index(drop=True)

    match_count = (sub21['regime_state'] == sub22['regime_state']).sum()
    match_pct = (match_count / len(sub21)) * 100.0
    print(f"[{time.strftime('%X')}] [HMM 국면 판정 일치율]: {match_count:,} / {len(sub21):,} ({match_pct:.2f}%)")

    # 4. 백테스트 시뮬레이터 실행 및 1.5년 평가 성과 산출
    sim = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        maker_fee_pct=0.0002,
        taker_fee_pct=0.0005,
        slippage_pct=0.0002,
        funding_fee_pct=0.0001
    )

    # 2021년 시작 데이터의 백테스트
    res21_full = sim.run(df21_proc.iloc[720:].reset_index(drop=True))
    t21 = res21_full['trades_df']
    t21['dt'] = pd.to_datetime(t21['entry_time'])
    t21_eval = t21[(t21['dt'] >= '2023-01-01') & (t21['dt'] <= '2024-06-01')].copy()

    # 2022년 시작 데이터의 백테스트
    res22_full = sim.run(df22_proc.iloc[720:].reset_index(drop=True))
    t22 = res22_full['trades_df']
    t22['dt'] = pd.to_datetime(t22['entry_time'])
    t22_eval = t22[(t22['dt'] >= '2023-01-01') & (t22['dt'] <= '2024-06-01')].copy()

    def calc_eval_stats(trades):
        if trades.empty:
            return 0.0, 0.0, 0, 0.0, 0.0, 10000.0
        wins = trades[trades['pnl'] > 0]
        losses = trades[trades['pnl'] < 0]
        total_pnl = trades['pnl'].sum()
        gp = wins['pnl'].sum()
        gl = abs(losses['pnl'].sum())
        pf = gp / gl if gl > 0 else 999.0
        wr = len(wins) / len(trades) * 100.0
        ret_pct = (total_pnl / 10000.0) * 100.0
        
        eq = [10000.0]
        c = 10000.0
        for p in trades['pnl']:
            c += p
            eq.append(c)
        peaks = np.maximum.accumulate(eq)
        dds = (peaks - eq) / peaks * 100.0
        mdd = np.max(dds) if len(dds) > 0 else 0.0
        return ret_pct, mdd, len(trades), wr, pf, (10000.0 + total_pnl)

    r21_ret, r21_mdd, r21_cnt, r21_wr, r21_pf, r21_eq = calc_eval_stats(t21_eval)
    r22_ret, r22_mdd, r22_cnt, r22_wr, r22_pf, r22_eq = calc_eval_stats(t22_eval)

    print("\n\n" + "=" * 95)
    print("   [실험 20] 캘린더 앵커링 적용 시 2023~2024 (1.5년 평가 구간) 성과 1:1 비교표")
    print("=" * 95)
    print(f"{'시작 시점':<25} | {'총수익률':<12} | {'MDD':<10} | {'거래횟수':<10} | {'승률':<10} | {'손익비(PF)':<12} | {'평가자산'}")
    print("-" * 95)
    print(f"{'2021년 시작 (4년 풀)':<25} | {r21_ret:+.2f}%     | {r21_mdd:.2f}%     | {r21_cnt}회       | {r21_wr:.2f}%     | {r21_pf:.2f}         | ${r21_eq:,.2f}")
    print(f"{'2022년 시작 (1년 웜업)':<25} | {r22_ret:+.2f}%     | {r22_mdd:.2f}%     | {r22_cnt}회       | {r22_wr:.2f}%     | {r22_pf:.2f}         | ${r22_eq:,.2f}")
    print("=" * 95)


if __name__ == "__main__":
    run_experiment_20()
