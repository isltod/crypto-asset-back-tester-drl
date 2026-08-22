"""
[실험 21] HMM 캘린더 앵커 요일 전수 검증 (월 vs 화 vs 수 vs 목 vs 금 vs 토 vs 일)
- 목적: 7개 요일별(00:00 UTC) HMM 재학습 백테스트를 전수 수행하여 요일별 성과 추세 및 최적 앵커 규명
- 데이터: 2021.01.01 ~ 2024.12.31 (4개년 풀데이터)
- 조건: 2% Risk, 3.0x 레버리지, 3-State HMM (Cash Mode), 동적 4.0x ATR 트레일링, 실전 수수료/슬리피지
"""
import os
import sys
import time
import warnings
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.backtest.simulator import BacktestSimulator


def fit_hmm_slice_fast(train_matrix: np.ndarray) -> Tuple[float, float, float, str]:
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


def calculate_regimes_by_anchor_day(df_ind: pd.DataFrame, anchor_dayofweek: int, hmm_window: int = 720) -> pd.DataFrame:
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
        is_anchor_time = (curr_dt.dayofweek == anchor_dayofweek and curr_dt.hour == 0)
        
        if not has_first_fit or is_anchor_time:
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


def run_experiment_21():
    print("==================================================================================")
    print(f"[{time.strftime('%X')}] === [실험 21] HMM 캘린더 앵커 요일(월~일) 전수 백테스트 시작 ===")
    print("==================================================================================")
    
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_all = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values(by='timestamp').reset_index(drop=True)
    df_all['datetime'] = pd.to_datetime(df_all['timestamp'], unit='ms', utc=True)
    
    print(f"[{time.strftime('%X')}] 4년 전체 {len(df_all)}개 캔들 로드 완료. 기술적 지표 계산 중...")
    df_ind = add_all_indicators(df_all)
    
    weekdays = [
        {"day_name": "월요일 (Monday 00:00 UTC)",    "dayofweek": 0},
        {"day_name": "화요일 (Tuesday 00:00 UTC)",   "dayofweek": 1},
        {"day_name": "수요일 (Wednesday 00:00 UTC)", "dayofweek": 2},
        {"day_name": "목요일 (Thursday 00:00 UTC)",  "dayofweek": 3},
        {"day_name": "금요일 (Friday 00:00 UTC)",    "dayofweek": 4},
        {"day_name": "토요일 (Saturday 00:00 UTC)",  "dayofweek": 5},
        {"day_name": "일요일 (Sunday 00:00 UTC)",    "dayofweek": 6},
    ]
    
    results = []
    sim = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        maker_fee_pct=0.0002,
        taker_fee_pct=0.0005,
        slippage_pct=0.0002,
        funding_fee_pct=0.0001
    )
    
    for w in weekdays:
        t0 = time.time()
        print(f"\n[{time.strftime('%X')}] >> 앵커 테스트: {w['day_name']}...")
        df_proc = calculate_regimes_by_anchor_day(df_ind, anchor_dayofweek=w['dayofweek'], hmm_window=720)
        res = sim.run(df_proc.iloc[720:].reset_index(drop=True))
        elapsed = time.time() - t0
        
        res['day_name'] = w['day_name']
        res['dayofweek'] = w['dayofweek']
        res['elapsed_sec'] = elapsed
        results.append(res)
        print(f"[{time.strftime('%X')}] [완료] ({elapsed:.1f}초): 수익률 {res['total_return_pct']:+.2f}%, MDD {res['mdd_pct']:.2f}%, 거래 {res['total_trades']}회, PF {res['profit_factor']:.2f}")

    print("\n\n" + "=" * 105)
    print(f"[{time.strftime('%X')}]       [실험 21] HMM 재학습 앵커 요일별 4개년(2021~2024) 성과 종합 비교표")
    print("=" * 105)
    print(f"{'앵커 요일 (00:00 UTC)':<30} | {'총수익률':<12} | {'MDD':<10} | {'거래횟수':<10} | {'승률':<10} | {'손익비(PF)':<12} | {'최종자산'}")
    print("-" * 105)
    
    # 수익률 내림차순 정렬
    sorted_results = sorted(results, key=lambda x: x['total_return_pct'], reverse=True)
    for r in sorted_results:
        name = r['day_name']
        ret = f"{r['total_return_pct']:+.2f}%"
        mdd = f"{r['mdd_pct']:.2f}%"
        cnt = f"{r['total_trades']}회"
        wr = f"{r['win_rate_pct']:.2f}%"
        pf = f"{r['profit_factor']:.2f}"
        eq = f"${r['final_equity']:,.2f}"
        print(f"{name:<30} | {ret:<12} | {mdd:<10} | {cnt:<10} | {wr:<10} | {pf:<12} | {eq}")
    print("=" * 105)


if __name__ == "__main__":
    run_experiment_21()
