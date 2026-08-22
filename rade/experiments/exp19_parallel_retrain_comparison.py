"""
[실험 19] 멀티프로세싱 초고속 병렬화 기반 HMM 재학습 주기 비교 검증 (4년 풀데이터 2021~2024)
- 비교군:
  1) retrain_interval = 168 (1주일 주기, 기존 백테스트 기준선)
  2) retrain_interval = 24  (매 1일 주기, 일 단위 연속 안정화)
  3) retrain_interval = 1   (매 1시간 주기, 현재 실전 페이퍼 트레이더와 100% 동일)
- 최적화: ProcessPoolExecutor 멀티코어 병렬 연산 + 순수 NumPy 고속 행렬 연산 + 경고 출력 차단
"""
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# 경고 억제
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.backtest.simulator import BacktestSimulator


def _fit_single_hmm_slice(train_data: np.ndarray) -> Tuple[float, float, float, str]:
    """단일 720봉 NumPy 슬라이스에 대해 HMM을 고속 피팅하고 마지막 봉의 확률과 국면을 반환"""
    import warnings
    warnings.filterwarnings('ignore')
    from hmmlearn.hmm import GaussianHMM
    
    try:
        X = np.nan_to_num(train_data, nan=0.0, posinf=0.0, neginf=0.0)
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


def _parallel_worker_batch(task_args: List[Tuple[np.ndarray, int]]) -> List[Tuple[int, float, float, float, str]]:
    """배치 단위 병렬 워커"""
    results = []
    for train_arr, idx in task_args:
        p_range, p_bull, p_bear, regime = _fit_single_hmm_slice(train_arr)
        results.append((idx, p_range, p_bull, p_bear, regime))
    return results


def calculate_regimes_fast_parallel(df_ind: pd.DataFrame, hmm_window: int = 720, retrain_interval: int = 1) -> pd.DataFrame:
    """멀티코어를 활용한 고속 병렬 HMM 국면 계산"""
    df = df_ind.copy()
    feature_cols = ['return', 'vol_change', 'atr_ratio']
    feat_matrix = df[feature_cols].values
    n = len(df)
    
    train_indices = []
    for i in range(hmm_window, n):
        if (i - hmm_window) % retrain_interval == 0:
            train_indices.append(i)
    
    print(f"[{time.strftime('%X')}] HMM 재학습 대상: 총 {len(train_indices)}개 시점 (retrain_interval={retrain_interval})")
    
    tasks = []
    for i in train_indices:
        train_slice = feat_matrix[i - hmm_window : i]
        tasks.append((train_slice, i))
    
    cpu_cores = os.cpu_count() or 4
    batch_size = max(1, len(tasks) // (cpu_cores * 4))
    batches = [tasks[k:k+batch_size] for k in range(0, len(tasks), batch_size)]
    
    print(f"[{time.strftime('%X')}] CPU {cpu_cores}개 코어로 {len(batches)}개 배치 병렬 연산 시작...")
    
    eval_dict = {}
    with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        futures = [executor.submit(_parallel_worker_batch, b) for b in batches]
        for f in futures:
            batch_res = f.result()
            for idx, p_range, p_bull, p_bear, regime in batch_res:
                eval_dict[idx] = (p_range, p_bull, p_bear, regime)
                
    print(f"[{time.strftime('%X')}] 병렬 연산 완료! 전체 타임라인 국면 매핑 중...")
    
    p_range_arr = np.full(n, np.nan)
    p_bull_arr = np.full(n, np.nan)
    p_bear_arr = np.full(n, np.nan)
    regime_arr = np.full(n, "RANGE", dtype=object)
    
    last_val = (0.34, 0.33, 0.33, "RANGE")
    for i in range(hmm_window, n):
        if i in eval_dict:
            last_val = eval_dict[i]
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


def run_experiment_19():
    print("==================================================================================")
    print(f"[{time.strftime('%X')}] === [실험 19] HMM 재학습 주기(1주일 vs 1일 vs 매시간) 4년 백테스트 시작 ===")
    print("==================================================================================")
    
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_is = pd.read_csv(f_is)
    df_oos = pd.read_csv(f_oos)
    df_all = pd.concat([df_is, df_oos], ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values(by='timestamp').reset_index(drop=True)
    df_all['datetime'] = pd.to_datetime(df_all['timestamp'], unit='ms', utc=True)
    
    print(f"[{time.strftime('%X')}] 4년 전체 {len(df_all)}개 캔들 로드 완료. 기술적 지표 계산 중...")
    df_ind = add_all_indicators(df_all)
    
    configs = [
        {"name": "1주일 주기 (retrain=168)", "interval": 168},
        {"name": "1일 주기 (retrain=24)",     "interval": 24},
        {"name": "매 1시간 주기 (실전 동일, retrain=1)", "interval": 1},
    ]
    
    results = []
    for cfg in configs:
        t0 = time.time()
        print(f"[{time.strftime('%X')}] -------------------------------------------------------------")
        print(f"[{time.strftime('%X')}] >> 테스트 실행: {cfg['name']}")
        print(f"[{time.strftime('%X')}] -------------------------------------------------------------")
        
        df_proc = calculate_regimes_fast_parallel(df_ind, hmm_window=720, retrain_interval=cfg['interval'])
        test_df = df_proc.iloc[720:].reset_index(drop=True)
        
        sim = BacktestSimulator(
            initial_capital=10000.0,
            risk_per_trade_pct=0.02,
            leverage=3.0,
            maker_fee_pct=0.0002,
            taker_fee_pct=0.0005,
            slippage_pct=0.0002,
            funding_fee_pct=0.0001
        )
        res = sim.run(test_df)
        elapsed = time.time() - t0
        
        res['name'] = cfg['name']
        res['interval'] = cfg['interval']
        res['elapsed_sec'] = elapsed
        
        trades_df = res['trades_df']
        if not trades_df.empty:
            trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
            res['yearly_counts'] = trades_df.groupby('year')['pnl'].count().to_dict()
            res['yearly_pnls'] = trades_df.groupby('year')['pnl'].sum().to_dict()
        else:
            res['yearly_counts'] = {}
            res['yearly_pnls'] = {}
            
        results.append(res)
        print(f"[{time.strftime('%X')}] [완료] ({elapsed:.1f}초 소요): 수익률 {res['total_return_pct']:+.2f}%, MDD {res['mdd_pct']:.2f}%, 거래 {res['total_trades']}회")

    print("\n\n" + "=" * 105)
    print(f"[{time.strftime('%X')}]       [실험 19] HMM 재학습 주기별 4개년(2021~2024) 성과 종합 비교표")
    print("=" * 105)
    print(f"{'재학습 주기':<32} | {'총수익률':<12} | {'MDD':<10} | {'거래횟수':<10} | {'승률':<10} | {'손익비(PF)':<12} | {'최종자산'}")
    print("-" * 105)
    for r in results:
        name = r['name']
        ret = f"{r['total_return_pct']:+.2f}%"
        mdd = f"{r['mdd_pct']:.2f}%"
        cnt = f"{r['total_trades']}회"
        wr = f"{r['win_rate_pct']:.2f}%"
        pf = f"{r['profit_factor']:.2f}"
        eq = f"${r['final_equity']:,.2f}"
        print(f"{name:<32} | {ret:<12} | {mdd:<10} | {cnt:<10} | {wr:<10} | {pf:<12} | {eq}")
    print("=" * 105)

    print("\n--- [연도별 순손익(PnL) 및 거래 횟수 분해] ---")
    for r in results:
        print(f"\n▶ {r['name']}:")
        for yr in sorted(r['yearly_counts'].keys()):
            cnt = r['yearly_counts'].get(yr, 0)
            pnl = r['yearly_pnls'].get(yr, 0.0)
            print(f"   [{yr}년] 거래: {cnt:3d}회 | 순익: ${pnl:+9.2f}")


if __name__ == "__main__":
    run_experiment_19()
