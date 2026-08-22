"""
[실험 22] 올바른 주간 앵커링(주 1회 피팅 + 매시간 최신 추론) 7개 전 요일(월~일) 전수 백테스트
- 구조:
  1) HMM 학습(Fit): 7개 각 요일(00:00 UTC)마다 주 1회 정기 수행 (720봉 윈도우)
  2) HMM 추론(Predict Proba): 매 1시간마다 최신 캔들을 모델에 넣어 실시간 국면 확률 추론
- 데이터: 2021.01.01 ~ 2024.12.31 (4개년 풀데이터)
- 조건: 2% Risk, 3.0x 레버리지, 3-State HMM (Cash Mode), 동적 4.0x ATR 트레일링, 실전 수수료/슬리피지
"""
import os
import sys
import time
import warnings
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.backtest.simulator import BacktestSimulator


def calculate_true_calendar_regimes(df_ind: pd.DataFrame, anchor_dayofweek: int, hmm_window: int = 720, trans_threshold: float = 0.45) -> pd.DataFrame:
    data = df_ind.copy()
    n = len(data)
    features = data[["return", "atr_ratio", "vol_change"]].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    dts = pd.to_datetime(data["datetime"], utc=True)

    states = ["RANGE"] * n
    p_ranges = np.full(n, np.nan)
    p_bulls = np.full(n, np.nan)
    p_bears = np.full(n, np.nan)
    curr_state = "RANGE"

    model = None
    bull_idx, range_idx, bear_idx = 1, 0, 2

    for i in range(hmm_window, n):
        curr_dt = dts.iloc[i]
        is_anchor = (curr_dt.dayofweek == anchor_dayofweek and curr_dt.hour == 0)

        # 1. 고정 앵커 요일 00:00 UTC에만 HMM 모델 재학습
        if model is None or is_anchor:
            X_train = features[i - hmm_window : i]
            m = GaussianHMM(n_components=3, covariance_type="full", min_covar=1e-3, n_iter=100, random_state=42)
            try:
                m.fit(X_train)
                model = m
                mean_returns = m.means_[:, 0]
                mean_atrs = m.means_[:, 1]
                bull_cand = int(np.argmax(mean_returns))
                rem = [k for k in range(3) if k != bull_cand]
                if mean_atrs[rem[0]] < mean_atrs[rem[1]]:
                    range_cand, bear_cand = rem[0], rem[1]
                else:
                    range_cand, bear_cand = rem[1], rem[0]
                bull_idx, range_idx, bear_idx = bull_cand, range_cand, bear_cand
            except Exception:
                pass

        # 2. 매 1시간마다 최신 캔들 데이터로 실시간 국면 확률 추론
        if model is not None:
            try:
                post = model.predict_proba(features[max(0, i-100) : i+1])
                last_p = post[-1]
                p_r = float(last_p[range_idx])
                p_u = float(last_p[bull_idx])
                p_d = float(last_p[bear_idx])
            except Exception:
                p_r, p_u, p_d = 0.34, 0.33, 0.33
        else:
            p_r, p_u, p_d = 0.34, 0.33, 0.33

        p_ranges[i] = p_r
        p_bulls[i] = p_u
        p_bears[i] = p_d

        probs = {"RANGE": p_r, "BULL_TREND": p_u, "BEAR_PANIC": p_d}
        max_s = max(probs, key=probs.get)
        if probs[max_s] >= trans_threshold:
            curr_state = max_s
        states[i] = curr_state

    data["p_range"] = p_ranges
    data["p_bull"] = p_bulls
    data["p_bear"] = p_bears
    data["regime_state"] = states
    data["state_3hmm"] = states
    data["regime_trend_prob"] = p_bulls
    data["regime_mr_prob"] = p_ranges
    return data


def run_experiment_22():
    print("==================================================================================")
    print(f"[{time.strftime('%X')}] === [실험 22] 정석 주간 앵커링(주 1회 학습+매시간 추론) 요일별 전수 백테스트 ===")
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

    sim = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        maker_fee_pct=0.0002,
        taker_fee_pct=0.0005,
        slippage_pct=0.0002,
        funding_fee_pct=0.0001
    )

    results = []
    for w in weekdays:
        t0 = time.time()
        print(f"\n[{time.strftime('%X')}] >> 요일 앵커 테스트: {w['day_name']}...")
        df_proc = calculate_true_calendar_regimes(df_ind, anchor_dayofweek=w['dayofweek'], hmm_window=720, trans_threshold=0.45)
        res = sim.run(df_proc.iloc[720:].reset_index(drop=True))
        elapsed = time.time() - t0

        res['day_name'] = w['day_name']
        res['dayofweek'] = w['dayofweek']
        res['elapsed_sec'] = elapsed
        results.append(res)
        print(f"[{time.strftime('%X')}] [완료] ({elapsed:.1f}초): 수익률 {res['total_return_pct']:+.2f}%, MDD {res['mdd_pct']:.2f}%, 거래 {res['total_trades']}회, PF {res['profit_factor']:.2f}")

    print("\n\n" + "=" * 105)
    print(f"[{time.strftime('%X')}]    [실험 22] 정석 주간 앵커링(주 1회 피팅+매시간 추론) 7개 요일별 4개년 성과 종합표")
    print("=" * 105)
    print(f"{'앵커 요일 (00:00 UTC)':<30} | {'총수익률':<12} | {'MDD':<10} | {'거래횟수':<10} | {'승률':<10} | {'손익비(PF)':<12} | {'최종자산'}")
    print("-" * 105)

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
    run_experiment_22()
