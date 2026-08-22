"""
[실험 17] 신규 3-State HMM 코어 기반 단기 타임프레임 (5m vs 15m vs 30m) 비교 검증
- 모델: 3-State Gaussian HMM (Range, Bull Trend, Bear Panic Cash Mode)
- 엔진: 추세추종 롱 (동적 4.0x ATR 트레일링) + 평균회귀 (80:20 분할익절)
- 체결: 실전 Taker(0.05%) + 슬리피지(0.02%) + 펀딩비 반영
- 비교 대상: 5분봉(5m) vs 15분봉(15m) vs 30분봉(30m)
- 검증 구간: 2023.01.01 ~ 2024.06.01 (1.5년)
"""
import os
import sys
from typing import Dict, Any
import pandas as pd
import numpy as np

# 프로젝트 루트 디렉토리 추가
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.data_collector.binance_fetcher import BinanceFuturesFetcher
from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager, RegimeState
from rade.backtest.simulator import BacktestSimulator


def run_timeframe_test(interval: str, hmm_window: int, retrain_interval: int) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"   [BTCUSDT - {interval}] 데이터 로드 및 3-State HMM 백테스트")
    print(f"=======================================================")
    
    fetcher = BinanceFuturesFetcher(data_dir="data")
    cache_file = os.path.join("data", f"BTCUSDT_{interval}.csv")
    
    if os.path.exists(cache_file):
        print(f"[Cache] 로컬 캐시 로드: {cache_file}")
        df_raw = pd.read_csv(cache_file)
        df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
    else:
        print(f"[Download] 바이낸스 선물 {interval} 데이터 다운로드 중 (2023.01.01 ~ 2024.06.01)...")
        df_raw = fetcher.fetch_klines(
            symbol="BTCUSDT",
            interval=interval,
            start_time_str="2023-01-01 00:00:00",
            end_time_str="2024-06-01 00:00:00"
        )
        if not df_raw.empty:
            df_raw.to_csv(cache_file, index=False)
            print(f"[Saved] 로컬 저장 완료: {cache_file}")

    print(f"총 {len(df_raw)}개 캔들 수집 완료. 기술적 지표 계산 중...")
    df_ind = add_all_indicators(df_raw)

    print(f"3-State HMM 국면 분석 중 (Window: {hmm_window}봉, 재학습: {retrain_interval}봉)...")
    manager = RegimeManager(
        hmm_window=hmm_window,
        retrain_interval=retrain_interval,
        trans_threshold=0.45,
        cooldown_bars=3
    )
    df_proc = manager.calculate_regime_probabilities(df_ind)
    test_df = df_proc.iloc[hmm_window:].reset_index(drop=True)

    print(f"백테스트 시뮬레이터 실행 중 (총 {len(test_df)}개 봉)...")
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
    res['interval'] = interval
    res['total_candles'] = len(test_df)
    return res


def run_experiment_17():
    print("=== [실험 17] 신규 3-State HMM 코어 기반 단기 타임프레임 (5m vs 15m vs 30m) 비교 검증 ===")

    # 타임프레임별 약 15일~30일 환산 슬라이딩 윈도우 설정
    configs = [
        {"interval": "30m", "hmm_window": 720, "retrain_interval": 168},   # 30m: 720봉 = 15일, 168봉 = 3.5일
        {"interval": "15m", "hmm_window": 1440, "retrain_interval": 336},  # 15m: 1440봉 = 15일, 336봉 = 3.5일
        {"interval": "5m",  "hmm_window": 2880, "retrain_interval": 576},  # 5m: 2880봉 = 10일, 576봉 = 2일
    ]

    results = []
    for cfg in configs:
        res = run_timeframe_test(
            interval=cfg['interval'],
            hmm_window=cfg['hmm_window'],
            retrain_interval=cfg['retrain_interval']
        )
        results.append(res)

    print("\n\n" + "=" * 85)
    print("      [실험 17] 3-State HMM 코어 기반 타임프레임별 성과 종합 비교표")
    print("=" * 85)
    print(f"{'타임프레임':<10} | {'총수익률':<12} | {'MDD':<10} | {'거래횟수':<10} | {'승률':<10} | {'손익비(PF)':<12} | {'최종자산'}")
    print("-" * 85)

    for r in results:
        itv = r['interval']
        ret = f"{r['total_return_pct']:+.2f}%"
        mdd = f"{r['mdd_pct']:.2f}%"
        cnt = f"{r['total_trades']}회"
        wr = f"{r['win_rate_pct']:.2f}%"
        pf = f"{r['profit_factor']:.2f}"
        eq = f"${r['final_equity']:,.2f}"
        print(f"{itv:<10} | {ret:<12} | {mdd:<10} | {cnt:<10} | {wr:<10} | {pf:<12} | {eq}")

    print("=" * 85)


if __name__ == "__main__":
    run_experiment_17()
