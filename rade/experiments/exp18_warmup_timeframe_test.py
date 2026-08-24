"""
[실험 18] 2022년 1년 선행 웜업 기반 타임프레임 (1H vs 30m vs 15m vs 5m) 정밀 비교 검증 (B안)
- 선행 웜업 구간: 2022.01.01 ~ 2022.12.31 (HMM 상태 안정화용)
- 성과 평가 구간: 2023.01.01 ~ 2024.06.01 (1.5년 동일 구간)
- 모델: 3-State Gaussian HMM (Cash Mode) + 동적 4.0x ATR 트레일링 + 평균회귀
- 체결: 실전 Taker 0.05% + 슬리피지 0.02% + 펀딩비 보수적 체결
"""
import os
import sys
from typing import Dict, Any
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.data_collector.binance_fetcher import BinanceFuturesFetcher
from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager, RegimeState
from rade.backtest.simulator import BacktestSimulator


def run_warmup_timeframe_test(interval: str, hmm_window: int, retrain_interval: int) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"   [BTCUSDT - {interval}] 2022년 선행 웜업 + 2023~2024 평가")
    print(f"=======================================================")

    fetcher = BinanceFuturesFetcher(data_dir="data")
    cache_file = os.path.join("data", f"BTCUSDT_{interval}_2022_2024.csv")

    if os.path.exists(cache_file):
        print(f"[Cache] 로컬 캐시 로드: {cache_file}")
        df_raw = pd.read_csv(cache_file)
        df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
    else:
        print(f"[Download] {interval} 데이터 다운로드 중 (2022.01.01 ~ 2024.06.01)...")
        df_raw = fetcher.fetch_klines(
            symbol="BTCUSDT",
            interval=interval,
            start_time_str="2022-01-01 00:00:00",
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

    # 전체 구간(2022~2024)으로 시뮬레이터를 연속 실행하여 지표 왜곡 방지
    test_df = df_proc.iloc[hmm_window:].reset_index(drop=True)

    print(f"백테스트 시뮬레이터 연속 실행 중 (총 {len(test_df)}개 봉)...")
    sim = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        maker_fee_pct=0.0002,
        taker_fee_pct=0.0005,
        slippage_pct=0.0002,
        funding_fee_pct=0.0001
    )
    full_res = sim.run(test_df)
    trades_df = full_res['trades_df']

    # 2023년 1월 1일 이후 체결된 거래만 정확히 분리하여 1.5년 평가 성과 산출
    if not trades_df.empty:
        trades_df['dt'] = pd.to_datetime(trades_df['entry_time'])
        eval_trades = trades_df[trades_df['dt'] >= '2023-01-01'].copy()
        
        wins = eval_trades[eval_trades['pnl'] > 0]
        losses = eval_trades[eval_trades['pnl'] < 0]
        total_pnl = eval_trades['pnl'].sum()
        gp = wins['pnl'].sum()
        gl = abs(losses['pnl'].sum())
        pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
        wr = len(wins) / len(eval_trades) * 100.0 if len(eval_trades) > 0 else 0.0
        
        # 평가 구간 자산 곡선 산출
        eval_equity = 10000.0 + total_pnl
        ret_pct = (total_pnl / 10000.0) * 100.0
        
        # MDD 계산
        equity_curve = [10000.0]
        curr_eq = 10000.0
        for pnl in eval_trades['pnl']:
            curr_eq += pnl
            equity_curve.append(curr_eq)
        peaks = np.maximum.accumulate(equity_curve)
        dds = (peaks - equity_curve) / peaks * 100.0
        mdd = np.max(dds) if len(dds) > 0 else 0.0

        res = {
            'interval': interval,
            'total_return_pct': ret_pct,
            'mdd_pct': mdd,
            'total_trades': len(eval_trades),
            'win_rate_pct': wr,
            'profit_factor': pf,
            'final_equity': eval_equity,
            'total_candles': len(test_df[pd.to_datetime(test_df['datetime']) >= '2023-01-01']),
        }
    else:
        res = {
            'interval': interval,
            'total_return_pct': 0.0,
            'mdd_pct': 0.0,
            'total_trades': 0,
            'win_rate_pct': 0.0,
            'profit_factor': 0.0,
            'final_equity': 10000.0,
            'total_candles': 0,
        }

    return res


def run_experiment_18():
    print("=== [실험 18] 2022년 선행 웜업 기반 타임프레임 (1H vs 30m vs 15m vs 5m) 정밀 비교 시작 ===")

    configs = [
        {"interval": "1h",  "hmm_window": 720,  "retrain_interval": 168},   # 1h: 30일=720봉, 7일=168봉
        {"interval": "30m", "hmm_window": 1440, "retrain_interval": 336},   # 30m: 30일=1440봉, 7일=336봉
        {"interval": "15m", "hmm_window": 2880, "retrain_interval": 672},   # 15m: 30일=2880봉, 7일=672봉
        {"interval": "5m",  "hmm_window": 4320, "retrain_interval": 1008},  # 5m: 15일=4320봉, 3.5일=1008봉
    ]

    results = []
    for cfg in configs:
        res = run_warmup_timeframe_test(
            interval=cfg['interval'],
            hmm_window=cfg['hmm_window'],
            retrain_interval=cfg['retrain_interval']
        )
        results.append(res)

    print("\n\n" + "=" * 95)
    print("   [실험 18] 2022년 1년 선행 웜업 기반 2023~2024(1.5년) 타임프레임별 성과 종합 비교표")
    print("=" * 95)
    print(f"{'타임프레임':<10} | {'총수익률':<12} | {'MDD':<10} | {'거래횟수':<10} | {'승률':<10} | {'손익비(PF)':<12} | {'최종자산'}")
    print("-" * 95)

    for r in results:
        itv = r['interval']
        ret = f"{r['total_return_pct']:+.2f}%"
        mdd = f"{r['mdd_pct']:.2f}%"
        cnt = f"{r['total_trades']}회"
        wr = f"{r['win_rate_pct']:.2f}%"
        pf = f"{r['profit_factor']:.2f}"
        eq = f"${r['final_equity']:,.2f}"
        print(f"{itv:<10} | {ret:<12} | {mdd:<10} | {cnt:<10} | {wr:<10} | {pf:<12} | {eq}")

    print("=" * 95)


if __name__ == "__main__":
    run_experiment_18()
