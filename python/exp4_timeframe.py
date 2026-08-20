"""
[실험 4] 타임프레임 단축 (1H vs 30m vs 15m) 독립 검증 스크립트
- 기준점: 1h (1시간봉, v0.4.0 Baseline)
- 비교군: 30m (30분봉), 15m (15분봉)
- 목적: 타임프레임 단축 시 거래 기회 증가 vs 수수료/노이즈 영향을 정량적으로 독립 비교
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any
from python.data.binance_fetcher import BinanceFuturesFetcher
from python.utils.indicators import add_all_indicators
from python.regime.regime_manager import RegimeManager
from python.backtest.simulator import BacktestSimulator


def run_timeframe_test(interval: str, hmm_window: int) -> Dict[str, Any]:
    print(f"\n[BTCUSDT - {interval}] 데이터 로드 및 RADE 국면 분석 중...")
    fetcher = BinanceFuturesFetcher(data_dir="data")
    df_raw = fetcher.get_or_download_data(
        symbol="BTCUSDT",
        interval=interval,
        start_time_str="2023-01-01 00:00:00",
        end_time_str="2024-06-01 00:00:00"
    )

    df_ind = add_all_indicators(df_raw)
    manager = RegimeManager(
        hmm_window=hmm_window,
        retrain_interval=int(hmm_window / 4),
        hysteresis_upper=0.65,
        hysteresis_lower=0.35,
        cooldown_bars=3
    )
    df_proc = manager.calculate_regime_probabilities(df_ind)
    test_df = df_proc.dropna(subset=['regime_trend_prob']).reset_index(drop=True)

    sim = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.02, # 2.0% Risk
        leverage=3.0
    )
    res = sim.run(test_df)
    res['interval'] = interval
    return res


def run_experiment_4():
    print("=== [실험 4] 타임프레임 단축 (1H vs 30m vs 15m) 독립 검증 시작 ===")

    # 타임프레임별 30일 환산 HMM 윈도우 봉 수
    intervals = [
        {"interval": "1h", "hmm_window": 720},    # 30일 = 720시간
        {"interval": "30m", "hmm_window": 1440},  # 30일 = 1440봉
        {"interval": "15m", "hmm_window": 2880},  # 30일 = 2880봉
    ]

    results = []
    equity_curves = {}

    for item in intervals:
        inter = item['interval']
        hmm_win = item['hmm_window']
        res = run_timeframe_test(inter, hmm_win)

        results.append({
            "타임프레임": inter,
            "총 수익률": f"{res['total_return_pct']:+.2f}%",
            "최종 자산": f"${res['final_equity']:,.2f}",
            "MDD": f"{res['mdd_pct']:.2f}%",
            "총 거래수": f"{res['total_trades']}회",
            "승률": f"{res['win_rate_pct']:.1f}%",
            "Profit Factor": f"{res['profit_factor']:.2f}",
            "Sharpe Ratio": f"{res['sharpe_ratio']:.2f}",
        })
        equity_curves[f"BTC {inter}"] = res['equity_curve']

    # 1. 비교 표 출력
    df_res = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("              [ 실험 4: 타임프레임별 (1H vs 30m vs 15m) 성과 비교표 ]              ")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print("=" * 80)

    # 2. 차트 시각화 저장
    plt.figure(figsize=(12, 6))
    for label, eq in equity_curves.items():
        plt.plot(eq, linewidth=1.8, label=label)

    plt.axhline(10000.0, color='gray', linestyle='--', label='Initial ($10k)')
    plt.title("RADE Experiment 4: Timeframe Comparison (1H vs 30m vs 15m)", fontsize=12)
    plt.xlabel("Trade Progress (Trades)")
    plt.ylabel("Account Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    chart_path = os.path.join("data", "exp4_timeframe_plot.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Done] 타임프레임 비교 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_experiment_4()
