"""
[실험 6] 리스크 1.0% 기준 타임프레임(1h vs 30m vs 15m vs 5m) 정밀 비교 검증
- 비교 타임프레임: 1h, 30m, 15m, 5m
- 고정 리스크: 1.0% Risk per Trade
- 목적: 타임프레임 축소에 따른 거래 횟수 폭증, 수수료 누적, 손실 증가 추세를 완벽히 정량화
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


def run_timeframe_sim(interval: str, hmm_window: int, risk_pct: float = 0.01) -> Dict[str, Any]:
    print(f"\n[BTCUSDT - {interval} (Risk {risk_pct*100:.0f}%)] 데이터 로드 및 분석 중...")
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
        risk_per_trade_pct=risk_pct,
        leverage=3.0
    )
    res = sim.run(test_df)
    res['interval'] = interval
    return res


def run_experiment_6():
    print("=== [실험 6] 리스크 1.0% 기준 타임프레임(1h vs 30m vs 15m vs 5m) 비교 검증 ===")

    # 타임프레임별 30일 환산 HMM 윈도우 봉 수
    intervals = [
        {"interval": "1h", "hmm_window": 720},     # 30일 = 720봉
        {"interval": "30m", "hmm_window": 1440},   # 30일 = 1440봉
        {"interval": "15m", "hmm_window": 2880},   # 30일 = 2880봉
        {"interval": "5m", "hmm_window": 8640},    # 30일 = 8640봉
    ]

    results_1pct = []
    equity_curves = {}

    for item in intervals:
        inter = item['interval']
        hmm_win = item['hmm_window']
        res = run_timeframe_sim(inter, hmm_win, risk_pct=0.01)

        results_1pct.append({
            "타임프레임": inter,
            "총 수익률": f"{res['total_return_pct']:+.2f}%",
            "최종 자산": f"${res['final_equity']:,.2f}",
            "MDD": f"{res['mdd_pct']:.2f}%",
            "총 거래수": f"{res['total_trades']}회",
            "승률": f"{res['win_rate_pct']:.1f}%",
            "Profit Factor": f"{res['profit_factor']:.2f}",
            "Sharpe Ratio": f"{res['sharpe_ratio']:.2f}",
        })
        equity_curves[f"BTC {inter} (1% Risk)"] = res['equity_curve']

    # 1. 비교 표 출력
    df_res = pd.DataFrame(results_1pct)
    print("\n" + "=" * 85)
    print("        [ 실험 6: 리스크 1.0% 기준 타임프레임별 (1h vs 30m vs 15m vs 5m) 성과표 ]        ")
    print("=" * 85)
    print(df_res.to_string(index=False))
    print("=" * 85)

    # 2. 차트 시각화 저장
    plt.figure(figsize=(12, 6))
    for label, eq in equity_curves.items():
        plt.plot(eq, linewidth=1.8, label=label)

    plt.axhline(10000.0, color='gray', linestyle='--', label='Initial ($10k)')
    plt.title("RADE Experiment 6: 1.0% Risk across Timeframes (1h vs 30m vs 15m vs 5m)", fontsize=12)
    plt.xlabel("Trade Progress (Trades)")
    plt.ylabel("Account Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    chart_path = os.path.join("data", "exp6_timeframe_risk1pct_plot.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Done] 타임프레임 1% 리스크 비교 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_experiment_6()
