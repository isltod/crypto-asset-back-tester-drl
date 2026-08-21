"""
[실험 5] 레버리지(Leverage) 변경 독립 검증 스크립트
- 기준점: 3x Leverage (v0.4.0 Baseline, 2.0% Risk)
- 비교군: 1x, 2x, 3x, 5x, 10x
- 목적: 레버리지 설정에 따른 증거금 제약 해소 효과 및 계좌 성과/안전성 독립 검증
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any
from rade.data_collector.binance_fetcher import BinanceFuturesFetcher
from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from rade.backtest.simulator import BacktestSimulator


def run_experiment_5():
    print("=== [실험 5] 레버리지(1x vs 2x vs 3x vs 5x vs 10x) 독립 검증 시작 ===")

    # 1. 데이터 로드 및 국면 분석
    fetcher = BinanceFuturesFetcher(data_dir="data")
    df_raw = fetcher.get_or_download_data(
        symbol="BTCUSDT",
        interval="1h",
        start_time_str="2023-01-01 00:00:00",
        end_time_str="2024-06-01 00:00:00"
    )

    df_ind = add_all_indicators(df_raw)
    manager = RegimeManager(
        hmm_window=720,
        retrain_interval=168,
        hysteresis_upper=0.65,
        hysteresis_lower=0.35,
        cooldown_bars=3
    )
    df_proc = manager.calculate_regime_probabilities(df_ind)
    test_df = df_proc.dropna(subset=['regime_trend_prob']).reset_index(drop=True)

    # 2. 레버리지별 시뮬레이션
    leverage_levels = [1.0, 2.0, 3.0, 5.0, 10.0]
    results = []
    equity_curves = {}

    for lev in leverage_levels:
        sim = BacktestSimulator(
            initial_capital=10000.0,
            risk_per_trade_pct=0.02, # 2.0% Risk
            leverage=lev,
        )
        res = sim.run(test_df)

        results.append({
            "레버리지": f"{int(lev)}x",
            "총 수익률": f"{res['total_return_pct']:+.2f}%",
            "최종 자산": f"${res['final_equity']:,.2f}",
            "MDD": f"{res['mdd_pct']:.2f}%",
            "총 거래수": f"{res['total_trades']}회",
            "승률": f"{res['win_rate_pct']:.1f}%",
            "Profit Factor": f"{res['profit_factor']:.2f}",
            "Sharpe Ratio": f"{res['sharpe_ratio']:.2f}",
        })
        equity_curves[f"{int(lev)}x Leverage"] = res['equity_curve']

    # 3. 비교 표 출력
    df_res = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("                [ 실험 5: 레버리지별 (1x vs 2x vs 3x vs 5x vs 10x) 성과 비교표 ]                ")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print("=" * 80)

    # 4. 차트 시각화 저장
    plt.figure(figsize=(12, 6))
    for label, eq in equity_curves.items():
        plt.plot(eq, linewidth=1.8, label=label)

    plt.axhline(10000.0, color='gray', linestyle='--', label='Initial ($10k)')
    plt.title("RADE Experiment 5: Leverage Comparison (1x vs 2x vs 3x vs 5x vs 10x)", fontsize=12)
    plt.xlabel("Bars (Hourly)")
    plt.ylabel("Account Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    chart_path = os.path.join("data", "exp5_leverage_plot.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Done] 레버리지 비교 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_experiment_5()
