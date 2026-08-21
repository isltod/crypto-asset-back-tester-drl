"""
[실험 1] 리스크 비율(Risk per Trade) 상향 독립 검증 스크립트
- 기준점: 1.0% (v0.3.0 Baseline)
- 비교군: 1.5%, 2.0%, 2.5%, 3.0%
- 목적: 리스크 상향 시 수익률 증가폭 및 MDD 변화를 정량적으로 비교 분석
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib.pyplot as plt
import pandas as pd
from rade.data_collector.binance_fetcher import BinanceFuturesFetcher
from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from rade.backtest.simulator import BacktestSimulator


def run_experiment_1():
    print("=== [실험 1] 리스크 비율(Risk per Trade) 독립 검증 시작 ===")

    # 1. 데이터 로드 및 지표/국면 계산
    fetcher = BinanceFuturesFetcher(data_dir="data")
    df_raw = fetcher.get_or_download_data(
        symbol="BTCUSDT",
        interval="1h",
        start_time_str="2023-01-01 00:00:00",
        end_time_str="2024-06-01 00:00:00"
    )

    df_indicators = add_all_indicators(df_raw)
    regime_manager = RegimeManager(
        hmm_window=720,
        retrain_interval=168,
        hysteresis_upper=0.65,
        hysteresis_lower=0.35,
        cooldown_bars=3
    )
    df_processed = regime_manager.calculate_regime_probabilities(df_indicators)
    test_df = df_processed.dropna(subset=['regime_trend_prob']).reset_index(drop=True)

    # 2. 리스크 비율별 시뮬레이션 비교
    risk_ratios = [0.010, 0.015, 0.020, 0.025, 0.030]
    comparison_results = []
    equity_curves = {}

    for risk in risk_ratios:
        sim = BacktestSimulator(
            initial_capital=10000.0,
            risk_per_trade_pct=risk,
            leverage=5.0 if risk > 0.02 else 3.0,
        )
        res = sim.run(test_df)
        
        calmar = res['total_return_pct'] / res['mdd_pct'] if res['mdd_pct'] > 0 else 0.0

        comparison_results.append({
            "Risk per Trade": f"{risk * 100:.1f}%",
            "Total Return (%)": f"{res['total_return_pct']:+.2f}%",
            "Final Equity ($)": f"${res['final_equity']:,.2f}",
            "MDD (%)": f"{res['mdd_pct']:.2f}%",
            "Profit Factor": f"{res['profit_factor']:.2f}",
            "Sharpe Ratio": f"{res['sharpe_ratio']:.2f}",
            "Return / MDD": f"{calmar:.2f}",
            "raw_return": res['total_return_pct'],
            "raw_mdd": res['mdd_pct'],
        })
        equity_curves[f"{risk*100:.1f}% Risk"] = res['equity_curve']

    # 3. 비교 표 출력
    df_comp = pd.DataFrame(comparison_results)
    print("\n" + "=" * 75)
    print("                [ 실험 1: 리스크 비율별 성과 비교표 ]                ")
    print("=" * 75)
    cols_to_print = ["Risk per Trade", "Total Return (%)", "Final Equity ($)", "MDD (%)", "Profit Factor", "Sharpe Ratio", "Return / MDD"]
    print(df_comp[cols_to_print].to_string(index=False))
    print("=" * 75)

    # 4. 차트 시각화 저장
    plt.figure(figsize=(12, 6))
    timestamps = test_df['datetime']
    for label, eq in equity_curves.items():
        plt.plot(eq, linewidth=1.8, label=label)

    plt.axhline(10000.0, color='gray', linestyle='--', label='Initial ($10k)')
    plt.title("RADE Experiment 1: Risk per Trade Comparison (1.0% vs 1.5% vs 2.0% vs 2.5% vs 3.0%)", fontsize=12)
    plt.xlabel("Bars (Hourly)")
    plt.ylabel("Account Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join("data", "exp1_risk_scaling_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n[Done] 실험 1 비교 차트 저장 완료: {plot_path}")


if __name__ == "__main__":
    run_experiment_1()
