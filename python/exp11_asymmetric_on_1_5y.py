"""
[실험 11] 비대칭 숏 4.5x 전략을 '무난한 1.5년(2023.01 ~ 2024.06)' 구간에 적용하여 대칭 3.0x와 1:1 비교
- 목적: 숏 4.5x 버퍼가 대폭락장(2022)뿐만 아니라 평범한 횡보/상승장(2023~2024)에서도 대칭 3.0x보다 우수한지 정밀 검증
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
import pandas as pd
from python.data.binance_fetcher import BinanceFuturesFetcher
from python.utils.indicators import add_all_indicators
from python.regime.regime_manager import RegimeManager
from python.exp9_asymmetric_short import run_asymmetric_simulation


def run_experiment_11():
    print("=== [실험 11] 무난한 1.5년(2023.01 ~ 2024.06) 구간에서 대칭 3.0x vs 비대칭 4.5x 1:1 검증 시작 ===")

    # 1. 1.5년치 데이터 로드
    fetcher = BinanceFuturesFetcher(data_dir="data")
    df_raw = fetcher.get_or_download_data(
        symbol="BTCUSDT",
        interval="1h",
        start_time_str="2023-01-01 00:00:00",
        end_time_str="2024-06-01 00:00:00"
    )

    df_ind = add_all_indicators(df_raw)
    manager = RegimeManager(hmm_window=720, retrain_interval=168, hysteresis_upper=0.65, hysteresis_lower=0.35, cooldown_bars=3)
    df_proc = manager.calculate_regime_probabilities(df_ind)
    test_df = df_proc.dropna(subset=['regime_trend_prob']).reset_index(drop=True)

    # 2. 1.5년 구간에서 3.0x vs 4.5x 비교 시뮬레이션
    configs = [
        {"name": "대칭 3.0x (롱3.0x / 숏3.0x)", "short_trail": 3.0},
        {"name": "비대칭 4.5x (롱3.0x / 숏4.5x)", "short_trail": 4.5},
    ]

    summary_rows = []
    equity_curves = {}

    for cfg in configs:
        res = run_asymmetric_simulation(test_df, cfg['short_trail'])
        df_t = res['trades_df']

        l_sub = df_t[df_t['side'] == "LONG"]
        s_sub = df_t[df_t['side'] == "SHORT"]

        l_pnl = l_sub['pnl'].sum() if not l_sub.empty else 0.0
        s_pnl = s_sub['pnl'].sum() if not s_sub.empty else 0.0

        summary_rows.append({
            "전략": cfg['name'],
            "1.5년 총수익률": f"{res['total_return_pct']:+.2f}%",
            "최종 자산 ($)": f"${res['final_equity']:,.2f}",
            "MDD (%)": f"{res['mdd_pct']:.2f}%",
            "전체 PF": f"{res['profit_factor']:.2f}",
            "LONG PnL ($)": f"${l_pnl:+,.2f}",
            "SHORT PnL ($)": f"${s_pnl:+,.2f}",
            "총 거래수": f"{len(df_t)}회",
        })
        equity_curves[cfg['name']] = res['equity_curve']

    df_sum = pd.DataFrame(summary_rows)
    print("\n" + "=" * 90)
    print("         [ 실험 11: 1.5년(2023~2024) 구간 대칭 3.0x vs 비대칭 4.5x 1:1 비교표 ]         ")
    print("=" * 90)
    print(df_sum.to_string(index=False))
    print("=" * 90)

    # 차트 저장
    plt.figure(figsize=(12, 6))
    for label, eq in equity_curves.items():
        plt.plot(eq, linewidth=1.8, label=label)

    plt.axhline(10000.0, color='gray', linestyle='--', label='Initial ($10k)')
    plt.title("RADE Experiment 11: 1.5Y Normal Period (Symmetric 3.0x vs Asymmetric Short 4.5x)", fontsize=12)
    plt.xlabel("Trade Progress (Trades)")
    plt.ylabel("Account Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    chart_path = os.path.join("data", "exp11_asymmetric_on_1_5y_plot.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Done] 1.5년 비교 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_experiment_11()
