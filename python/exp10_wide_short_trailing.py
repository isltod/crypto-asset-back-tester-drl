"""
[실험 10] 숏(SHORT) 트레일링 버퍼 확장 (3.0x vs 3.5x vs 4.0x vs 4.5x vs 5.0x) 독립 검증
- 가설: 하락장의 폭발적 변동성을 견디기 위해 숏 트레일링 버퍼를 3.5x~4.5x로 확대하면 성과가 개선되는가?
- 검증 기간: 2021.01 ~ 2024.06 (3.5년 장기)
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
import pandas as pd
from python.exp9_asymmetric_short import run_asymmetric_simulation
from python.utils.indicators import add_all_indicators
from python.regime.regime_manager import RegimeManager


def run_experiment_10():
    print("=== [실험 10] 숏(SHORT) 트레일링 버퍼 확장(3.0x ~ 5.0x) 독립 검증 시작 ===")

    cache_file = os.path.join("data", "BTCUSDT_1h_2021_2024.csv")
    df_raw = pd.read_csv(cache_file)
    df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)

    df_ind = add_all_indicators(df_raw)
    manager = RegimeManager(hmm_window=720, retrain_interval=168, hysteresis_upper=0.65, hysteresis_lower=0.35, cooldown_bars=3)
    df_proc = manager.calculate_regime_probabilities(df_ind)
    test_df = df_proc.dropna(subset=['regime_trend_prob']).reset_index(drop=True)

    configs = [
        {"name": "숏 3.0x (기준점)", "short_trail": 3.0},
        {"name": "숏 3.5x ATR", "short_trail": 3.5},
        {"name": "숏 4.0x ATR", "short_trail": 4.0},
        {"name": "숏 4.5x ATR", "short_trail": 4.5},
        {"name": "숏 5.0x ATR", "short_trail": 5.0},
    ]

    summary_rows = []
    equity_curves = {}

    for cfg in configs:
        res = run_asymmetric_simulation(test_df, cfg['short_trail'])
        df_t = res['trades_df']
        
        # [추세추종 - 숏] 성과
        tf_short = df_t[(df_t['engine'] == "TREND_FOLLOWING") & (df_t['side'] == "SHORT")]
        tf_short_pnl = tf_short['pnl'].sum() if not tf_short.empty else 0.0
        tf_short_wr = (len(tf_short[tf_short['pnl'] > 0]) / len(tf_short)) * 100.0 if not tf_short.empty else 0.0

        # 2022년 하락장 손익
        df_t['year'] = pd.to_datetime(df_t['exit_time']).dt.year
        pnl_2022 = df_t[df_t['year'] == 2022]['pnl'].sum() if not df_t.empty else 0.0

        summary_rows.append({
            "설정": cfg['name'],
            "3.5년 총수익률": f"{res['total_return_pct']:+.2f}%",
            "최종 자산 ($)": f"${res['final_equity']:,.2f}",
            "MDD (%)": f"{res['mdd_pct']:.2f}%",
            "전체 PF": f"{res['profit_factor']:.2f}",
            "추세 숏 PnL ($)": f"${tf_short_pnl:+,.2f}",
            "추세 숏 승률": f"{tf_short_wr:.1f}%",
            "2022년 PnL ($)": f"${pnl_2022:+,.2f}",
        })
        equity_curves[cfg['name']] = res['equity_curve']

    df_sum = pd.DataFrame(summary_rows)
    print("\n" + "=" * 95)
    print("          [ 실험 10: 숏(SHORT) 트레일링 버퍼 확장 성과 비교표 (3.5년 장기) ]          ")
    print("=" * 95)
    print(df_sum.to_string(index=False))
    print("=" * 95)

    # 차트 저장
    plt.figure(figsize=(12, 6))
    for label, eq in equity_curves.items():
        plt.plot(eq, linewidth=1.8, label=label)

    plt.axhline(10000.0, color='gray', linestyle='--', label='Initial ($10k)')
    plt.title("RADE Experiment 10: Wide Short Trailing Comparison (3.0x vs 3.5x vs 4.0x vs 4.5x vs 5.0x)", fontsize=12)
    plt.xlabel("Trade Progress (Trades)")
    plt.ylabel("Account Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    chart_path = os.path.join("data", "exp10_wide_short_trailing_plot.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Done] 숏 버퍼 확장 비교 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_experiment_10()
