"""
[실험 9] 숏(SHORT) 비대칭 트레일링 스탑 독립 검증 스크립트
- 문제점: 하락장은 급락 후 급반등(숏스퀴즈)이 잦아 대칭 3.0*ATR 트레일링 시 수익 반납
- 해결책: 롱(3.0*ATR 유지) vs 숏(1.5*ATR, 1.8*ATR, 2.0*ATR로 타이트화) 비대칭 검증
- 평가: 3.5년 전체 수익률 및 [추세추종 - 숏], 2022년 하락장 성과 변화 정밀 비교
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any
from python.utils.indicators import add_all_indicators
from python.regime.regime_manager import RegimeManager
from python.engines.trend_following import TrendFollowingEngine
from python.backtest.simulator import BacktestSimulator


def run_asymmetric_simulation(test_df: pd.DataFrame, short_trailing_atr: float, long_trailing_atr: float = 3.0) -> Dict[str, Any]:
    """표준 BacktestSimulator를 활용한 비대칭 트레일링 스탑 시뮬레이션 래퍼"""
    trend_eng = TrendFollowingEngine(
        long_trailing_atr=long_trailing_atr,
        short_trailing_atr=short_trailing_atr
    )
    sim = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        trend_engine=trend_eng
    )
    return sim.run(test_df)


def run_experiment_9():
    print("=== [실험 9] 숏(SHORT) 비대칭 트레일링 스탑 독립 검증 시작 (보수적 체결 모델 적용) ===")

    cache_file = os.path.join("data", "BTCUSDT_1h_2021_2024.csv")
    df_raw = pd.read_csv(cache_file)
    df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)

    df_ind = add_all_indicators(df_raw)
    manager = RegimeManager(hmm_window=720, retrain_interval=168, hysteresis_upper=0.65, hysteresis_lower=0.35, cooldown_bars=3)
    df_proc = manager.calculate_regime_probabilities(df_ind)
    test_df = df_proc.dropna(subset=['regime_trend_prob']).reset_index(drop=True)

    configs = [
        {"name": "대칭 3.0x (Baseline)", "short_trail": 3.0},
        {"name": "숏 2.0x ATR", "short_trail": 2.0},
        {"name": "숏 1.8x ATR", "short_trail": 1.8},
        {"name": "숏 1.5x ATR", "short_trail": 1.5},
    ]

    summary_rows = []
    equity_curves = {}

    for cfg in configs:
        res = run_asymmetric_simulation(test_df, cfg['short_trail'])
        df_t = res['trades_df']
        
        # [추세추종 - 숏]만 발췌
        if not df_t.empty:
            tf_short = df_t[(df_t['engine'] == "TREND_FOLLOWING") & (df_t['side'] == "SHORT")]
            tf_short_pnl = tf_short['pnl'].sum() if not tf_short.empty else 0.0
            tf_short_wr = (len(tf_short[tf_short['pnl'] > 0]) / len(tf_short)) * 100.0 if not tf_short.empty else 0.0

            # 2022년 하락장 손익
            df_t['year'] = pd.to_datetime(df_t['exit_time']).dt.year
            pnl_2022 = df_t[df_t['year'] == 2022]['pnl'].sum() if not df_t.empty else 0.0
        else:
            tf_short_pnl, tf_short_wr, pnl_2022 = 0.0, 0.0, 0.0

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
    print("\n" + "=" * 90)
    print("           [ 실험 9: 숏(SHORT) 비대칭 트레일링 스탑 성과 비교표 (3.5년 장기) ]           ")
    print("=" * 90)
    print(df_sum.to_string(index=False))
    print("=" * 90)

    # 차트 저장
    plt.figure(figsize=(12, 6))
    for label, eq in equity_curves.items():
        plt.plot(eq, linewidth=1.8, label=label)

    plt.axhline(10000.0, color='gray', linestyle='--', label='Initial ($10k)')
    plt.title("RADE Experiment 9: Asymmetric Short Trailing Stop Comparison (Conservative Fill)", fontsize=12)
    plt.xlabel("Timeline (Hours)")
    plt.ylabel("Account Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    chart_path = os.path.join("data", "exp9_asymmetric_short_plot.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Done] 숏 비대칭 비교 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_experiment_9()
