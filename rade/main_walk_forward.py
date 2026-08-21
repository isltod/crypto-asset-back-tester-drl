"""
RADE Walk-Forward Optimization 실행 및 Out-of-Sample 성과 분석 스크립트
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rade.data.binance_fetcher import BinanceFuturesFetcher
from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from rade.backtest.walk_forward import WalkForwardOptimizer


def main():
    print("=== [Day 3-5] RADE Walk-Forward Optimization 가동 ===")

    # 1. 데이터 로드 및 전처리
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

    print(f"전체 분석 가능 구간: 총 {len(test_df)}개 캔들 ({test_df['datetime'].iloc[0]} ~ {test_df['datetime'].iloc[-1]})")

    # 2. Walk-Forward 최적화기 실행 (6개월 학습: 4320봉, 1개월 검증: 720봉)
    optimizer = WalkForwardOptimizer(
        train_window_bars=4320,  # 6개월
        test_window_bars=720,    # 1개월
        step_bars=720            # 1개월씩 전진
    )

    wf_results = optimizer.run_optimization(test_df)
    reports = wf_results['window_reports']
    df_oos_trades = wf_results['all_oos_trades']

    # 3. 윈도우별 성과 표 출력
    print("\n" + "=" * 65)
    print("          [ WALK-FORWARD OUT-OF-SAMPLE REPORTS ]          ")
    print("=" * 65)
    for r in reports:
        print(f" * Window {r['window']} [{r['test_period']}]: OOS Return {r['oos_return_pct']:+.2f}% | MDD {r['oos_mdd_pct']:.2f}% | Trades {r['oos_trades']}회 (WR: {r['oos_win_rate']:.1f}%)")
    print("=" * 65)

    # 4. 전체 OOS 누적 자산 곡선 산출
    if not df_oos_trades.empty:
        initial_capital = 10000.0
        equity = initial_capital
        eq_curve = [equity]

        for pnl in df_oos_trades['pnl']:
            equity += pnl
            eq_curve.append(equity)

        final_equity = eq_curve[-1]
        total_ret_pct = ((final_equity - initial_capital) / initial_capital) * 100.0

        peak = np.maximum.accumulate(eq_curve)
        drawdowns = ((np.array(eq_curve) - peak) / peak) * 100.0
        mdd_pct = abs(drawdowns.min())

        wins = df_oos_trades[df_oos_trades['pnl'] > 0]
        losses = df_oos_trades[df_oos_trades['pnl'] < 0]
        wr = (len(wins) / len(df_oos_trades)) * 100.0
        pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else 0.0

        print(f"\n[최종 Out-of-Sample 종합 성과 (과적합 제거 실전 지표)]")
        print(f" * 초기 자본: ${initial_capital:,.2f} -> 최종 자산: ${final_equity:,.2f}")
        print(f" * OOS 총 수익률: {total_ret_pct:+.2f}%")
        print(f" * OOS 최대 낙폭 (MDD): {mdd_pct:.2f}%")
        print(f" * OOS 총 거래: {len(df_oos_trades)}회 | 승률: {wr:.1f}% | Profit Factor: {pf:.2f}")

        # OOS 차트 저장
        plt.figure(figsize=(12, 6))
        plt.plot(eq_curve, color='darkgreen', linewidth=2.0, label='Walk-Forward OOS Equity Curve ($)')
        plt.axhline(initial_capital, color='gray', linestyle='--', label='Initial Capital ($10,000)')
        plt.title(f"RADE Walk-Forward Out-of-Sample Performance (Return: {total_ret_pct:+.2f}%, MDD: {mdd_pct:.2f}%)")
        plt.xlabel("OOS Trade Index")
        plt.ylabel("Equity ($)")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

        chart_path = os.path.join("data", "walk_forward_result.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"[Done] Walk-Forward 결과 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    main()
