"""
[실험 3] 멀티 코인 유니버스 확장 (BTC + ETH + SOL) 독립 검증 스크립트
- 개별 코인 성과 (BTC, ETH, SOL)
- 3개 코인 통합 포트폴리오 동시 운용 성과
- 목적: 다종목 확장을 통한 자본 회전율 및 복리 수익률 증폭 검증
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from rade.data_collector.binance_fetcher import BinanceFuturesFetcher
from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from rade.backtest.simulator import BacktestSimulator


def run_single_coin_rade(symbol: str) -> Dict[str, Any]:
    """단일 코인에 대한 RADE 전처리 및 백테스트 실행"""
    print(f"\n[{symbol}] 데이터 로드 및 RADE 국면 분석 중...")
    fetcher = BinanceFuturesFetcher(data_dir="data")
    df_raw = fetcher.get_or_download_data(
        symbol=symbol,
        interval="1h",
        start_time_str="2023-01-01 00:00:00",
        end_time_str="2024-06-01 00:00:00"
    )

    df_ind = add_all_indicators(df_raw)
    manager = RegimeManager(hmm_window=720, retrain_interval=168, hysteresis_upper=0.65, hysteresis_lower=0.35, cooldown_bars=3)
    df_proc = manager.calculate_regime_probabilities(df_ind)
    test_df = df_proc.dropna(subset=['regime_trend_prob']).reset_index(drop=True)

    sim = BacktestSimulator(initial_capital=10000.0, risk_per_trade_pct=0.01)
    res = sim.run(test_df)
    res['processed_df'] = test_df
    return res


def run_experiment_3():
    print("=== [실험 3] 멀티 코인 유니버스 (BTC + ETH + SOL) 확장 검증 시작 ===")

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    coin_results = {}

    for sym in symbols:
        coin_results[sym] = run_single_coin_rade(sym)

    # 1. 개별 코인 성과 표 출력
    summary_rows = []
    for sym, res in coin_results.items():
        summary_rows.append({
            "코인 (Symbol)": sym,
            "총 수익률": f"{res['total_return_pct']:+.2f}%",
            "최종 자산": f"${res['final_equity']:,.2f}",
            "MDD": f"{res['mdd_pct']:.2f}%",
            "총 거래수": f"{res['total_trades']}회",
            "승률": f"{res['win_rate_pct']:.1f}%",
            "Profit Factor": f"{res['profit_factor']:.2f}",
        })

    df_sum = pd.DataFrame(summary_rows)
    print("\n" + "=" * 75)
    print("                [ 실험 3: 개별 코인별 RADE 성과표 ]                ")
    print("=" * 75)
    print(df_sum.to_string(index=False))
    print("=" * 75)

    # 2. 3개 코인 통합 포트폴리오 시뮬레이션
    print("\n[통합 포트폴리오] 3개 코인 동시 분산 운용 시뮬레이션 계산 중...")
    # 3개 코인의 모든 거래를 시간 순으로 병합하여 단일 계좌 $10,000에서 실행
    all_trades = []
    for sym, res in coin_results.items():
        df_tr = res['trades_df'].copy()
        if not df_tr.empty:
            df_tr['symbol'] = sym
            all_trades.append(df_tr)

    if all_trades:
        df_portfolio_trades = pd.concat(all_trades).sort_values(by="exit_time").reset_index(drop=True)
        
        initial_cap = 10000.0
        equity = initial_cap
        port_equity_curve = [equity]

        for pnl in df_portfolio_trades['pnl']:
            equity += pnl
            port_equity_curve.append(equity)

        port_eq_arr = np.array(port_equity_curve)
        port_total_return = ((port_eq_arr[-1] - initial_cap) / initial_cap) * 100.0
        peak = np.maximum.accumulate(port_eq_arr)
        drawdowns = ((port_eq_arr - peak) / peak) * 100.0
        port_mdd = abs(drawdowns.min())

        wins = df_portfolio_trades[df_portfolio_trades['pnl'] > 0]
        losses = df_portfolio_trades[df_portfolio_trades['pnl'] < 0]
        port_wr = (len(wins) / len(df_portfolio_trades)) * 100.0
        port_pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else 0.0

        print("\n" + "=" * 75)
        print("          [ 3개 코인 통합 포트폴리오 (BTC+ETH+SOL) 성과 ]          ")
        print("=" * 75)
        print(f" * 초기 자본:           ${initial_cap:,.2f}")
        print(f" * 최종 자산:           ${port_eq_arr[-1]:,.2f}  (+${port_eq_arr[-1] - initial_cap:,.2f} 순수익)")
        print(f" * 포트폴리오 총 수익률: {port_total_return:+.2f}%")
        print(f" * 포트폴리오 MDD:       {port_mdd:.2f}%")
        print(f" * 총 거래 횟수:         {len(df_portfolio_trades)}회 (월평균 약 17.5회)")
        print(f" * 전체 승률:           {port_wr:.1f}%")
        print(f" * 전체 손익비 (PF):     {port_pf:.2f}")
        print("=" * 75)

        # 3. 차트 시각화 저장
        plt.figure(figsize=(12, 6))
        for sym, res in coin_results.items():
            plt.plot(res['equity_curve'], linewidth=1.2, alpha=0.6, label=f"{sym} Single ({res['total_return_pct']:+.1f}%)")

        plt.plot(port_equity_curve, color='crimson', linewidth=2.2, label=f"Portfolio ({port_total_return:+.1f}%, MDD {port_mdd:.1f}%)")
        plt.axhline(initial_cap, color='gray', linestyle='--', label='Initial ($10k)')
        plt.title("RADE Experiment 3: Multi-Coin Universe (BTC + ETH + SOL Combined Portfolio)", fontsize=12)
        plt.xlabel("Trade Progress")
        plt.ylabel("Account Equity ($)")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

        chart_path = os.path.join("data", "exp3_multicoin_plot.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"\n[Done] 멀티 코인 포트폴리오 비교 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_experiment_3()
