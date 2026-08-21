"""
RADE 전체 백테스트 실행 및 성과 분석 메인 스크립트
1. 과거 데이터 및 국면 분류 데이터 준비
2. 듀얼 엔진 선물 백테스트 시뮬레이션
3. 성과 리포트 출력 및 자산 곡선/MDD 차트 시각화 저장
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from python.data.binance_fetcher import BinanceFuturesFetcher
from python.utils.indicators import add_all_indicators
from python.regime.regime_manager import RegimeManager
from python.backtest.simulator import BacktestSimulator


def run_full_backtest():
    print("=== [Phase 2] RADE 시스템 선물 백테스트 시작 ===")

    # 1. 데이터 로드 (2021년~현재 4년 풀데이터 로드)
    fetcher = BinanceFuturesFetcher(data_dir="data")
    cache_file = os.path.join("data", "BTCUSDT_1h_2021_2024.csv")
    if os.path.exists(cache_file):
        df_raw = pd.read_csv(cache_file)
        df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
    else:
        df_raw = fetcher.get_or_download_data(
            symbol="BTCUSDT",
            interval="1h",
            start_time_str="2021-01-01 00:00:00",
            end_time_str="2024-12-31 23:59:59"
        )

    if df_raw.empty:
        print("[Error] 데이터 로드 실패")
        return

    # 2. 지표 및 3-State 국면 계산
    print("기술적 지표 계산 중...")
    df_indicators = add_all_indicators(df_raw)

    print("3-State HMM 국면 분석 (Range, Bull, Bear/Panic Cash Mode) 계산 중...")
    regime_manager = RegimeManager(
        hmm_window=720,
        retrain_interval=168,
        trans_threshold=0.45,
        cooldown_bars=3
    )
    df_processed = regime_manager.calculate_regime_probabilities(df_indicators)

    # 초기 HMM 윈도우(720봉) 이후부터 유효 백테스트 실행
    test_df = df_processed.iloc[720:].reset_index(drop=True)
    print(f"백테스트 구간: 총 {len(test_df)}개 캔들 ({test_df['datetime'].iloc[0]} ~ {test_df['datetime'].iloc[-1]})")

    # 3. 백테스트 시뮬레이터 실행
    print("\n선물 백테스트 시뮬레이션 가동...")
    simulator = BacktestSimulator(
        initial_capital=10000.0,      # 시작 자금 $10,000
        risk_per_trade_pct=0.02,      # 1회 2.0% 리스크 (검증된 최적 밸런스 세팅)
        leverage=3.0,                 # 레버리지 3x
    )

    results = simulator.run(test_df)

    # 4. 성과 리포트 출력
    print("\n" + "=" * 55)
    print("          [ RADE SYSTEM FUTURES BACKTEST REPORT ]          ")
    print("=" * 55)
    print(f" * 초기 자본 (Initial Capital):  ${results['initial_capital']:,.2f}")
    print(f" * 최종 자산 (Final Equity):     ${results['final_equity']:,.2f}")
    print(f" * 총 수익률 (Total Return):     {results['total_return_pct']:+.2f}%")
    print(f" * 최대 낙폭 (MDD):              {results['mdd_pct']:.2f}%")
    print(f" * 샤프 지수 (Sharpe Ratio):     {results['sharpe_ratio']:.2f}")
    print(f" * 총 거래 횟수 (Total Trades):  {results['total_trades']}회")
    print(f" * 승률 (Win Rate):              {results['win_rate_pct']:.2f}%")
    print(f" * 손익비 (Profit Factor):       {results['profit_factor']:.2f}")
    print(f" * 총 총수익 (Gross Profit):     ${results.get('gross_profit', 0):,.2f}")
    print(f" * 총 총손실 (Gross Loss):       ${results.get('gross_loss', 0):,.2f}")
    print("=" * 55)

    # 5. 거래 로그 저장
    trades_df = results['trades_df']
    if not trades_df.empty:
        trades_csv_path = os.path.join("data", "trades_log.csv")
        trades_df.to_csv(trades_csv_path, index=False)
        print(f"\n[Saved] 전체 매매 로그 저장 완료: {trades_csv_path}")

        # 엔진별 세부 통계
        print("\n--- 엔진별 거래 성과 ---")
        for eng in ["MEAN_REVERSION", "TREND_FOLLOWING"]:
            eng_trades = trades_df[trades_df['engine'] == eng]
            if not eng_trades.empty:
                eng_wins = eng_trades[eng_trades['pnl'] > 0]
                eng_wr = (len(eng_wins) / len(eng_trades)) * 100
                eng_pnl = eng_trades['pnl'].sum()
                print(f"  [{eng}] 거래: {len(eng_trades)}회 | 승률: {eng_wr:.1f}% | 누적 PnL: ${eng_pnl:,.2f}")

    # 6. 차트 시각화 저장
    equity_arr = np.array(results['equity_curve'])
    peak = np.maximum.accumulate(equity_arr)
    drawdowns = ((equity_arr - peak) / peak) * 100.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # 상단: 자산 곡선
    ax1.plot(results['timestamps'], equity_arr, color='royalblue', linewidth=1.8, label='RADE Equity Curve ($)')
    ax1.axhline(results['initial_capital'], color='gray', linestyle='--', alpha=0.7, label='Initial Capital ($10,000)')
    ax1.set_title(f"RADE System Futures Backtest (Return: {results['total_return_pct']:+.2f}%, MDD: {results['mdd_pct']:.2f}%, PF: {results['profit_factor']:.2f})", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Account Equity ($)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 하단: Drawdown (MDD)
    ax2.fill_between(results['timestamps'], drawdowns, 0, color='crimson', alpha=0.4, label='Drawdown (%)')
    ax2.plot(results['timestamps'], drawdowns, color='crimson', linewidth=1.0)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Time (Hourly)")
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join("data", "backtest_result.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[Done] 백테스트 결과 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_full_backtest()
