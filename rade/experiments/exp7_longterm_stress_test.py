"""
[실험 7] 3.5년 장기 스트레스 테스트 (2021.01.01 ~ 2024.06.01)
- 포함된 시장 위기: 2021년 5월 대폭락, 2021년 11월 69k, 2022년 루나/테라, FTX 파산, 2023~2024 대불장
- 목적: 3.5년 장기 데이터(약 30,000시간) 및 연도별 하락장/상승장 분해 검증
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rade.data.binance_fetcher import BinanceFuturesFetcher
from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from rade.backtest.simulator import BacktestSimulator


def run_longterm_stress_test():
    print("=== [실험 7] 3.5년(2021.01.01 ~ 2024.06.01) 장기 스트레스 테스트 시작 ===")

    fetcher = BinanceFuturesFetcher(data_dir="data")
    cache_file = os.path.join("data", "BTCUSDT_1h_2021_2024.csv")

    if os.path.exists(cache_file):
        print(f"[Cache] 3.5년 로컬 캐시 로드: {cache_file}")
        df_raw = pd.read_csv(cache_file)
        df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
    else:
        print("[Download] 바이낸스에서 2021.01.01 ~ 2024.06.01 (약 30,000개 캔들) 수집 시작...")
        df_raw = fetcher.fetch_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time_str="2021-01-01 00:00:00",
            end_time_str="2024-06-01 00:00:00"
        )
        if not df_raw.empty:
            df_raw.to_csv(cache_file, index=False)
            print(f"[Saved] 3.5년 데이터 저장 완료: {cache_file}")

    print(f"\n총 로드된 캔들 수: {len(df_raw)}개 ({df_raw['datetime'].iloc[0]} ~ {df_raw['datetime'].iloc[-1]})")

    # 2. 지표 및 HMM 국면 분석
    print("기술적 지표 및 HMM 국면(720봉 윈도우, 168봉 재학습) 계산 중...")
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

    print(f"유효 백테스트 구간: 총 {len(test_df)}개 캔들 ({test_df['datetime'].iloc[0]} ~ {test_df['datetime'].iloc[-1]})")

    # 3. 2.0% Risk 기준 3.5년 장기 백테스트 실행
    print("\n선물 백테스트 시뮬레이션 가동...")
    sim = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.02, # 2.0% Risk
        leverage=3.0
    )
    res = sim.run(test_df)

    # 4. 종합 성과 리포트 출력
    print("\n" + "=" * 65)
    print("        [ RADE 3.5-YEAR LONG-TERM STRESS TEST REPORT ]        ")
    print("=" * 65)
    print(f" * 테스트 기간:          2021년 1월 ~ 2024년 6월 (총 41개월, 3.5년)")
    print(f" * 초기 자본:            ${res['initial_capital']:,.2f}")
    print(f" * 최종 자산:            ${res['final_equity']:,.2f}  (+${res['final_equity'] - res['initial_capital']:,.2f} 순수익)")
    print(f" * 3.5년 총 수익률:      {res['total_return_pct']:+.2f}%")
    print(f" * 3.5년 최대 낙폭(MDD): {res['mdd_pct']:.2f}%")
    print(f" * 샤프 지수 (Sharpe):   {res['sharpe_ratio']:.2f}")
    print(f" * 총 거래 횟수:         {res['total_trades']}회 (3.5년 누적)")
    print(f" * 승률 (Win Rate):      {res['win_rate_pct']:.2f}%")
    print(f" * 손익비 (PF):          {res['profit_factor']:.2f}")
    print(f" * 총 총수익 (Gross):    ${res['gross_profit']:,.2f}")
    print(f" * 총 총손실 (Gross):    ${res['gross_loss']:,.2f}")
    print("=" * 65)

    # 5. 연도별 성과 분해 분석
    trades_df = res['trades_df']
    if not trades_df.empty:
        trades_df['year'] = pd.to_datetime(trades_df['exit_time']).dt.year
        print("\n--- 연도별 세부 성과 (상승장/하락장/횡보장 분해) ---")
        for yr, group in trades_df.groupby('year'):
            yr_wins = group[group['pnl'] > 0]
            yr_losses = group[group['pnl'] < 0]
            yr_pnl = group['pnl'].sum()
            yr_wr = (len(yr_wins) / len(group)) * 100.0
            yr_pf = yr_wins['pnl'].sum() / abs(yr_losses['pnl'].sum()) if not yr_losses.empty else 0.0
            print(f"  [{yr}년] 거래: {len(group):2d}회 | 승률: {yr_wr:4.1f}% | PF: {yr_pf:.2f} | 누적 PnL: ${yr_pnl:+,.2f}")

    # 6. 차트 시각화 저장
    equity_arr = np.array(res['equity_curve'])
    peak = np.maximum.accumulate(equity_arr)
    drawdowns = ((equity_arr - peak) / peak) * 100.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(res['timestamps'], equity_arr, color='darkgreen', linewidth=1.8, label=f"RADE 3.5Y Equity (${res['final_equity']:,.0f})")
    ax1.axhline(10000.0, color='gray', linestyle='--', label='Initial ($10,000)')
    ax1.set_title(f"RADE 3.5-Year Stress Test (Return: {res['total_return_pct']:+.2f}%, MDD: {res['mdd_pct']:.2f}%, PF: {res['profit_factor']:.2f})", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Account Equity ($)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(res['timestamps'], drawdowns, 0, color='crimson', alpha=0.4, label='Drawdown (%)')
    ax2.plot(res['timestamps'], drawdowns, color='crimson', linewidth=1.0)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Time (2021 ~ 2024)")
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join("data", "exp7_longterm_stress_test_plot.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Done] 3.5년 장기 스트레스 테스트 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_longterm_stress_test()
