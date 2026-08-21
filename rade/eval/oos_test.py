"""
[Out-of-Sample] 완전 미지 구간 (2024.06.01 ~ 2024.12.31, 최근 7개월) 블라인드 테스트
- 목적: 2021~2024.06까지의 연구/개발에 단 한 번도 노출되지 않은 순수 Out-of-Sample 데이터에서 실전 성능 평가
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rade.data_collector.binance_fetcher import BinanceFuturesFetcher
from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from rade.backtest.simulator import BacktestSimulator


def run_out_of_sample_test():
    print("=== [P1 Out-of-Sample 블라인드 검증] 2024.06.01 ~ 2024.12.31 (최근 7개월) 시작 ===")

    fetcher = BinanceFuturesFetcher(data_dir="data")
    cache_file = os.path.join("data", "BTCUSDT_1h_2024_OOS.csv")

    if os.path.exists(cache_file):
        print(f"[Cache] OOS 데이터 로드: {cache_file}")
        df_raw = pd.read_csv(cache_file)
        df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
    else:
        print("[Download] 바이낸스에서 2024.06.01 ~ 2024.12.31 데이터 수집 중...")
        # HMM 720봉 윈도우 확보를 위해 2024.05.01부터 수집
        df_raw = fetcher.fetch_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_time_str="2024-05-01 00:00:00",
            end_time_str="2024-12-31 23:59:59"
        )
        if not df_raw.empty:
            df_raw.to_csv(cache_file, index=False)
            print(f"[Saved] OOS 데이터 저장 완료: {cache_file}")

    print(f"총 수집된 캔들: {len(df_raw)}개 ({df_raw['datetime'].iloc[0]} ~ {df_raw['datetime'].iloc[-1]})")

    df_ind = add_all_indicators(df_raw)
    manager = RegimeManager(hmm_window=720, retrain_interval=168, hysteresis_upper=0.65, hysteresis_lower=0.35, cooldown_bars=3)
    df_proc = manager.calculate_regime_probabilities(df_ind)
    
    # 2024.06.01 이후만 순수 OOS 구간으로 슬라이스
    test_df = df_proc.dropna(subset=['regime_trend_prob']).reset_index(drop=True)
    test_df = test_df[test_df['datetime'] >= "2024-06-01 00:00:00+00:00"].reset_index(drop=True)
    print(f"순수 OOS 백테스트 구간: 총 {len(test_df)}개 캔들 ({test_df['datetime'].iloc[0]} ~ {test_df['datetime'].iloc[-1]})")

    sim = BacktestSimulator(initial_capital=10000.0, risk_per_trade_pct=0.02, leverage=3.0)
    res = sim.run(test_df)
    trades_df = res['trades_df']

    print("\n" + "=" * 65)
    print("      [ OUT-OF-SAMPLE (2024.06 ~ 2024.12) BLIND TEST REPORT ]      ")
    print("=" * 65)
    print(f" * OOS 테스트 기간:       2024년 6월 1일 ~ 2024년 12월 31일 (7개월)")
    print(f" * 초기 자본:             ${res['initial_capital']:,.2f}")
    print(f" * 최종 자산:             ${res['final_equity']:,.2f}  (+${res['final_equity'] - res['initial_capital']:,.2f} 순수익)")
    print(f" * 7개월 OOS 총 수익률:   {res['total_return_pct']:+.2f}%")
    print(f" * OOS 최대 낙폭(MDD):    {res['mdd_pct']:.2f}%")
    print(f" * OOS 샤프 지수:         {res['sharpe_ratio']:.2f}")
    print(f" * 총 거래 횟수:          {res['total_trades']}회")
    print(f" * 승률 (Win Rate):       {res['win_rate_pct']:.2f}%")
    print(f" * 손익비 (PF):           {res['profit_factor']:.2f}")
    print(f" * 총 총수익 (Gross):     ${res['gross_profit']:,.2f}")
    print(f" * 총 총손실 (Gross):     ${res['gross_loss']:,.2f}")
    print("=" * 65)

    if not trades_df.empty:
        print("\n--- 엔진별 성과 ---")
        for eng in ["MEAN_REVERSION", "TREND_FOLLOWING"]:
            et = trades_df[trades_df['engine'] == eng]
            if not et.empty:
                ew = et[et['pnl'] > 0]
                print(f"  [{eng}] 거래: {len(et)}회 | 승률: {(len(ew)/len(et))*100:.1f}% | PnL: ${et['pnl'].sum():+,.2f}")

    # 차트 저장
    plt.figure(figsize=(12, 6))
    plt.plot(res['timestamps'], res['equity_curve'], color='royalblue', linewidth=1.8, label='OOS Equity Curve ($)')
    plt.axhline(10000.0, color='gray', linestyle='--', label='Initial ($10k)')
    plt.title(f"RADE Out-of-Sample Test (2024.06~2024.12): Return {res['total_return_pct']:+.2f}%, MDD {res['mdd_pct']:.2f}%, PF {res['profit_factor']:.2f}", fontsize=12)
    plt.xlabel("Timeline")
    plt.ylabel("Account Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    chart_path = os.path.join("data", "oos_2024_result_plot.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Done] OOS 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_out_of_sample_test()
