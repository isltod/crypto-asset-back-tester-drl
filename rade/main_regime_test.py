"""
Phase 1 국면 탐지 파이프라인 검증 스크립트
1. 바이낸스 BTCUSDT 1H 데이터 다운로드
2. 기술적 지표 및 HMM/Rule 국면 확률 산출
3. 국면 분류 결과 통계 및 시각화 저장
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
import pandas as pd
from rade.data.binance_fetcher import BinanceFuturesFetcher
from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager


def main():
    print("=== [Phase 1] RADE 국면 탐지 파이프라인 테스트 시작 ===")

    # 1. 데이터 수집
    fetcher = BinanceFuturesFetcher(data_dir="data")
    # 2023년 1월 ~ 2024년 6월까지 (약 1.5년치)
    df_raw = fetcher.get_or_download_data(
        symbol="BTCUSDT",
        interval="1h",
        start_time_str="2023-01-01 00:00:00",
        end_time_str="2024-06-01 00:00:00"
    )

    if df_raw.empty:
        print("[Error] 데이터 로드 실패")
        return

    print(f"데이터 로드 완료: 총 {len(df_raw)}개 캔들")

    # 2. 지표 추가
    print("기술적 지표 계산 중...")
    df_indicators = add_all_indicators(df_raw)

    # 3. 국면 탐지 관리자 실행
    print("HMM 및 룰 기반 국면 시뮬레이션 계산 중...")
    manager = RegimeManager(
        hmm_window=720,         # 30일 학습 윈도우
        retrain_interval=168,   # 1주일마다 재학습
        hysteresis_upper=0.65,
        hysteresis_lower=0.35,
        cooldown_bars=3
    )

    df_regime = manager.calculate_regime_probabilities(df_indicators)

    # 4. 결과 통계 출력
    valid_data = df_regime.dropna(subset=['regime_trend_prob'])
    state_counts = valid_data['regime_state'].value_counts()
    print("\n--- 국면 분류 결과 요약 ---")
    print(f"유효 분석 캔들 수: {len(valid_data)}개")
    for state, count in state_counts.items():
        pct = (count / len(valid_data)) * 100
        print(f"  * {state}: {count}개 ({pct:.1f}%)")

    # 5. 차트 시각화 저장
    print("\n국면 시각화 차트 생성 중...")
    # 최근 500개 캔들 샘플링하여 시각화
    sample_df = valid_data.tail(500).reset_index(drop=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1.5, 1.5]})

    # 상단: 가격 차트 + 국면 배경색
    ax1.plot(sample_df.index, sample_df['close'], label='BTCUSDT Close', color='black', alpha=0.8, linewidth=1.2)
    ax1.plot(sample_df.index, sample_df['bb_upper'], label='BB Upper', color='gray', linestyle='--', alpha=0.5)
    ax1.plot(sample_df.index, sample_df['bb_lower'], label='BB Lower', color='gray', linestyle='--', alpha=0.5)

    # 국면별 배경 하이라이트 (TREND: 초록/주황, RANGE: 파랑)
    for i in range(len(sample_df) - 1):
        if sample_df.loc[i, 'regime_state'] == 'TREND':
            ax1.axvspan(i, i+1, color='orange', alpha=0.15)
        else:
            ax1.axvspan(i, i+1, color='lightblue', alpha=0.15)

    ax1.set_title("BTCUSDT 1H Price & Market Regimes (Orange: Trend, Blue: Range)")
    ax1.set_ylabel("Price (USDT)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 중단: HMM 확률 및 Rule 추세 확률 & 결합 확률
    ax2.plot(sample_df.index, sample_df['hmm_trend_prob'], label='HMM Trend Prob', color='purple', alpha=0.7)
    ax2.plot(sample_df.index, sample_df['rule_trend_prob'], label='Rule Trend Prob', color='teal', alpha=0.7)
    ax2.plot(sample_df.index, sample_df['regime_trend_prob'], label='Combined Prob', color='red', linewidth=1.5)
    ax2.axhline(0.65, color='orange', linestyle=':', label='Hysteresis Upper (0.65)')
    ax2.axhline(0.35, color='blue', linestyle=':', label='Hysteresis Lower (0.35)')
    ax2.set_ylabel("Trend Probability")
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # 하단: ADX & Choppiness Index
    ax3.plot(sample_df.index, sample_df['adx'], label='ADX(14)', color='green')
    ax3.plot(sample_df.index, sample_df['choppiness'], label='Choppiness(14)', color='brown')
    ax3.axhline(25, color='green', linestyle=':', alpha=0.7, label='ADX Trend Line (25)')
    ax3.axhline(61.8, color='brown', linestyle=':', alpha=0.7, label='CI Range Line (61.8)')
    ax3.set_xlabel("Bars (Recent 500 Hours)")
    ax3.set_ylabel("Indicator Value")
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join("data", "regime_test_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Done] 국면 시각화 차트 저장 완료: {plot_path}")


if __name__ == "__main__":
    main()
