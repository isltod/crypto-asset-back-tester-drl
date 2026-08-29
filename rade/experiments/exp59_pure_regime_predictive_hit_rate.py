"""
[실험 59] 비대칭 HMM 국면 판정 모델 자체의 순수 방향성 예측 적중률 (Pure Regime Predictive Hit Rate)
- 목적: 매매 룰(서브엔진, SL, TP)을 배제하고, HMM 국면 판정 신호 자체의 통계적 알파 및 미래 가격 예측력 검증
- 검증 대상:
  1. 비대칭 HMM 모델 (BULL TH=0.74, RANGE TH=0.74, BEAR TH=0.80)
  2. 대칭 표준 HMM 모델 (TH=0.74 동등)
  3. 초기 HMM 모델 (TH=0.45)
- 측정 타임 호라이즌: 1시간(t+1), 4시간(t+4), 12시간(t+12), 24시간(t+24), 48시간(t+48)
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# UTF-8 콘솔 출력 보장
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager, RegimeState


def get_regime_series(df_ind: pd.DataFrame, base_th: float = 0.74, bear_th: float = 0.80) -> pd.DataFrame:
    reg_raw = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.30, cooldown_bars=0)
    df_raw = reg_raw.calculate_regime_probabilities(df_ind)
    
    curr = RegimeState.RANGE
    asym_states = []
    for idx, row in df_raw.iterrows():
        p_r = row["p_range"]
        p_u = row["p_bull"]
        p_d = row["p_bear"]
        if p_d >= bear_th and p_d >= p_u and p_d >= p_r:
            curr = RegimeState.BEAR_PANIC
        elif p_u >= base_th and p_u >= p_r and p_u >= p_d:
            curr = RegimeState.BULL_TREND
        elif p_r >= base_th and p_r >= p_u and p_r >= p_d:
            curr = RegimeState.RANGE
        asym_states.append(curr.value if hasattr(curr, 'value') else str(curr))
    df_raw["regime_state"] = asym_states
    return df_raw


def analyze_regime_predictive_power(df: pd.DataFrame, model_name: str):
    test_df = df.iloc[720:].copy().reset_index(drop=True)
    
    # 미래 N시간 후 수익률 계산
    horizons = [1, 4, 12, 24, 48]
    for h in horizons:
        test_df[f"fwd_ret_{h}h"] = (test_df["close"].shift(-h) - test_df["close"]) / test_df["close"] * 100.0

    print("\n" + "=" * 105)
    print(f"      [ 분석 모델: {model_name} ]")
    print("=" * 105)

    # 1. 국면별 발생 빈도 및 점유율
    counts = test_df["regime_state"].value_counts()
    total_bars = len(test_df)
    print("\n[1. 4개년 국면별 발생 빈도 및 점유율]")
    for state, cnt in counts.items():
        pct = cnt / total_bars * 100.0
        print(f" * {state:<15} : {cnt:6d} 캔들 ({pct:5.1f}%) | 일수 환산: 약 {cnt/24:5.1f}일")

    # 2. 미래 타임 호라이즌별 방향성 적중률 (Hit Rate, %)
    print("\n[2. 국면별 미래 방향성 적중률 (Hit Rate, %)]")
    print(f"{'국면 상태':<15} | {'1시간 후(t+1)':<16} | {'4시간 후(t+4)':<16} | {'12시간 후(t+12)':<16} | {'24시간 후(t+24)':<16} | {'48시간 후(t+48)':<16}")
    print("-" * 105)

    for state in ["BULL_TREND", "BEAR_PANIC", "RANGE"]:
        sub = test_df[test_df["regime_state"] == state]
        if len(sub) == 0:
            continue
        
        hit_rates = []
        for h in horizons:
            rets = sub[f"fwd_ret_{h}h"].dropna()
            if state == "BULL_TREND":
                # 상승 적중률 (수익률 > 0)
                hr = (rets > 0).mean() * 100.0
            elif state == "BEAR_PANIC":
                # 하락 적중률 (수익률 < 0)
                hr = (rets < 0).mean() * 100.0
            else: # RANGE
                # 횡보 국면: 변동폭이 작을 확률 (|수익률| <= 1.5%)
                hr = (rets.abs() <= 1.5).mean() * 100.0
            hit_rates.append(f"{hr:5.1f}%")
        
        state_label = "BULL (상승 적중)" if state == "BULL_TREND" else ("BEAR (하락 적중)" if state == "BEAR_PANIC" else "RANGE (박스 유지)")
        print(f"{state_label:<15} | {hit_rates[0]:<16} | {hit_rates[1]:<16} | {hit_rates[2]:<16} | {hit_rates[3]:<16} | {hit_rates[4]:<16}")
    print("-" * 105)

    # 3. 국면별 평균 미래 기대 수익률 (Mean Forward Return, %)
    print("\n[3. 국면별 평균 미래 기대 수익률 (Mean Forward Return, %)]")
    print(f"{'국면 상태':<15} | {'1시간 후(t+1)':<16} | {'4시간 후(t+4)':<16} | {'12시간 후(t+12)':<16} | {'24시간 후(t+24)':<16} | {'48시간 후(t+48)':<16}")
    print("-" * 105)

    for state in ["BULL_TREND", "BEAR_PANIC", "RANGE"]:
        sub = test_df[test_df["regime_state"] == state]
        if len(sub) == 0:
            continue
        
        mean_rets = []
        for h in horizons:
            rets = sub[f"fwd_ret_{h}h"].dropna()
            m_ret = rets.mean()
            mean_rets.append(f"{m_ret:+6.2f}%")
        
        print(f"{state:<15} | {mean_rets[0]:<16} | {mean_rets[1]:<16} | {mean_rets[2]:<16} | {mean_rets[3]:<16} | {mean_rets[4]:<16}")
    
    # 전체 시장 평균 벤치마크
    market_means = []
    for h in horizons:
        m_ret = test_df[f"fwd_ret_{h}h"].dropna().mean()
        market_means.append(f"{m_ret:+6.2f}%")
    print("-" * 105)
    print(f"{'전체 시장 베이스':<15} | {market_means[0]:<16} | {market_means[1]:<16} | {market_means[2]:<16} | {market_means[3]:<16} | {market_means[4]:<16}")
    print("-" * 105)


def run_experiment_59():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    print("=" * 105)
    print("      [실험 59] HMM 국면 판정 모델 자체의 순수 방향성 예측 적중률 (Hit Rate) 전수 검증")
    print("=" * 105)

    # 1. 비대칭 80% 숏 모델 (BULL TH=0.74, BEAR TH=0.80) ⭐ 우리 최신 모델
    df_asym_80 = get_regime_series(df_ind, base_th=0.74, bear_th=0.80)
    analyze_regime_predictive_power(df_asym_80, "비대칭 HMM 모델 (BULL 74% vs BEAR 80% 숏)")

    # 2. 대칭 표준 모델 (TH=0.74 동등) ⭐ 우리 공식 표준
    df_sym_74 = get_regime_series(df_ind, base_th=0.74, bear_th=0.74)
    analyze_regime_predictive_power(df_sym_74, "대칭 표준 HMM 모델 (BULL 74% vs BEAR 74%)")


if __name__ == "__main__":
    run_experiment_59()
