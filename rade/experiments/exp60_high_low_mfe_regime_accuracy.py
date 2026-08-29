"""
[실험 60] High/Low 기반 최대 유리 파동(MFE) 및 ±1.5% 대칭 기준 HMM 국면 판정 정밀 적중률
- 사용자 피드백 반영:
  1. 상승 적중: N시간 내 최고가(High) 상승률 >= +1.5%
  2. 하락 적중: N시간 내 최저가(Low) 하락률 <= -1.5%
  3. 횡보 적중: N시간 내 최고가 <= +1.5% AND 최저가 >= -1.5% (박스권 완전 갇힘)
- 추가 분석:
  - +2.0% / -2.0% 대형 파동 기준
  - +3.0% / -3.0% 초대형 파동 기준
"""
import os
import sys
import numpy as np
import pandas as pd

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


def run_experiment_60():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    df_asym = get_regime_series(df_ind, base_th=0.74, bear_th=0.80)
    test_df = df_asym.iloc[720:].copy().reset_index(drop=True)

    horizons = [4, 12, 24, 48, 72]
    n_len = len(test_df)

    print("=" * 105)
    print("      [실험 60] High / Low 최대 파동(MFE) 및 ±1.5% 일관 대칭 기준 HMM 국면 정밀 적중률")
    print("=" * 105)

    # N시간 동안의 High 최고가 상승률 및 Low 최저가 하락률 계산
    for h in horizons:
        # t+1부터 t+h까지의 max_high, min_low
        highs = []
        lows = []
        closes = test_df["close"].values
        high_arr = test_df["high"].values
        low_arr = test_df["low"].values
        
        for i in range(n_len):
            if i + h < n_len:
                win_high = np.max(high_arr[i+1 : i+h+1])
                win_low = np.min(low_arr[i+1 : i+h+1])
                c = closes[i]
                highs.append((win_high - c) / c * 100.0)
                lows.append((win_low - c) / c * 100.0)
            else:
                highs.append(np.nan)
                lows.append(np.nan)
        
        test_df[f"max_up_{h}h"] = highs
        test_df[f"max_down_{h}h"] = lows

    # 1. ±1.5% 기준 적중률 분석
    print("\n[1. ±1.5% 대칭 기준: 구간 내 최대 파동 적중률 (Hit Rate, %)]")
    print(" * BULL 적중 : N시간 내 최고가(High) >= +1.5%")
    print(" * BEAR 적중 : N시간 내 최저가(Low)  <= -1.5%")
    print(" * RANGE 적중: N시간 내 최고가 <= +1.5% AND 최저가 >= -1.5% (박스권 완전 갇힘)")
    print("-" * 105)
    print(f"{'국면 상태':<18} | {'4시간 내':<15} | {'12시간 내':<15} | {'24시간 내(⭐)':<15} | {'48시간 내':<15} | {'72시간 내':<15}")
    print("-" * 105)

    for state in ["BULL_TREND", "BEAR_PANIC", "RANGE"]:
        sub = test_df[test_df["regime_state"] == state]
        hit_rates = []
        for h in horizons:
            ups = sub[f"max_up_{h}h"].dropna()
            downs = sub[f"max_down_{h}h"].dropna()
            if state == "BULL_TREND":
                hr = (ups >= 1.5).mean() * 100.0
            elif state == "BEAR_PANIC":
                hr = (downs <= -1.5).mean() * 100.0
            else: # RANGE
                hr = ((ups <= 1.5) & (downs >= -1.5)).mean() * 100.0
            hit_rates.append(f"{hr:5.1f}%")
        
        state_label = "🟢 BULL (상승 파동)" if state == "BULL_TREND" else ("🔴 BEAR (폭락 파동)" if state == "BEAR_PANIC" else "🟡 RANGE (박스 유지)")
        print(f"{state_label:<18} | {hit_rates[0]:<15} | {hit_rates[1]:<15} | {hit_rates[2]:<15} | {hit_rates[3]:<15} | {hit_rates[4]:<15}")
    print("-" * 105)

    # 2. +2.0% / -2.0% 대형 파동 기준 적중률
    print("\n[2. ±2.0% 대형 파동 기준: 24시간 및 48시간 내 돌파 확률 (%)]")
    print("-" * 105)
    for state in ["BULL_TREND", "BEAR_PANIC", "RANGE"]:
        sub = test_df[test_df["regime_state"] == state]
        hr_24 = (sub["max_up_24h"] >= 2.0).mean() * 100.0 if state == "BULL_TREND" else ((sub["max_down_24h"] <= -2.0).mean() * 100.0 if state == "BEAR_PANIC" else ((sub["max_up_24h"] <= 2.0) & (sub["max_down_24h"] >= -2.0)).mean() * 100.0)
        hr_48 = (sub["max_up_48h"] >= 2.0).mean() * 100.0 if state == "BULL_TREND" else ((sub["max_down_48h"] <= -2.0).mean() * 100.0 if state == "BEAR_PANIC" else ((sub["max_up_48h"] <= 2.0) & (sub["max_down_48h"] >= -2.0)).mean() * 100.0)
        print(f" * {state:<15} : 24시간 내 2% 파동 적중 = {hr_24:5.1f}% | 48시간 내 2% 파동 적중 = {hr_48:5.1f}%")
    print("-" * 105)

    # 3. 국면별 평균 최대 유리 파동 크기 (Average MFE)
    print("\n[3. 국면별 도달하는 평균 최대 파동 크기 (Mean MFE, %)]")
    print(f"{'국면 상태':<18} | {'24시간 내 평균 최고가(High)':<26} | {'24시간 내 평균 최저가(Low)':<26}")
    print("-" * 105)
    for state in ["BULL_TREND", "BEAR_PANIC", "RANGE"]:
        sub = test_df[test_df["regime_state"] == state]
        mean_up_24 = sub["max_up_24h"].dropna().mean()
        mean_down_24 = sub["max_down_24h"].dropna().mean()
        state_label = "🟢 BULL_TREND" if state == "BULL_TREND" else ("🔴 BEAR_PANIC" if state == "BEAR_PANIC" else "🟡 RANGE")
        print(f"{state_label:<18} | {mean_up_24:+6.2f}% (위로 찌름)           | {mean_down_24:+6.2f}% (아래로 찌름)")
    print("-" * 105)


if __name__ == "__main__":
    run_experiment_60()
