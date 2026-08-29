"""
[실험 61] 국면 에피소드(Regime Episode) 단위 실측 적중률 및 파동 분석
- 사용자 피드백 100% 반영:
  - 고정 N시간 윈도우가 아니라, 연속된 동일 국면 구간(에피소드) 단위로 시계열 분할
  - 각 에피소드가 시작해서 끝날 때까지의 '라이프타임(지속 시간)' 내에서 시작가 대비 High/Low 파동 분석
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


def extract_episodes(df: pd.DataFrame):
    """연속된 국면을 개별 에피소드로 분할 및 집계"""
    test_df = df.iloc[720:].copy().reset_index(drop=True)
    
    episodes = []
    curr_state = None
    start_idx = 0
    
    for i, row in test_df.iterrows():
        st = row["regime_state"]
        if st != curr_state:
            if curr_state is not None:
                # 이전 에피소드 마무리
                sub = test_df.iloc[start_idx:i]
                start_p = sub.iloc[0]["open"]
                max_h = sub["high"].max()
                min_l = sub["low"].min()
                end_p = sub.iloc[-1]["close"]
                duration = len(sub)
                
                episodes.append({
                    "state": curr_state,
                    "duration_hours": duration,
                    "start_price": start_p,
                    "end_price": end_p,
                    "max_high": max_h,
                    "min_low": min_l,
                    "max_up_pct": (max_h - start_p) / start_p * 100.0,
                    "max_down_pct": (min_l - start_p) / start_p * 100.0,
                    "net_ret_pct": (end_p - start_p) / start_p * 100.0,
                })
            curr_state = st
            start_idx = i
            
    # 마지막 에피소드
    if curr_state is not None:
        sub = test_df.iloc[start_idx:]
        start_p = sub.iloc[0]["open"]
        episodes.append({
            "state": curr_state,
            "duration_hours": len(sub),
            "start_price": start_p,
            "end_price": sub.iloc[-1]["close"],
            "max_high": sub["high"].max(),
            "min_low": sub["low"].min(),
            "max_up_pct": (sub["high"].max() - start_p) / start_p * 100.0,
            "max_down_pct": (sub["low"].min() - start_p) / start_p * 100.0,
            "net_ret_pct": (sub.iloc[-1]["close"] - start_p) / start_p * 100.0,
        })
        
    return pd.DataFrame(episodes)


def run_experiment_61():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    df_asym = get_regime_series(df_ind, base_th=0.74, bear_th=0.80)
    ep_df = extract_episodes(df_asym)

    print("=" * 105)
    print("      [실험 61] 국면 에피소드(Regime Episode) 단위 실측 적중률 및 생애주기 파동 분석")
    print("=" * 105)

    print(f"\n[1. 4개년 총 발생 국면 에피소드 수: 총 {len(ep_df)}회]")
    print("-" * 105)
    print(f"{'국면 상태':<18} | {'발생 횟수':<12} | {'평균 지속 시간':<16} | {'중앙값 지속 시간':<16} | {'최대 지속 시간':<16}")
    print("-" * 105)
    for state in ["BULL_TREND", "BEAR_PANIC", "RANGE"]:
        sub = ep_df[ep_df["state"] == state]
        cnt = len(sub)
        m_dur = sub["duration_hours"].mean()
        med_dur = sub["duration_hours"].median()
        max_dur = sub["duration_hours"].max()
        state_label = "🟢 BULL_TREND" if state == "BULL_TREND" else ("🔴 BEAR_PANIC" if state == "BEAR_PANIC" else "🟡 RANGE")
        print(f"{state_label:<18} | {cnt:5d}회 ({cnt/len(ep_df)*100:4.1f}%) | {m_dur:6.1f}시간 ({m_dur/24:4.1f}일) | {med_dur:6.1f}시간 ({med_dur/24:4.1f}일) | {max_dur:6.1f}시간 ({max_dur/24:4.1f}일)")
    print("-" * 105)

    print("\n[2. 에피소드 생애주기 내 실측 적중률 (Hit Rate, %)]")
    print(" * BULL 적중 : 에피소드 지속 중 '최고가 상승률 >= +1.5%' 달성")
    print(" * BEAR 적중 : 에피소드 지속 중 '최저가 하락률 <= -1.5%' 폭락 달성")
    print(" * RANGE 적중: 에피소드 지속 중 '최고가 <= +1.5% AND 최저가 >= -1.5%' (박스권 완전 유지)")
    print("-" * 105)
    print(f"{'국면 상태':<18} | {'±1.0% 기준 적중률':<18} | {'±1.5% 기준 적중률(⭐)':<22} | {'±2.0% 대형 파동 적중률':<22} | {'±3.0% 초대형 파동':<18}")
    print("-" * 105)

    for state in ["BULL_TREND", "BEAR_PANIC", "RANGE"]:
        sub = ep_df[ep_df["state"] == state]
        if state == "BULL_TREND":
            hr_10 = (sub["max_up_pct"] >= 1.0).mean() * 100.0
            hr_15 = (sub["max_up_pct"] >= 1.5).mean() * 100.0
            hr_20 = (sub["max_up_pct"] >= 2.0).mean() * 100.0
            hr_30 = (sub["max_up_pct"] >= 3.0).mean() * 100.0
        elif state == "BEAR_PANIC":
            hr_10 = (sub["max_down_pct"] <= -1.0).mean() * 100.0
            hr_15 = (sub["max_down_pct"] <= -1.5).mean() * 100.0
            hr_20 = (sub["max_down_pct"] <= -2.0).mean() * 100.0
            hr_30 = (sub["max_down_pct"] <= -3.0).mean() * 100.0
        else: # RANGE
            hr_10 = ((sub["max_up_pct"] <= 1.0) & (sub["max_down_pct"] >= -1.0)).mean() * 100.0
            hr_15 = ((sub["max_up_pct"] <= 1.5) & (sub["max_down_pct"] >= -1.5)).mean() * 100.0
            hr_20 = ((sub["max_up_pct"] <= 2.0) & (sub["max_down_pct"] >= -2.0)).mean() * 100.0
            hr_30 = ((sub["max_up_pct"] <= 3.0) & (sub["max_down_pct"] >= -3.0)).mean() * 100.0

        state_label = "🟢 BULL (상승 파동)" if state == "BULL_TREND" else ("🔴 BEAR (폭락 파동)" if state == "BEAR_PANIC" else "🟡 RANGE (박스 유지)")
        print(f"{state_label:<18} | {hr_10:5.1f}%            | {hr_15:5.1f}%                | {hr_20:5.1f}%                | {hr_30:5.1f}%")
    print("-" * 105)

    print("\n[3. 에피소드 생애주기 동안 기록한 평균 파동 크기 및 최종 손익]")
    print(f"{'국면 상태':<18} | {'평균 최대 상승폭 (High)':<22} | {'평균 최대 하락폭 (Low)':<22} | {'에피소드 종료 시점 평균 순손익':<25}")
    print("-" * 105)
    for state in ["BULL_TREND", "BEAR_PANIC", "RANGE"]:
        sub = ep_df[ep_df["state"] == state]
        mean_up = sub["max_up_pct"].mean()
        mean_down = sub["max_down_pct"].mean()
        net_ret = sub["net_ret_pct"].mean()
        state_label = "🟢 BULL_TREND" if state == "BULL_TREND" else ("🔴 BEAR_PANIC" if state == "BEAR_PANIC" else "🟡 RANGE")
        print(f"{state_label:<18} | {mean_up:+6.2f}% (상승 파동)        | {mean_down:+6.2f}% (하락 파동)        | {net_ret:+6.2f}%")
    print("-" * 105)


if __name__ == "__main__":
    run_experiment_61()
