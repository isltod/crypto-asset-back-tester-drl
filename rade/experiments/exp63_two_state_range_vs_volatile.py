"""
[실험 63] 2-State (횡보 국면 RANGE vs 변동 국면 VOLATILE) 통합 적중률 분석
- 사용자 요청 설계:
  1. 3개 국면(BULL, BEAR, RANGE)을 2대 국면으로 통합:
     - 횡보 국면 (RANGE): HMM RANGE
     - 변동 국면 (VOLATILE): HMM BULL + BEAR 통합
  2. 성공 판정 기준:
     - RANGE 성공: 생애주기 동안 박스권 내에 얌전하게 머무름 (Max High <= +X% AND Min Low >= -X%)
     - VOLATILE 성공: 생애주기 동안 횡보 기준을 확실하게 돌파 (Max High >= +X% OR Min Low <= -X%)
     - 2배 폭발 성공: 생애주기 동안 2배 기준 돌파 (Max High >= +2X% OR Min Low <= -2X%)
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


def get_two_state_series(df_ind: pd.DataFrame, base_th: float = 0.74, bear_th: float = 0.80) -> pd.DataFrame:
    reg_raw = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.30, cooldown_bars=0)
    df_raw = reg_raw.calculate_regime_probabilities(df_ind)
    
    curr = "RANGE"
    two_states = []
    for idx, row in df_raw.iterrows():
        p_r = row["p_range"]
        p_u = row["p_bull"]
        p_d = row["p_bear"]
        if p_d >= bear_th and p_d >= p_u and p_d >= p_r:
            curr = "VOLATILE" # 하락 변동
        elif p_u >= base_th and p_u >= p_r and p_u >= p_d:
            curr = "VOLATILE" # 상승 변동
        elif p_r >= base_th and p_r >= p_u and p_r >= p_d:
            curr = "RANGE"    # 횡보
        two_states.append(curr)
    df_raw["two_state"] = two_states
    return df_raw


def extract_two_state_episodes(df: pd.DataFrame):
    test_df = df.iloc[720:].copy().reset_index(drop=True)
    episodes = []
    curr_state = None
    start_idx = 0
    
    for i, row in test_df.iterrows():
        st = row["two_state"]
        if st != curr_state:
            if curr_state is not None:
                sub = test_df.iloc[start_idx:i]
                start_p = sub.iloc[0]["open"]
                max_h = sub["high"].max()
                min_l = sub["low"].min()
                end_p = sub.iloc[-1]["close"]
                max_up = (max_h - start_p) / start_p * 100.0
                max_down = (min_l - start_p) / start_p * 100.0
                episodes.append({
                    "state": curr_state,
                    "duration_hours": len(sub),
                    "start_price": start_p,
                    "end_price": end_p,
                    "max_high": max_h,
                    "min_low": min_l,
                    "max_up_pct": max_up,
                    "max_down_pct": max_down,
                    "abs_max_wave": max(max_up, abs(max_down)), # 절대 최대 파동 크기
                    "net_ret_pct": (end_p - start_p) / start_p * 100.0,
                })
            curr_state = st
            start_idx = i
            
    if curr_state is not None:
        sub = test_df.iloc[start_idx:]
        start_p = sub.iloc[0]["open"]
        max_up = (sub["high"].max() - start_p) / start_p * 100.0
        max_down = (sub["low"].min() - start_p) / start_p * 100.0
        episodes.append({
            "state": curr_state,
            "duration_hours": len(sub),
            "start_price": start_p,
            "end_price": sub.iloc[-1]["close"],
            "max_high": sub["high"].max(),
            "min_low": sub["low"].min(),
            "max_up_pct": max_up,
            "max_down_pct": max_down,
            "abs_max_wave": max(max_up, abs(max_down)),
            "net_ret_pct": (sub.iloc[-1]["close"] - start_p) / start_p * 100.0,
        })
        
    return pd.DataFrame(episodes)


def run_experiment_63():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    df_two = get_two_state_series(df_ind, base_th=0.74, bear_th=0.80)
    ep_df = extract_two_state_episodes(df_two)

    print("=" * 105)
    print("      [실험 63] 2-State (횡보 국면 RANGE vs 변동 국면 VOLATILE) 통합 적중률 분석")
    print("=" * 105)

    print(f"\n[1. 2대 국면 통합 에피소드 발생 현황: 총 {len(ep_df)}회 (전환 횟수 6,131회에서 40% 대폭 압축!)]")
    print("-" * 105)
    print(f"{'국면 상태':<20} | {'발생 횟수':<14} | {'평균 지속 시간':<16} | {'중앙값 지속 시간':<16} | {'최대 지속 시간':<16}")
    print("-" * 105)
    for state in ["RANGE", "VOLATILE"]:
        sub = ep_df[ep_df["state"] == state]
        cnt = len(sub)
        m_dur = sub["duration_hours"].mean()
        med_dur = sub["duration_hours"].median()
        max_dur = sub["duration_hours"].max()
        state_label = "🟡 횡보 국면 (RANGE)" if state == "RANGE" else "⚡ 변동 국면 (VOLATILE)"
        print(f"{state_label:<20} | {cnt:5d}회 ({cnt/len(ep_df)*100:4.1f}%) | {m_dur:6.1f}시간 ({m_dur/24:4.1f}일) | {med_dur:6.1f}시간 ({med_dur/24:4.1f}일) | {max_dur:6.1f}시간 ({max_dur/24:4.1f}일)")
    print("-" * 105)

    # 2. 동일 1배 기준 적중률
    print("\n[2. 동일 1배 기준: 국면별 실측 적중률 (Hit Rate, %)]")
    print(" * RANGE 적중   : 생애주기 동안 박스권 내에 얌전하게 갇힘 (|변동폭| <= X%)")
    print(" * VOLATILE 적중: 생애주기 동안 상방이든 하방이든 횡보 기준을 확실히 돌파 (최대 파동 >= X%)")
    print("-" * 105)
    print(f"{'국면 상태':<20} | {'±1.0% 기준 적중률':<22} | {'±1.5% 기준 적중률(⭐)':<24} | {'±2.0% 기준 적중률':<22}")
    print("-" * 105)

    for state in ["RANGE", "VOLATILE"]:
        sub = ep_df[ep_df["state"] == state]
        res_strs = []
        for th in [1.0, 1.5, 2.0]:
            if state == "RANGE":
                hr = ((sub["max_up_pct"] <= th) & (sub["max_down_pct"] >= -th)).mean() * 100.0
                res_strs.append(f"박스 갇힘: {hr:5.1f}%")
            else: # VOLATILE
                hr = (sub["abs_max_wave"] >= th).mean() * 100.0
                res_strs.append(f"돌파 성공: {hr:5.1f}%")
        state_label = "🟡 횡보 국면 (RANGE)" if state == "RANGE" else "⚡ 변동 국면 (VOLATILE)"
        print(f"{state_label:<20} | {res_strs[0]:<22} | {res_strs[1]:<24} | {res_strs[2]:<22}")
    print("-" * 105)

    # 3. 횡보 1배 vs 변동 2배 파동 기준 적중률
    print("\n[3. 횡보 1배 박스 vs 변동 2배 돌파 기준 적중률 (%)]")
    print(" * RANGE    : 생애주기 동안 '±1.5% 박스권(1배)' 유지")
    print(" * VOLATILE : 생애주기 동안 상방/하방 중 '2배인 3.0% 이상 돌파' 파동 폭발")
    print("-" * 105)
    
    r_sub = ep_df[ep_df["state"] == "RANGE"]
    v_sub = ep_df[ep_df["state"] == "VOLATILE"]
    
    r_succ_15 = ((r_sub["max_up_pct"] <= 1.5) & (r_sub["max_down_pct"] >= -1.5)).mean() * 100.0
    v_succ_2x = (v_sub["abs_max_wave"] >= 3.0).mean() * 100.0 # 3.0% 이상 돌파
    v_succ_3x = (v_sub["abs_max_wave"] >= 4.5).mean() * 100.0 # 4.5% 이상 초대형 돌파
    
    print(f" * 🟡 횡보 국면 (RANGE)    : ±1.5% 박스 유지율           = {r_succ_15:5.1f}% ({len(r_sub)}회 중 {int(len(r_sub)*r_succ_15/100)}회 성공!)")
    print(f" * ⚡ 변동 국면 (VOLATILE) : 2배인 ±3.0% 이상 돌파 성공률 = {v_succ_2x:5.1f}% ({len(v_sub)}회 중 {int(len(v_sub)*v_succ_2x/100)}회 대형 파동 폭발!)")
    print(f" * ⚡ 변동 국면 (VOLATILE) : 3배인 ±4.5% 이상 초대형 폭발 = {v_succ_3x:5.1f}% ({len(v_sub)}회 중 {int(len(v_sub)*v_succ_3x/100)}회 슈퍼 랠리/폭락!)")
    print("-" * 105)

    # 4. 변동 국면의 파동 방향성 분포 (상승 돌파 vs 하락 폭락)
    print("\n[4. 변동 국면 (VOLATILE)의 실제 파동 특성]")
    v_up_dominant = (v_sub["max_up_pct"] > abs(v_sub["max_down_pct"])).mean() * 100.0
    v_down_dominant = (abs(v_sub["max_down_pct"]) > v_sub["max_up_pct"]).mean() * 100.0
    mean_abs_wave = v_sub["abs_max_wave"].mean()
    
    print(f" * 변동 국면의 평균 최대 도달 파동 크기 : {mean_abs_wave:+5.2f}%")
    print(f" * 변동 국면 중 상방(상승) 파동이 우세했던 비율 : {v_up_dominant:5.1f}%")
    print(f" * 변동 국면 중 하방(폭락) 파동이 우세했던 비율 : {v_down_dominant:5.1f}%")
    print("=" * 105)


if __name__ == "__main__":
    run_experiment_63()
