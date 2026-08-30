"""
[실험 64] 생애주기 내 [최종 종가 기준] 및 [기준선 체류 봉(Bar) 비율 기준] 국면 정밀 적중률
- 사용자 요청 2대 판정 방식:
  1. 방법 ① [최종 종가 기준]: 국면 종료 시점의 마지막 종가(End Close)가 기준선을 돌파/유지했는가?
  2. 방법 ② [봉(Bar) 비율 기준]: 국면 생애주기 동안 발생한 전체 캔들 중, 기준선 밖에/안에 머문 캔들의 점유율(%)
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
    
    curr3 = "RANGE"
    curr2 = "RANGE"
    two_states = []
    three_states = []
    for idx, row in df_raw.iterrows():
        p_r = row["p_range"]
        p_u = row["p_bull"]
        p_d = row["p_bear"]
        if p_d >= bear_th and p_d >= p_u and p_d >= p_r:
            curr3 = "BEAR_PANIC"
            curr2 = "VOLATILE"
        elif p_u >= base_th and p_u >= p_r and p_u >= p_d:
            curr3 = "BULL_TREND"
            curr2 = "VOLATILE"
        elif p_r >= base_th and p_r >= p_u and p_r >= p_d:
            curr3 = "RANGE"
            curr2 = "RANGE"
        two_states.append(curr2)
        three_states.append(curr3)
        
    df_raw["two_state"] = two_states
    df_raw["three_state"] = three_states
    return df_raw


def analyze_episodes_detailed(df: pd.DataFrame, state_col: str = "two_state"):
    test_df = df.iloc[720:].copy().reset_index(drop=True)
    episodes = []
    curr_state = None
    start_idx = 0
    
    for i, row in test_df.iterrows():
        st = row[state_col]
        if st != curr_state:
            if curr_state is not None:
                sub = test_df.iloc[start_idx:i]
                start_p = sub.iloc[0]["open"]
                end_p = sub.iloc[-1]["close"]
                closes = sub["close"].values
                rets = (closes - start_p) / start_p * 100.0
                
                # 봉 비율 계산 (1.5% 기준)
                # 횡보 캔들: |ret| <= 1.5%
                range_bars_15 = np.sum(np.abs(rets) <= 1.5)
                # 변동 캔들: |ret| > 1.5%
                vol_bars_15 = np.sum(np.abs(rets) > 1.5)
                # 2배 변동 캔들: |ret| >= 3.0%
                vol_bars_30 = np.sum(np.abs(rets) >= 3.0)
                
                episodes.append({
                    "state": curr_state,
                    "duration_hours": len(sub),
                    "start_price": start_p,
                    "end_price": end_p,
                    "net_ret_pct": (end_p - start_p) / start_p * 100.0,
                    "range_bar_ratio_15": range_bars_15 / len(sub) * 100.0,
                    "vol_bar_ratio_15": vol_bars_15 / len(sub) * 100.0,
                    "vol_bar_ratio_30": vol_bars_30 / len(sub) * 100.0,
                })
            curr_state = st
            start_idx = i
            
    if curr_state is not None:
        sub = test_df.iloc[start_idx:]
        start_p = sub.iloc[0]["open"]
        end_p = sub.iloc[-1]["close"]
        closes = sub["close"].values
        rets = (closes - start_p) / start_p * 100.0
        episodes.append({
            "state": curr_state,
            "duration_hours": len(sub),
            "start_price": start_p,
            "end_price": end_p,
            "net_ret_pct": (end_p - start_p) / start_p * 100.0,
            "range_bar_ratio_15": np.sum(np.abs(rets) <= 1.5) / len(sub) * 100.0,
            "vol_bar_ratio_15": np.sum(np.abs(rets) > 1.5) / len(sub) * 100.0,
            "vol_bar_ratio_30": np.sum(np.abs(rets) >= 3.0) / len(sub) * 100.0,
        })
        
    return pd.DataFrame(episodes)


def run_experiment_64():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    df_proc = get_regime_series(df_ind, base_th=0.74, bear_th=0.80)
    ep_df = analyze_episodes_detailed(df_proc, state_col="two_state")

    print("=" * 105)
    print("      [실험 64] 국면 생애주기 내 [최종 종가 기준] & [봉(Bar) 점유율 기준] 정밀 실측")
    print("=" * 105)

    # 1. 방법 ① [최종 종가(End Close) 기준 성공률]
    print("\n[방법 ①: 국면 종료 시점의 '최종 종가(End Close)' 기준 성공 적중률]")
    print(" * 🟡 RANGE 성공    : 에피소드 마지막 종가가 |수익률| <= X% 박스 내 마감")
    print(" * ⚡ VOLATILE 성공 : 에피소드 마지막 종가가 |수익률| >= X% 기준선 밖에서 추세 마감")
    print("-" * 105)
    print(f"{'국면 상태':<20} | {'±1.0% 기준 마감률':<24} | {'±1.5% 기준 마감률(⭐)':<26} | {'±2.0% 기준 마감률':<24} | {'2배(±3%) 대형 마감':<22}")
    print("-" * 105)

    for state in ["RANGE", "VOLATILE"]:
        sub = ep_df[ep_df["state"] == state]
        res = []
        for th in [1.0, 1.5, 2.0, 3.0]:
            if state == "RANGE":
                hr = (sub["net_ret_pct"].abs() <= th).mean() * 100.0
                res.append(f"박스 마감: {hr:5.1f}%")
            else: # VOLATILE
                hr = (sub["net_ret_pct"].abs() >= th).mean() * 100.0
                res.append(f"추세 마감: {hr:5.1f}%")
        state_label = "🟡 횡보 국면 (RANGE)" if state == "RANGE" else "⚡ 변동 국면 (VOLATILE)"
        print(f"{state_label:<20} | {res[0]:<24} | {res[1]:<26} | {res[2]:<24} | {res[3]:<22}")
    print("-" * 105)

    # 2. 방법 ② [봉(Bar) 체류 시간 점유율 기준]
    print("\n[방법 ②: 에피소드 지속 시간 동안의 '봉(Bar) 체류 점유율' 실측]")
    print("-" * 105)
    r_sub = ep_df[ep_df["state"] == "RANGE"]
    v_sub = ep_df[ep_df["state"] == "VOLATILE"]

    mean_r_stay = r_sub["range_bar_ratio_15"].mean()
    perfect_r_stay = (r_sub["range_bar_ratio_15"] == 100.0).mean() * 100.0
    over80_r_stay = (r_sub["range_bar_ratio_15"] >= 80.0).mean() * 100.0

    print(f"🟡 횡보 국면 (RANGE - 총 {len(r_sub)}회) :")
    print(f" * 전체 지속 시간 중 '±1.5% 박스 안에 머문 시간 비율' (평균) : {mean_r_stay:5.1f}% (시간의 90% 이상 박스 체류!)")
    print(f" * 100% 단 1봉도 이탈 없이 완벽하게 박스를 지킨 에피소드 비율   : {perfect_r_stay:5.1f}% (총 {int(len(r_sub)*perfect_r_stay/100)}회)")
    print(f" * 시간의 80% 이상을 박스 안에서 보낸 에피소드 비율           : {over80_r_stay:5.1f}% (총 {int(len(r_sub)*over80_r_stay/100)}회)")

    print("-" * 105)
    mean_v_stay = v_sub["vol_bar_ratio_15"].mean()
    over50_v_stay = (v_sub["vol_bar_ratio_15"] >= 50.0).mean() * 100.0
    over1bar_v_stay = (v_sub["vol_bar_ratio_15"] > 0.0).mean() * 100.0
    mean_v_stay_30 = v_sub["vol_bar_ratio_30"].mean()

    print(f"⚡ 변동 국면 (VOLATILE - 총 {len(v_sub)}회) :")
    print(f" * 전체 지속 시간 중 '±1.5% 기준선 밖에서 체류한 시간 비율' (평균) : {mean_v_stay:5.1f}%")
    print(f" * 지속 시간의 '과반수(50% 이상)'를 기준선 밖에서 보낸 에피소드 비율   : {over50_v_stay:5.1f}% (총 {int(len(v_sub)*over50_v_stay/100)}회)")
    print(f" * 단 1봉이라도 1.5% 밖으로 나간 적이 있는 에피소드 비율             : {over1bar_v_stay:5.1f}% (총 {int(len(v_sub)*over1bar_v_stay/100)}회)")
    print(f" * 전체 지속 시간 중 '2배인 ±3.0% 이상 대형 파동 체류 비율' (평균)  : {mean_v_stay_30:5.1f}%")
    print("=" * 105)


if __name__ == "__main__":
    run_experiment_64()
