"""
[실험 62] 횡보 박스(1x) 대비 추세 2배(2x) 돌파 기준 국면 적중률 정밀 측정
- 사용자 요청 조건:
  - Level 1: RANGE [±1.0% 박스 유지] vs BULL [+2.0% 이상 돌파] / BEAR [-2.0% 이하 폭락]
  - Level 2: RANGE [±1.5% 박스 유지] vs BULL [+3.0% 이상 돌파] / BEAR [-3.0% 이하 폭락] (⭐ 표준)
  - Level 3: RANGE [±2.0% 박스 유지] vs BULL [+4.0% 이상 돌파] / BEAR [-4.0% 이하 폭락] (🚀 대형)
- 생애주기 내 단 한 번이라도 해당 2배 목표 파동을 찔렀는가(High/Low MFE)로 성공 판정
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
    test_df = df.iloc[720:].copy().reset_index(drop=True)
    episodes = []
    curr_state = None
    start_idx = 0
    
    for i, row in test_df.iterrows():
        st = row["regime_state"]
        if st != curr_state:
            if curr_state is not None:
                sub = test_df.iloc[start_idx:i]
                start_p = sub.iloc[0]["open"]
                max_h = sub["high"].max()
                min_l = sub["low"].min()
                end_p = sub.iloc[-1]["close"]
                episodes.append({
                    "state": curr_state,
                    "duration_hours": len(sub),
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


def run_experiment_62():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    df_asym = get_regime_series(df_ind, base_th=0.74, bear_th=0.80)
    ep_df = extract_episodes(df_asym)

    print("=" * 105)
    print("      [실험 62] 횡보 박스(1x) 대비 추세 2배(2x) 돌파 기준 국면 적중률 정밀 측정")
    print("=" * 105)

    levels = [
        {"name": "Level 1 (기본)", "range_th": 1.0, "trend_th": 2.0},
        {"name": "Level 2 (표준 ⭐)", "range_th": 1.5, "trend_th": 3.0},
        {"name": "Level 3 (대형 🚀)", "range_th": 2.0, "trend_th": 4.0},
        {"name": "Level 4 (초대형 🔥)", "range_th": 2.5, "trend_th": 5.0},
    ]

    print("\n[1. 국면별 비대칭 성공 적중률 (Hit Rate, %)]")
    print("-" * 105)
    print(f"{'국면 상태':<18} | {'Level 1 [횡보1% vs 추세2%]':<26} | {'Level 2 [횡보1.5% vs 추세3% ⭐]':<28} | {'Level 3 [횡보2% vs 추세4% 🚀]':<26}")
    print("-" * 105)

    for state in ["BULL_TREND", "BEAR_PANIC", "RANGE"]:
        sub = ep_df[ep_df["state"] == state]
        res_strs = []
        for lv in levels[:3]:
            r_th = lv["range_th"]
            t_th = lv["trend_th"]
            if state == "BULL_TREND":
                hr = (sub["max_up_pct"] >= t_th).mean() * 100.0
                label = f"+{t_th:.1f}% 돌파: {hr:5.1f}%"
            elif state == "BEAR_PANIC":
                hr = (sub["max_down_pct"] <= -t_th).mean() * 100.0
                label = f"-{t_th:.1f}% 폭락: {hr:5.1f}%"
            else: # RANGE
                hr = ((sub["max_up_pct"] <= r_th) & (sub["max_down_pct"] >= -r_th)).mean() * 100.0
                label = f"±{r_th:.1f}% 갇힘: {hr:5.1f}%"
            res_strs.append(label)
        
        state_label = "🟢 BULL_TREND" if state == "BULL_TREND" else ("🔴 BEAR_PANIC" if state == "BEAR_PANIC" else "🟡 RANGE")
        print(f"{state_label:<18} | {res_strs[0]:<26} | {res_strs[1]:<28} | {res_strs[2]:<26}")
    print("-" * 105)

    # 2. 2배 돌파에 성공한 에피소드들의 특성 분석 (Level 2: 3% 기준)
    print("\n[2. Level 2 (3% 돌파) 성공 에피소드 vs 실패 에피소드 정밀 분석]")
    print("-" * 105)
    
    bull_all = ep_df[ep_df["state"] == "BULL_TREND"]
    bull_succ = bull_all[bull_all["max_up_pct"] >= 3.0]
    bull_fail = bull_all[bull_all["max_up_pct"] < 3.0]
    print(f" * 🟢 BULL 대형 성공 에피소드 ({len(bull_succ)}회, {len(bull_succ)/len(bull_all)*100:.1f}%) : 평균 지속 {bull_succ['duration_hours'].mean():.1f}시간 | 평균 최대 상승 {bull_succ['max_up_pct'].mean():+6.2f}% | 최종 마감 손익 {bull_succ['net_ret_pct'].mean():+6.2f}%")
    print(f" * 🟢 BULL 미달/단기 에피소드 ({len(bull_fail)}회, {len(bull_fail)/len(bull_all)*100:.1f}%) : 평균 지속 {bull_fail['duration_hours'].mean():.1f}시간 | 평균 최대 상승 {bull_fail['max_up_pct'].mean():+6.2f}% | 최종 마감 손익 {bull_fail['net_ret_pct'].mean():+6.2f}%")
    
    print("-" * 105)
    bear_all = ep_df[ep_df["state"] == "BEAR_PANIC"]
    bear_succ = bear_all[bear_all["max_down_pct"] <= -3.0]
    bear_fail = bear_all[bear_all["max_down_pct"] > -3.0]
    print(f" * 🔴 BEAR 대형 폭락 에피소드 ({len(bear_succ)}회, {len(bear_succ)/len(bear_all)*100:.1f}%) : 평균 지속 {bear_succ['duration_hours'].mean():.1f}시간 | 평균 최대 하락 {bear_succ['max_down_pct'].mean():+6.2f}% | 최종 마감 손익 {bear_succ['net_ret_pct'].mean():+6.2f}%")
    print(f" * 🔴 BEAR 미달/단기 에피소드 ({len(bear_fail)}회, {len(bear_fail)/len(bear_all)*100:.1f}%) : 평균 지속 {bear_fail['duration_hours'].mean():.1f}시간 | 평균 최대 하락 {bear_fail['max_down_pct'].mean():+6.2f}% | 최종 마감 손익 {bear_fail['net_ret_pct'].mean():+6.2f}%")
    print("-" * 105)


if __name__ == "__main__":
    run_experiment_62()
