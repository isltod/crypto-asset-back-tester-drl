"""
[실험 67] 변동 국면(BULL vs BEAR) 내 정방향 적중 vs 역방향 휩소/오판 정밀 분석
- 목적:
  1. BULL 예측 시 하락(-1.0%, -1.5%)이 나타난 비율 (역방향 롱 손절 위험)
  2. BEAR 예측 시 상승(+1.0%, +1.5%)이 나타난 비율 (역방향 숏 스퀴즈 위험)
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


def get_three_state_series(df_ind: pd.DataFrame, base_th: float = 0.74, bear_th: float = 0.80) -> pd.DataFrame:
    reg_raw = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.30, cooldown_bars=0)
    df_raw = reg_raw.calculate_regime_probabilities(df_ind)
    
    curr3 = "RANGE"
    three_states = []
    for idx, row in df_raw.iterrows():
        p_r = row["p_range"]
        p_u = row["p_bull"]
        p_d = row["p_bear"]
        if p_d >= bear_th and p_d >= p_u and p_d >= p_r:
            curr3 = "BEAR_PANIC"
        elif p_u >= base_th and p_u >= p_r and p_u >= p_d:
            curr3 = "BULL_TREND"
        elif p_r >= base_th and p_r >= p_u and p_r >= p_d:
            curr3 = "RANGE"
        three_states.append(curr3)
        
    df_raw["three_state"] = three_states
    return df_raw


def run_experiment_67():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    df_proc = get_three_state_series(df_ind, base_th=0.74, bear_th=0.80)
    test_df = df_proc.iloc[720:].copy().reset_index(drop=True)

    episodes = []
    curr_state = None
    start_idx = 0
    
    for i, row in test_df.iterrows():
        st = row["three_state"]
        if st != curr_state:
            if curr_state is not None:
                sub = test_df.iloc[start_idx:i]
                start_p = sub.iloc[0]["open"]
                max_h = sub["high"].max()
                min_l = sub["low"].min()
                max_up = (max_h - start_p) / start_p * 100.0
                max_down = (min_l - start_p) / start_p * 100.0
                episodes.append({
                    "state": curr_state,
                    "duration_hours": len(sub),
                    "max_up_pct": max_up,
                    "max_down_pct": max_down,
                    "net_ret_pct": (sub.iloc[-1]["close"] - start_p) / start_p * 100.0,
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
            "max_up_pct": max_up,
            "max_down_pct": max_down,
            "net_ret_pct": (sub.iloc[-1]["close"] - start_p) / start_p * 100.0,
        })
        
    ep_df = pd.DataFrame(episodes)
    bull_df = ep_df[ep_df["state"] == "BULL_TREND"]
    bear_df = ep_df[ep_df["state"] == "BEAR_PANIC"]

    print("=" * 105)
    print("      [실험 67] 변동 국면(BULL vs BEAR) 내 정방향 적중 vs 역방향 휩소/오판 정밀 분석")
    print("=" * 105)
    print(f"* 4개년 총 분석 대상: 🟢 BULL 에피소드 {len(bull_df)}회 vs 🔴 BEAR 에피소드 {len(bear_df)}회\n")

    for th in [1.0, 1.5, 2.0]:
        print(f"[{th:.1f}% 기준선 정밀 파동 분석]")
        print("-" * 105)
        
        # 1. BULL 국면 분석
        # 정방향 상승만: max_up >= th and max_down > -th
        bull_pure_win = bull_df[(bull_df["max_up_pct"] >= th) & (bull_df["max_down_pct"] > -th)]
        # 양방향 휩소: max_up >= th and max_down <= -th
        bull_both = bull_df[(bull_df["max_up_pct"] >= th) & (bull_df["max_down_pct"] <= -th)]
        # 역방향 하락만(오판): max_up < th and max_down <= -th  (★ 사용자 관심사!)
        bull_reverse_loss = bull_df[(bull_df["max_up_pct"] < th) & (bull_df["max_down_pct"] <= -th)]
        # 미달 횡보: max_up < th and max_down > -th
        bull_no_wave = bull_df[(bull_df["max_up_pct"] < th) & (bull_df["max_down_pct"] > -th)]

        total_bull = len(bull_df)
        print(f"🟢 [HMM 예측: BULL_TREND (상승 예측 - 총 {total_bull}회)]")
        print(f" * ① 정방향 상승 성공 (상승만 발생)     : {len(bull_pure_win):4d}회 ({len(bull_pure_win)/total_bull*100:5.1f}%)")
        print(f" * ② 양방향 휩소 (상승+하락 둘 다 발생) : {len(bull_both):4d}회 ({len(bull_both)/total_bull*100:5.1f}%)")
        print(f" * ③ 역방향 오판 (★ 하락 폭락만 발생)   : {len(bull_reverse_loss):4d}회 ({len(bull_reverse_loss)/total_bull*100:5.1f}%)")
        print(f" * ④ 미달 횡보 (둘 다 기준선 미달)     : {len(bull_no_wave):4d}회 ({len(bull_no_wave)/total_bull*100:5.1f}%)")
        print(f" 👉 【BULL 중 역방향 하락(-{th}%)이 포함된 총 비율 (②+③)】: {(len(bull_both)+len(bull_reverse_loss))/total_bull*100:5.1f}% ({len(bull_both)+len(bull_reverse_loss)}회)")

        print("-" * 105)
        # 2. BEAR 국면 분석
        # 정방향 하락만: max_down <= -th and max_up < th
        bear_pure_win = bear_df[(bear_df["max_down_pct"] <= -th) & (bear_df["max_up_pct"] < th)]
        # 양방향 휩소: max_down <= -th and max_up >= th
        bear_both = bear_df[(bear_df["max_down_pct"] <= -th) & (bear_df["max_up_pct"] >= th)]
        # 역방향 상승만(오판 숏스퀴즈): max_down > -th and max_up >= th  (★ 사용자 관심사!)
        bear_reverse_loss = bear_df[(bear_df["max_down_pct"] > -th) & (bear_df["max_up_pct"] >= th)]
        # 미달 횡보: max_down > -th and max_up < th
        bear_no_wave = bear_df[(bear_df["max_down_pct"] > -th) & (bear_df["max_up_pct"] < th)]

        total_bear = len(bear_df)
        print(f"🔴 [HMM 예측: BEAR_PANIC (하락 예측 - 총 {total_bear}회)]")
        print(f" * ① 정방향 폭락 성공 (하락만 발생)     : {len(bear_pure_win):4d}회 ({len(bear_pure_win)/total_bear*100:5.1f}%)")
        print(f" * ② 양방향 휩소 (하락+상승 둘 다 발생) : {len(bear_both):4d}회 ({len(bear_both)/total_bear*100:5.1f}%)")
        print(f" * ③ 역방향 오판 (★ 상승 급등만 발생)   : {len(bear_reverse_loss):4d}회 ({len(bear_reverse_loss)/total_bear*100:5.1f}%)")
        print(f" * ④ 미달 횡보 (둘 다 기준선 미달)     : {len(bear_no_wave):4d}회 ({len(bear_no_wave)/total_bear*100:5.1f}%)")
        print(f" 👉 【BEAR 중 역방향 상승(+{th}%)이 포함된 총 비율 (②+③)】: {(len(bear_both)+len(bear_reverse_loss))/total_bear*100:5.1f}% ({len(bear_both)+len(bear_reverse_loss)}회)")
        print("=" * 105 + "\n")


if __name__ == "__main__":
    run_experiment_67()
