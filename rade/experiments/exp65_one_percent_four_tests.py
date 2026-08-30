"""
[실험 65] 1.0% 기준 4대 테스트 정확도 집계
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
    
    curr2 = "RANGE"
    two_states = []
    for idx, row in df_raw.iterrows():
        p_r = row["p_range"]
        p_u = row["p_bull"]
        p_d = row["p_bear"]
        if p_d >= bear_th and p_d >= p_u and p_d >= p_r:
            curr2 = "VOLATILE"
        elif p_u >= base_th and p_u >= p_r and p_u >= p_d:
            curr2 = "VOLATILE"
        elif p_r >= base_th and p_r >= p_u and p_r >= p_d:
            curr2 = "RANGE"
        two_states.append(curr2)
        
    df_raw["two_state"] = two_states
    return df_raw


def run_experiment_65():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    df_proc = get_two_state_series(df_ind, base_th=0.74, bear_th=0.80)
    test_df = df_proc.iloc[720:].copy().reset_index(drop=True)

    episodes = []
    curr_state = None
    start_idx = 0
    
    for i, row in test_df.iterrows():
        st = row["two_state"]
        if st != curr_state:
            if curr_state is not None:
                sub = test_df.iloc[start_idx:i]
                start_p = sub.iloc[0]["open"]
                end_p = sub.iloc[-1]["close"]
                max_h = sub["high"].max()
                min_l = sub["low"].min()
                closes = sub["close"].values
                rets = (closes - start_p) / start_p * 100.0
                
                max_up = (max_h - start_p) / start_p * 100.0
                max_down = (min_l - start_p) / start_p * 100.0
                
                # 1.0% 기준 봉 비율
                range_bars_10 = np.sum(np.abs(rets) <= 1.0)
                vol_bars_10 = np.sum(np.abs(rets) > 1.0)
                
                episodes.append({
                    "state": curr_state,
                    "duration_hours": len(sub),
                    "max_up_pct": max_up,
                    "max_down_pct": max_down,
                    "abs_max_wave": max(max_up, abs(max_down)),
                    "net_ret_pct": (end_p - start_p) / start_p * 100.0,
                    "range_bar_ratio_10": range_bars_10 / len(sub) * 100.0,
                    "vol_bar_ratio_10": vol_bars_10 / len(sub) * 100.0,
                })
            curr_state = st
            start_idx = i
            
    ep_df = pd.DataFrame(episodes)
    r_sub = ep_df[ep_df["state"] == "RANGE"]
    v_sub = ep_df[ep_df["state"] == "VOLATILE"]

    print("=" * 105)
    print("      [1.0% 기준 4대 테스트 정확도 집계]")
    print("=" * 105)
    
    # 1. High/Low 1회 이상
    r_mfe_10 = ((r_sub["max_up_pct"] <= 1.0) & (r_sub["max_down_pct"] >= -1.0)).mean() * 100.0
    v_mfe_10 = (v_sub["abs_max_wave"] >= 1.0).mean() * 100.0
    print(f"1. High/Low 1.0% 돌파/갇힘 : RANGE={r_mfe_10:.1f}% ({int(len(r_sub)*r_mfe_10/100)}회) | VOLATILE={v_mfe_10:.1f}% ({int(len(v_sub)*v_mfe_10/100)}회)")

    # 2. 2배 (2.0%) 돌파
    v_2x_20 = (v_sub["abs_max_wave"] >= 2.0).mean() * 100.0
    print(f"2. 2배인 2.0% 대형 돌파   : RANGE={r_mfe_10:.1f}% | VOLATILE={v_2x_20:.1f}% ({int(len(v_sub)*v_2x_20/100)}회)")

    # 3. 최종 종가 마감
    r_close_10 = (r_sub["net_ret_pct"].abs() <= 1.0).mean() * 100.0
    v_close_10 = (v_sub["net_ret_pct"].abs() >= 1.0).mean() * 100.0
    print(f"3. 최종 종가 1.0% 마감     : RANGE={r_close_10:.1f}% ({int(len(r_sub)*r_close_10/100)}회) | VOLATILE={v_close_10:.1f}% ({int(len(v_sub)*v_close_10/100)}회)")

    # 4. 시간 체류 점유율 (과반수)
    r_stay_80 = (r_sub["range_bar_ratio_10"] >= 80.0).mean() * 100.0
    r_stay_100 = (r_sub["range_bar_ratio_10"] == 100.0).mean() * 100.0
    v_stay_50 = (v_sub["vol_bar_ratio_10"] >= 50.0).mean() * 100.0
    v_stay_any = (v_sub["vol_bar_ratio_10"] > 0.0).mean() * 100.0
    print(f"4. 봉 점유율 (1.0% 기준)   : RANGE(80%체류)={r_stay_80:.1f}% (완벽={r_stay_100:.1f}%) | VOLATILE(과반50%체류)={v_stay_50:.1f}% ({int(len(v_sub)*v_stay_50/100)}회, 1봉이탈={v_stay_any:.1f}%)")


if __name__ == "__main__":
    run_experiment_65()
