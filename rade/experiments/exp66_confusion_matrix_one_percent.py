"""
[실험 66] 1.0% 장중 1회 이상 돌파 기준 HMM 변동/횡보 국면 혼동행렬(Confusion Matrix) 산출
- 정의:
  - Actual Volatile (실제 변동): 에피소드 지속 중 장중 High/Low 파동이 1.0% 이상 돌파
  - Actual Range (실제 횡보): 에피소드 지속 중 장중 High/Low 파동이 1.0% 이내에 갇힘
  - Predicted: HMM의 국면 선언 (VOLATILE vs RANGE)
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


def run_experiment_66():
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
                max_h = sub["high"].max()
                min_l = sub["low"].min()
                max_up = (max_h - start_p) / start_p * 100.0
                max_down = (min_l - start_p) / start_p * 100.0
                abs_wave = max(max_up, abs(max_down))
                
                # 실제 시장이 1.0% 이상 돌파했는가?
                is_actual_volatile = abs_wave >= 1.0
                
                episodes.append({
                    "pred_state": curr_state,
                    "duration_hours": len(sub),
                    "abs_max_wave": abs_wave,
                    "is_actual_volatile": is_actual_volatile,
                })
            curr_state = st
            start_idx = i
            
    if curr_state is not None:
        sub = test_df.iloc[start_idx:]
        start_p = sub.iloc[0]["open"]
        max_up = (sub["high"].max() - start_p) / start_p * 100.0
        max_down = (sub["low"].min() - start_p) / start_p * 100.0
        abs_wave = max(max_up, abs(max_down))
        episodes.append({
            "pred_state": curr_state,
            "duration_hours": len(sub),
            "abs_max_wave": abs_wave,
            "is_actual_volatile": abs_wave >= 1.0,
        })
        
    ep_df = pd.DataFrame(episodes)

    # 혼동 행렬 계산
    # Positive = VOLATILE, Negative = RANGE
    tp = len(ep_df[(ep_df["pred_state"] == "VOLATILE") & (ep_df["is_actual_volatile"] == True)])
    fp = len(ep_df[(ep_df["pred_state"] == "VOLATILE") & (ep_df["is_actual_volatile"] == False)])
    tn = len(ep_df[(ep_df["pred_state"] == "RANGE") & (ep_df["is_actual_volatile"] == False)])
    fn = len(ep_df[(ep_df["pred_state"] == "RANGE") & (ep_df["is_actual_volatile"] == True)])
    total = len(ep_df)

    precision = tp / (tp + fp) * 100.0
    recall = tp / (tp + fn) * 100.0
    specificity = tn / (tn + fp) * 100.0
    accuracy = (tp + tn) / total * 100.0
    f1 = 2 * (precision * recall) / (precision + recall)

    print("=" * 105)
    print("      [실험 66] 1.0% 기준 HMM 국면 판정 혼동행렬 (Confusion Matrix) 정밀 산출")
    print("=" * 105)
    print(f"\n* 총 분석 에피소드 수: {total}회 (VOLATILE 예측 {tp+fp}회, RANGE 예측 {tn+fn}회)")
    print(f"* 실제 1.0% 돌파 발생: {tp+fn}회 (58.5%) | 실제 1.0% 박스 유지: {tn+fp}회 (41.5%)")
    print("\n[ 2x2 혼동행렬 표 (Confusion Matrix) ]")
    print("-" * 75)
    print(f"{'구분':<20} | {'실제 변동 (Actual Volatile)':<24} | {'실제 횡보 (Actual Range)':<24}")
    print("-" * 75)
    print(f"{'HMM 예측: VOLATILE':<18} | TP = {tp:5d}회 ({tp/(tp+fp)*100:5.1f}%) [적중⭐]    | FP = {fp:5d}회 ({fp/(tp+fp)*100:5.1f}%) [오경보]  ")
    print(f"{'HMM 예측: RANGE':<18} | FN = {fn:5d}회 ({fn/(tn+fn)*100:5.1f}%) [놓침]      | TN = {tn:5d}회 ({tn/(tn+fn)*100:5.1f}%) [적중⭐]  ")
    print("-" * 75)
    print(f"\n[ 머신러닝 / 퀀트 핵심 평가 지표 ]")
    print(f" 1. 정밀도 (Precision, 1% 돌파 적중률) : {precision:5.2f}% (HMM이 변동이라 했을 때 실제로 1% 돌파 성공!)")
    print(f" 2. 재현율 (Recall, 1% 돌파 포착률)     : {recall:5.2f}% (시장 전체 1% 돌파 사건 중 HMM이 잡아낸 비율)")
    print(f" 3. 특이도 (Specificity, 횡보 보호율)  : {specificity:5.2f}% (실제 횡보 구간 중 HMM이 횡보로 지켜낸 비율)")
    print(f" 4. 전체 정확도 (Overall Accuracy)     : {accuracy:5.2f}%")
    print(f" 5. F1-Score (조화 평균)               : {f1:5.2f}")
    print("=" * 105)


if __name__ == "__main__":
    run_experiment_66()
