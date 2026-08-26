"""
flare.research.diagnose_2021_oos

2021년 OOS 손실의 원인을 데이터로 정밀 진단하는 스크립트
- 2021년 진입 당시의 실제 fundingRate 분포 확인
- 2021년 30일 백분위(RSI) 하위 5%가 실제 마이너스 펀딩비(숏 과열)였는지 확인
- 2022~2024년 진입 조건과의 펀딩비 절대치 차이 분석
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.data.features import generate_all_features


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    klines_2021 = data_dir / "BTCUSDT_5m_2021.csv"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
    df = pd.read_csv(klines_2021)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    df = pd.merge_asof(df, df_fr[["fundingTime", "fundingRate"]], left_on="datetime", right_on="fundingTime", direction="backward")
    df["fundingRate"] = df["fundingRate"].ffill().fillna(0.0001)
    
    df, _ = generate_all_features(df)
    eval_df = df.iloc[8640:].reset_index(drop=True)
    
    is_settle_bar = eval_df["datetime"].dt.minute == 0
    is_settle_hour = eval_df["datetime"].dt.hour.isin([0, 8, 16])
    sig_swing = is_settle_bar & is_settle_hour & (eval_df["feat_funding_rsi_30d"] <= 0.05)
    
    triggered_df = eval_df[sig_swing].copy()
    
    print("=" * 95)
    print("🔬 [진단] 2021년 불장에서 진입 신호가 떴을 때의 실제 펀딩비(fundingRate) 분포")
    print("=" * 95)
    print(f"[*] 총 신호 발생 횟수: {len(triggered_df)}회")
    print(f"[*] 진입 시점 실제 fundingRate 평균 : {triggered_df['fundingRate'].mean():.6f}")
    print(f"[*] 진입 시점 실제 fundingRate 중앙값: {triggered_df['fundingRate'].median():.6f}")
    print(f"[*] 진입 시점 실제 fundingRate 최댓값: {triggered_df['fundingRate'].max():.6f}")
    print(f"[*] 진입 시점 실제 fundingRate 최솟값: {triggered_df['fundingRate'].min():.6f}")
    print("-" * 95)
    
    neg_cnt = (triggered_df['fundingRate'] < 0).sum()
    pos_cnt = (triggered_df['fundingRate'] >= 0).sum()
    print(f"[*] 🔴 진입 시점 펀딩비가 '양수(롱 과열)'였던 가짜 신호 비율: {pos_cnt}회 ({pos_cnt/len(triggered_df)*100:.1f}%) 🚨")
    print(f"[*] 🟢 진입 시점 펀딩비가 '진짜 음수(숏 과열)'였던 신호 비율  : {neg_cnt}회 ({neg_cnt/len(triggered_df)*100:.1f}%)")
    print("=" * 95)


if __name__ == "__main__":
    main()
