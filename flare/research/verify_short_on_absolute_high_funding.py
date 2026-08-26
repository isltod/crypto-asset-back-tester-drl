"""
flare.research.verify_short_on_absolute_high_funding

롱 과열(극단적 양수 펀딩비 절대값) 시 숏(Short) 진입 가설 정밀 재검증
- 조건: FR >= +0.03% (+0.0003), +0.05% (+0.0005), +0.10% (+0.0010)
- 24시간 내 최대 하락폭(Short MFE) vs 최대 상승폭(Short MAE) 비대칭성 분석
- 2021~2024년 전체 4개년 실전 백테스트 (SL -4%, No TP, 24h 숏)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.data.features import generate_all_features


def test_short_on_extreme_positive_funding():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    klines_4y = data_dir / "BTCUSDT_1h_4years_full.csv"
    
    df = pd.read_csv(klines_4y)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    df = pd.merge_asof(df, df_fr[["fundingTime", "fundingRate"]], left_on="datetime", right_on="fundingTime", direction="backward")
    df["fundingRate"] = df["fundingRate"].ffill().fillna(0.0001)
    
    # 8시간 정산 시점 필터링
    is_settle = df["datetime"].dt.hour.isin([0, 8, 16])
    settle_df = df[is_settle].copy().reset_index(drop=True)
    
    thresholds = [
        ("FR >= +0.02% (2배 과열)", 0.0002),
        ("FR >= +0.03% (3배 과열)", 0.0003),
        ("FR >= +0.05% (5배 과열)", 0.0005),
        ("FR >= +0.08% (8배 과열)", 0.0008),
        ("FR >= +0.10% (10배 폭탄)", 0.0010)
    ]
    
    print("=" * 105)
    print("🔬 [재검증] 롱 극단 과열(양수 펀딩비 절대값) 발생 시 24시간 숏(Short) 진입 통계 (2021~2024, 4개년)")
    print("=" * 105)
    header = "{:<24} | {:<6} | {:<16} | {:<16} | {:<12} | {:<10}"
    row = "{:<24} | {:>6} | {:>15.2f}% | {:>15.2f}% | {:>11.2f}배 | {:>9.1f}%"
    print(header.format("펀딩비 절대값 수준", "발생건수", "24h 숏 MFE(하락폭)", "24h 숏 MAE(역풍폭)", "MFE/MAE 비", "숏 승률(24h)"))
    print("-" * 105)
    
    for label, th in thresholds:
        target_indices = settle_df[settle_df["fundingRate"] >= th].index
        
        mfes = []
        maes = []
        win_cnt = 0
        
        for idx in target_indices:
            # 24시간 = 1h 기준 24개 봉
            if idx + 24 >= len(settle_df):
                continue
            entry_p = settle_df.loc[idx, "close"]
            future_window = df[(df["datetime"] > settle_df.loc[idx, "datetime"]) & (df["datetime"] <= settle_df.loc[idx, "datetime"] + pd.Timedelta(hours=24))]
            if len(future_window) == 0:
                continue
                
            min_low = future_window["low"].min()
            max_high = future_window["high"].max()
            exit_close = future_window["close"].iloc[-1]
            
            # 숏 관점: 가격 하락이 MFE(이익), 가격 상승이 MAE(손실)
            short_mfe = (entry_p - min_low) / entry_p * 100.0
            short_mae = (max_high - entry_p) / entry_p * 100.0
            short_ret = (entry_p - exit_close) / entry_p * 100.0
            
            mfes.append(short_mfe)
            maes.append(short_mae)
            if short_ret > 0:
                win_cnt += 1
                
        if len(mfes) > 0:
            avg_mfe = np.mean(mfes)
            avg_mae = np.mean(maes)
            ratio = avg_mfe / avg_mae if avg_mae > 0 else 0
            wr = win_cnt / len(mfes) * 100.0
            print(row.format(label, f"{len(mfes)}회", avg_mfe, avg_mae, ratio, wr))


if __name__ == "__main__":
    test_short_on_extreme_positive_funding()
