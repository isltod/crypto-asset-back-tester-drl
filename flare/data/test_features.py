"""
flare.data.test_features

features.py 모듈의 피처 생성 정상 동작 및 NaN/결측치 여부를 검증하는 테스트 스크립트
"""

import sys
from pathlib import Path
import pandas as pd

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.data.features import generate_all_features


def test_features_pipeline():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    klines_file = data_dir / "BTCUSDT_5m_2022_2024.csv"
    funding_file = Path(__file__).resolve().parent / "btcusdt_funding_rate.csv"
    
    print(f"[*] 5분봉 데이터 로드 중: {klines_file.name}...")
    df = pd.read_csv(klines_file)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # 펀딩비 매핑
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    
    df = pd.merge_asof(
        df,
        df_fr[["fundingTime", "fundingRate"]],
        left_on="datetime",
        right_on="fundingTime",
        direction="backward"
    )
    df["fundingRate"] = df["fundingRate"].ffill().fillna(0.0001)
    
    print("[*] 피처 엔지니어링 파이프라인 가동...")
    df_feat, feature_cols = generate_all_features(df)
    
    # 웜업 구간 제외 (30일 = 8640개 봉)
    df_valid = df_feat.iloc[8640:].reset_index(drop=True)
    
    print("=" * 80)
    print(f"[+] 총 생성된 피처 개수: {len(feature_cols)}개")
    print(f"[+] 유효 데이터 행 수: {len(df_valid):,}개 (전체 {len(df_feat):,}개 중)")
    print("-" * 80)
    print("📋 생성된 피처 목록 및 통계 요약:")
    
    summary = df_valid[feature_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    print(summary.to_string())
    print("=" * 80)
    
    # 결측치 체크
    null_counts = df_valid[feature_cols].isnull().sum()
    if null_counts.sum() == 0:
        print("✅ 모든 피처에 결측치(NaN) 없음! 파이프라인 검증 통과!")
    else:
        print("⚠️ 결측치 발생 피처:", null_counts[null_counts > 0])


if __name__ == "__main__":
    test_features_pipeline()
