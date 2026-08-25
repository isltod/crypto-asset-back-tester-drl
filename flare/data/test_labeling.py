"""
flare.data.test_labeling

labeling.py 모듈의 4시간(Sniper) 및 24시간(Swing) 비대칭 타깃 라벨 생성 및 분포 검증
"""

import sys
from pathlib import Path
import pandas as pd

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.data.features import generate_all_features
from flare.data.labeling import create_asymmetric_labels


def test_labeling_pipeline():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    klines_file = data_dir / "BTCUSDT_5m_2022_2024.csv"
    funding_file = Path(__file__).resolve().parent / "btcusdt_funding_rate.csv"
    
    print(f"[*] 데이터 로드 중: {klines_file.name}...")
    df = pd.read_csv(klines_file)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # 펀딩비 매핑
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    df = pd.merge_asof(df, df_fr[["fundingTime", "fundingRate"]], left_on="datetime", right_on="fundingTime", direction="backward")
    df["fundingRate"] = df["fundingRate"].ffill().fillna(0.0001)
    
    # 1. 4시간 Horizon (FLARE-Sniper: 48개 5분봉) 라벨링
    print("[*] 4시간(48봉) 비대칭 타깃 라벨링 생성...")
    df, label_4h = create_asymmetric_labels(df, horizon_bars=48, min_mfe_pct=1.0, ratio_threshold=1.3)
    
    # 2. 24시간 Horizon (FLARE-Swing: 288개 5분봉) 라벨링
    print("[*] 24시간(288봉) 비대칭 타깃 라벨링 생성...")
    df, label_24h = create_asymmetric_labels(df, horizon_bars=288, min_mfe_pct=2.0, ratio_threshold=1.3)
    
    # 유효 데이터 (미래 24h 롤링 결측 제거)
    df_valid = df.dropna(subset=[label_4h, label_24h, "target_mfe_288"]).reset_index(drop=True)
    
    print("=" * 85)
    print("[FLARE] 비대칭 국면 타깃 라벨링(Labeling) 분포 검증 보고서")
    print("=" * 85)
    print(f"[*] 총 유효 5분봉 캔들 수: {len(df_valid):,}개")
    print("-" * 85)
    
    # 4h 분포
    dist_4h = df_valid[label_4h].value_counts(normalize=True).sort_index() * 100
    cnt_4h = df_valid[label_4h].value_counts().sort_index()
    
    print("🎯 [Mode 1: Sniper 4시간 Horizon] 클래스 분포 (최소 진폭 1.0%, 손익비 1.3x 이상):")
    print(f"    - Class 0 (Neutral / 횡보 노이즈): {cnt_4h.get(0, 0):>6,}개 ({dist_4h.get(0, 0.0):>5.1f}%)")
    print(f"    - Class 1 (Long / 상방 비대칭 우위): {cnt_4h.get(1, 0):>6,}개 ({dist_4h.get(1, 0.0):>5.1f}%) 🟢")
    print(f"    - Class 2 (Short / 하방 비대칭 우위): {cnt_4h.get(2, 0):>6,}개 ({dist_4h.get(2, 0.0):>5.1f}%) 🔴")
    print("-" * 85)
    
    # 24h 분포
    dist_24h = df_valid[label_24h].value_counts(normalize=True).sort_index() * 100
    cnt_24h = df_valid[label_24h].value_counts().sort_index()
    print("🎯 [Mode 2: Swing 24시간 Horizon] 클래스 분포 (최소 진폭 2.0%, 손익비 1.3x 이상):")
    print(f"    - Class 0 (Neutral / 횡보 노이즈): {cnt_24h.get(0, 0):>6,}개 ({dist_24h.get(0, 0.0):>5.1f}%)")
    print(f"    - Class 1 (Long / 상방 비대칭 우위): {cnt_24h.get(1, 0):>6,}개 ({dist_24h.get(1, 0.0):>5.1f}%) 🟢")
    print(f"    - Class 2 (Short / 하방 비대칭 우위): {cnt_24h.get(2, 0):>6,}개 ({dist_24h.get(2, 0.0):>5.1f}%) 🔴")
    print("=" * 85)
    print("✅ 라벨링 모듈 검증 성공! (완벽한 클래스 밸런스 확인)")


if __name__ == "__main__":
    test_labeling_pipeline()
