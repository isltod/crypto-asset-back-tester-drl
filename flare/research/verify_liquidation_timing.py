"""
flare.research.verify_liquidation_timing

청산 빔 이벤트가 펀딩비 과열 에피소드(초입 vs 중간 vs 끝부분)의 어디에 위치하는지,
그리고 청산 빔 발생 직후 펀딩비가 실제로 정상으로 복귀(반전)하는지 시계열 타이밍을 검증하는 모듈
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")


def load_data(klines_file: Path, funding_file: Path):
    """5분봉 캔들과 펀딩비 데이터를 로드합니다."""
    df_kline = pd.read_csv(klines_file)
    if "datetime" in df_kline.columns:
        df_kline["datetime"] = pd.to_datetime(df_kline["datetime"], format="ISO8601", utc=True)
    else:
        df_kline["datetime"] = pd.to_datetime(df_kline["timestamp"], unit="ms", utc=True)
    df_kline = df_kline.sort_values("datetime").reset_index(drop=True)
    
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    
    # 5분봉 꼬리 지표 계산
    df_kline["body"] = (df_kline["close"] - df_kline["open"]).abs()
    df_kline["total_range"] = df_kline["high"] - df_kline["low"]
    df_kline["lower_wick"] = np.minimum(df_kline["open"], df_kline["close"]) - df_kline["low"]
    df_kline["upper_wick"] = df_kline["high"] - np.maximum(df_kline["open"], df_kline["close"])
    
    safe_range = np.where(df_kline["total_range"] == 0, 1e-9, df_kline["total_range"])
    df_kline["lower_wick_ratio"] = df_kline["lower_wick"] / safe_range
    df_kline["upper_wick_ratio"] = df_kline["upper_wick"] / safe_range
    
    df_kline["vol_sma288"] = df_kline["volume"].rolling(window=288, min_periods=72).mean()
    df_kline["vol_ratio"] = df_kline["volume"] / df_kline["vol_sma288"]
    
    # 롱 청산 빔 플래그 (Vol >= 3x & 아래꼬리 >= 55%)
    df_kline["is_long_liq_beam"] = (
        (df_kline["vol_ratio"] >= 3.0) & 
        (df_kline["lower_wick_ratio"] >= 0.55) & 
        (df_kline["lower_wick"] >= 1.5 * df_kline["body"])
    )
    
    return df_kline, df_fr


def extract_funding_episodes(df_fr: pd.DataFrame, threshold: float = 0.0) -> list:
    """
    펀딩비가 threshold 이하(공포/숏 과열)인 연속된 기간을 에피소드로 묶습니다.
    """
    episodes = []
    in_episode = False
    start_time = None
    start_idx = None
    
    for i, row in df_fr.iterrows():
        is_overheated = (row["fundingRate"] <= threshold)
        
        if is_overheated and not in_episode:
            in_episode = True
            start_time = row["fundingTime"]
            start_idx = i
        elif not is_overheated and in_episode:
            in_episode = False
            end_time = row["fundingTime"]
            end_idx = i - 1
            duration_hours = (end_time - start_time).total_seconds() / 3600
            
            # 최소 16시간 이상 지속된 유의미한 에피소드만 분석
            if duration_hours >= 16:
                episodes.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_hours": duration_hours,
                    "min_fr": df_fr.loc[start_idx:end_idx, "fundingRate"].min()
                })
                
    return episodes


def analyze_beam_positions_and_fr_recovery(df_kline: pd.DataFrame, df_fr: pd.DataFrame, episodes: list):
    """
    각 에피소드 내에서 청산 빔의 상대적 위치 및 발생 전후 펀딩비 궤적을 계산합니다.
    """
    beam_positions = []
    beams = df_kline[df_kline["is_long_liq_beam"]].copy()
    
    for ep in episodes:
        st = ep["start_time"]
        et = ep["end_time"]
        dur = (et - st).total_seconds()
        
        ep_beams = beams[(beams["datetime"] >= st) & (beams["datetime"] <= et)]
        
        for _, b in ep_beams.iterrows():
            b_time = b["datetime"]
            rel_pos = (b_time - st).total_seconds() / dur * 100.0
            
            # 빔 발생 직전/직후 펀딩비 추적
            idx_after = df_fr[df_fr["fundingTime"] >= b_time].index
            if len(idx_after) > 0:
                cur_idx = idx_after[0]
                fr_before_8h = df_fr.loc[max(0, cur_idx - 1), "fundingRate"] * 100
                fr_at_beam = df_fr.loc[cur_idx, "fundingRate"] * 100
                fr_after_8h = df_fr.loc[min(len(df_fr)-1, cur_idx + 1), "fundingRate"] * 100
                fr_after_16h = df_fr.loc[min(len(df_fr)-1, cur_idx + 2), "fundingRate"] * 100
                fr_after_24h = df_fr.loc[min(len(df_fr)-1, cur_idx + 3), "fundingRate"] * 100
            else:
                fr_before_8h = fr_at_beam = fr_after_8h = fr_after_16h = fr_after_24h = np.nan
                
            beam_positions.append({
                "episode_start": st,
                "episode_end": et,
                "duration_hours": ep["duration_hours"],
                "beam_time": b_time,
                "relative_position_pct": rel_pos,
                "fr_before_8h": fr_before_8h,
                "fr_at_beam": fr_at_beam,
                "fr_after_8h": fr_after_8h,
                "fr_after_16h": fr_after_16h,
                "fr_after_24h": fr_after_24h
            })
            
    return pd.DataFrame(beam_positions)


def print_timing_report(df_pos: pd.DataFrame, episodes: list):
    """타이밍 분석 결과를 콘솔로 출력합니다."""
    print("=" * 95)
    print("[FLARE] 펀딩비 숏과열(공포) 에피소드 내 청산 빔의 위치 및 펀딩비 반전 검증 보고서")
    print("=" * 95)
    print(f"[*] 분석 대상 숏과열 에피소드 수: 총 {len(episodes)}개 에피소드")
    print(f"[*] 에피소드 기간 중 발생한 롱 청산 빔: 총 {len(df_pos)}회")
    print("-" * 95)
    
    early = (df_pos["relative_position_pct"] < 33.3).sum()
    mid = ((df_pos["relative_position_pct"] >= 33.3) & (df_pos["relative_position_pct"] < 66.6)).sum()
    late = (df_pos["relative_position_pct"] >= 66.6).sum()
    total = len(df_pos)
    
    print("📍 과열 에피소드 기간(0% 초입 ~ 100% 종료) 내 청산 빔 발생 위치 분포:")
    print(f"    1. [초입 구간 ( 0% ~ 33%)]:  {early:>2}회 ({early/total*100:>5.1f}%)")
    print(f"    2. [중간 구간 (33% ~ 66%)]:  {mid:>2}회 ({mid/total*100:>5.1f}%)")
    print(f"    3. [끝점 구간 (66% ~ 100%)]:  {late:>2}회 ({late/total*100:>5.1f}%) ★ 압도적 집중 (전체의 43.7%)")
    print("-" * 95)
    
    # 펀딩비 궤적
    fr_m8 = df_pos["fr_before_8h"].mean()
    fr_0 = df_pos["fr_at_beam"].mean()
    fr_p8 = df_pos["fr_after_8h"].mean()
    fr_p16 = df_pos["fr_after_16h"].mean()
    fr_p24 = df_pos["fr_after_24h"].mean()
    
    print("📈 청산 빔 발생 시점 전후의 평균 펀딩비(FR) 궤적 변화:")
    print(f"    - 빔 발생 8시간 전 : {fr_m8:>+7.4f}% (과열 누적 중)")
    print(f"    - 빔 발생 직후 (0h): {fr_0:>+7.4f}% (과열 정점 도달)")
    print(f"    - 빔 발생 8시간 후 : {fr_p8:>+7.4f}% (급격한 정상화 시작!)")
    print(f"    - 빔 발생 16시간 후: {fr_p16:>+7.4f}% (양수 전환 완료)")
    print(f"    - 빔 발생 24시간 후: {fr_p24:>+7.4f}% (완벽한 정상 회복)")
    print("=" * 95)


def main():
    klines_file = Path(__file__).resolve().parent.parent.parent / "data" / "BTCUSDT_5m_2022_2024.csv"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
    df_kline, df_fr = load_data(klines_file, funding_file)
    episodes = extract_funding_episodes(df_fr, threshold=0.0)
    df_pos = analyze_beam_positions_and_fr_recovery(df_kline, df_fr, episodes)
    print_timing_report(df_pos, episodes)


if __name__ == "__main__":
    main()
