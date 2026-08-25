"""
flare.research.verify_funding_rate

수집된 펀딩비 이력 데이터와 1시간봉 OHLCV 가격 데이터를 정밀 결합하여,
극단적 펀딩비 발생 시점 이후 24시간 동안의:
1) 단순 24h 후 종가 수익률
2) 24시간 내 최대 도달 수익 (MFE) 및 역대 최고 극단값 (Max Peak)
3) 24시간 내 최대 역행 손실 (MAE) 및 역대 최악 극단값 (Worst Drawdown)
4) 상/하방 극단 바운드(Bound) 한계치를 전수 검증하는 모듈
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")


def load_and_merge_mfe_mae_data(funding_file: Path, klines_file: Path) -> pd.DataFrame:
    """
    펀딩비 데이터와 1시간봉 캔들(High, Low, Close)을 결합하여 향후 24시간 내 MFE/MAE를 계산합니다.
    """
    df_funding = pd.read_csv(funding_file)
    df_funding["fundingTime"] = pd.to_datetime(df_funding["fundingTime"], format="ISO8601", utc=True)
    df_funding = df_funding.sort_values("fundingTime").reset_index(drop=True)
    
    df_kline = pd.read_csv(klines_file)
    if "datetime" in df_kline.columns:
        df_kline["datetime"] = pd.to_datetime(df_kline["datetime"], format="ISO8601", utc=True)
    else:
        df_kline["datetime"] = pd.to_datetime(df_kline["timestamp"], unit="ms", utc=True)
    df_kline = df_kline.sort_values("datetime").reset_index(drop=True)
    
    rolling_max_high_24h = df_kline["high"].iloc[::-1].rolling(window=24, min_periods=24).max().iloc[::-1]
    rolling_min_low_24h = df_kline["low"].iloc[::-1].rolling(window=24, min_periods=24).min().iloc[::-1]
    close_24h_later = df_kline["close"].shift(-24)
    
    df_kline["max_high_24h"] = rolling_max_high_24h
    df_kline["min_low_24h"] = rolling_min_low_24h
    df_kline["close_24h"] = close_24h_later
    
    df_merged = pd.merge_asof(
        df_funding,
        df_kline[["datetime", "open", "close", "max_high_24h", "min_low_24h", "close_24h"]],
        left_on="fundingTime",
        right_on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta(hours=1)
    )
    
    df_merged["entry_price"] = df_merged["markPrice"].fillna(df_merged["close"])
    df_merged = df_merged.dropna(subset=["entry_price", "max_high_24h", "min_low_24h", "close_24h"]).copy()
    
    df_merged["max_up_pct"] = (df_merged["max_high_24h"] - df_merged["entry_price"]) / df_merged["entry_price"] * 100
    df_merged["max_down_pct"] = (df_merged["min_low_24h"] - df_merged["entry_price"]) / df_merged["entry_price"] * 100
    df_merged["final_close_pct"] = (df_merged["close_24h"] - df_merged["entry_price"]) / df_merged["entry_price"] * 100
    
    return df_merged


def analyze_extreme_bounds(df: pd.DataFrame) -> dict:
    """
    구간별 최대 유리 진폭(MFE)의 극단값과 최대 역행 손실(MAE)의 최악값을 산출합니다.
    """
    p01 = df["fundingRate"].quantile(0.01)
    p05 = df["fundingRate"].quantile(0.05)
    p10 = df["fundingRate"].quantile(0.10)
    p90 = df["fundingRate"].quantile(0.90)
    p95 = df["fundingRate"].quantile(0.95)
    p99 = df["fundingRate"].quantile(0.99)
    
    normal_mask = (df["fundingRate"] >= p10) & (df["fundingRate"] <= p90)
    
    categories = [
        ("극단 숏 과열 (하위 1%)", df["fundingRate"] <= p01, "LONG"),
        ("강한 숏 과열 (하위 5%)", df["fundingRate"] <= p05, "LONG"),
        ("약한 숏 과열 (하위 10%)", df["fundingRate"] <= p10, "LONG"),
        ("정상/중립 (10% ~ 90%)", normal_mask, "NEUTRAL"),
        ("약한 롱 과열 (상위 10%)", df["fundingRate"] >= p90, "SHORT"),
        ("강한 롱 과열 (상위 5%)", df["fundingRate"] >= p95, "SHORT"),
        ("극단 롱 과열 (상위 1%)", df["fundingRate"] >= p99, "SHORT"),
        ("극단 숏 패닉 (FR <= -0.03%)", df["fundingRate"] <= -0.0003, "LONG"),
        ("극단 롱 광기 (FR >= +0.08%)", df["fundingRate"] >= 0.0008, "SHORT"),
    ]
    
    results = []
    
    for name, mask, rev_dir in categories:
        sub = df[mask]
        count = len(sub)
        if count == 0:
            continue
            
        mean_fr = sub["fundingRate"].mean() * 100
        
        if rev_dir == "LONG":
            # 롱 진입: 유리=상방(max_up), 불리=하방(max_down)
            mean_mfe = sub["max_up_pct"].mean()
            abs_max_mfe = sub["max_up_pct"].max()
            p90_mfe = sub["max_up_pct"].quantile(0.90)
            
            mean_mae = sub["max_down_pct"].abs().mean()
            worst_mae = sub["max_down_pct"].abs().max()
            p90_mae = sub["max_down_pct"].abs().quantile(0.90)
        elif rev_dir == "SHORT":
            # 숏 진입: 유리=하방(max_down 절대값), 불리=상방(max_up)
            mean_mfe = sub["max_down_pct"].abs().mean()
            abs_max_mfe = sub["max_down_pct"].abs().max()
            p90_mfe = sub["max_down_pct"].abs().quantile(0.90)
            
            mean_mae = sub["max_up_pct"].mean()
            worst_mae = sub["max_up_pct"].max()
            p90_mae = sub["max_up_pct"].quantile(0.90)
        else:
            mean_mfe = sub["max_up_pct"].mean()
            abs_max_mfe = sub["max_up_pct"].max()
            p90_mfe = sub["max_up_pct"].quantile(0.90)
            mean_mae = sub["max_down_pct"].abs().mean()
            worst_mae = sub["max_down_pct"].abs().max()
            p90_mae = sub["max_down_pct"].abs().quantile(0.90)
            
        results.append({
            "구간명": name,
            "표본수": count,
            "방향": rev_dir,
            "평균MFE": mean_mfe,
            "상위10% MFE": p90_mfe,
            "역대최대 MFE (Max)": abs_max_mfe,
            "평균MAE": mean_mae,
            "상위10% MAE": p90_mae,
            "역대최악 MAE (Worst)": worst_mae,
        })
        
    return {
        "results": pd.DataFrame(results),
        "total_samples": len(df),
        "start_date": df["fundingTime"].min(),
        "end_date": df["fundingTime"].max(),
    }


def print_bounds_report(summary: dict):
    """극단 바운드 분석 결과를 콘솔로 출력합니다."""
    print("=" * 110)
    print("[FLARE] 24시간 내 극단 바운드(Upper/Lower Extreme Bounds) 한계치 검증 보고서")
    print("=" * 110)
    print(f"[*] 분석 기간: {summary['start_date'].strftime('%Y-%m-%d')} ~ {summary['end_date'].strftime('%Y-%m-%d')} (총 {summary['total_samples']:,}개 8시간 캔들)")
    print("-" * 110)
    
    res_df = summary["results"]
    
    header_fmt = "{:<24} | {:<5} | {:<5} | {:<9} | {:<10} | {:<13} | {:<9} | {:<10} | {:<13}"
    row_fmt = "{:<24} | {:<5} | {:<5} | {:>+8.2f}% | {:>+9.2f}% | {:>+12.2f}% | {:>8.2f}% | {:>9.2f}% | {:>12.2f}%"
    
    print(header_fmt.format(
        "구간 및 조건", "표본", "방향", "평균MFE", "상위10%MFE", "역대최대MFE(Max)", "평균MAE", "상위10%MAE", "역대최악MAE(Worst)"
    ))
    print("-" * 110)
    
    for _, row in res_df.iterrows():
        print(row_fmt.format(
            row['구간명'],
            row['표본수'],
            row['방향'],
            row['평균MFE'],
            row['상위10% MFE'],
            row['역대최대 MFE (Max)'],
            row['평균MAE'],
            row['상위10% MAE'],
            row['역대최악 MAE (Worst)']
        ))
        
    print("=" * 110)


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    funding_file = data_dir / "btcusdt_funding_rate.csv"
    klines_file = Path(__file__).resolve().parent.parent.parent / "data" / "BTCUSDT_1h_4years_full.csv"
    
    df = load_and_merge_mfe_mae_data(funding_file, klines_file)
    summary = analyze_extreme_bounds(df)
    print_bounds_report(summary)


if __name__ == "__main__":
    main()
