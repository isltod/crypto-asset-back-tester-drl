"""
flare.research.verify_strict_comparison

시계열 중복을 완전히 제거한 엄격한 1:1 대조:
1) [펀딩비 단독 진입]: 8시간 펀딩 정산 시점(00:00, 08:00, 16:00)에 '딱 1회만 진입' 후 24h Close MFE/MAE
2) [펀딩비 + 꼬리 결합 진입]: 펀딩 과열 중 '청산 꼬리 완성 시점에 진입' 후 4h Close MFE/MAE
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")


def load_and_prepare_clean_data(klines_file: Path, funding_file: Path):
    """5분봉 캔들과 펀딩비를 정밀 매핑합니다."""
    df_kline = pd.read_csv(klines_file)
    if "datetime" in df_kline.columns:
        df_kline["datetime"] = pd.to_datetime(df_kline["datetime"], format="ISO8601", utc=True)
    else:
        df_kline["datetime"] = pd.to_datetime(df_kline["timestamp"], unit="ms", utc=True)
    df_kline = df_kline.sort_values("datetime").reset_index(drop=True)
    
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    
    # 5분봉 캔들 지표
    df_kline["body"] = (df_kline["close"] - df_kline["open"]).abs()
    df_kline["total_range"] = df_kline["high"] - df_kline["low"]
    df_kline["lower_wick"] = np.minimum(df_kline["open"], df_kline["close"]) - df_kline["low"]
    df_kline["upper_wick"] = df_kline["high"] - np.maximum(df_kline["open"], df_kline["close"])
    
    safe_range = np.where(df_kline["total_range"] == 0, 1e-9, df_kline["total_range"])
    df_kline["lower_wick_ratio"] = df_kline["lower_wick"] / safe_range
    df_kline["upper_wick_ratio"] = df_kline["upper_wick"] / safe_range
    
    df_kline["vol_sma288"] = df_kline["volume"].rolling(window=288, min_periods=72).mean()
    df_kline["vol_ratio"] = df_kline["volume"] / df_kline["vol_sma288"]
    
    # 4h / 24h 롤링
    rolling_max_close_4h = df_kline["close"].iloc[::-1].rolling(window=48, min_periods=48).max().iloc[::-1]
    rolling_min_close_4h = df_kline["close"].iloc[::-1].rolling(window=48, min_periods=48).min().iloc[::-1]
    df_kline["close_4h"] = df_kline["close"].shift(-48)
    df_kline["ret_4h"] = (df_kline["close_4h"] - df_kline["close"]) / df_kline["close"] * 100
    df_kline["mfe_close_4h"] = (rolling_max_close_4h - df_kline["close"]) / df_kline["close"] * 100
    df_kline["mae_close_4h"] = (df_kline["close"] - rolling_min_close_4h) / df_kline["close"] * 100
    
    rolling_max_close_24h = df_kline["close"].iloc[::-1].rolling(window=288, min_periods=288).max().iloc[::-1]
    rolling_min_close_24h = df_kline["close"].iloc[::-1].rolling(window=288, min_periods=288).min().iloc[::-1]
    df_kline["close_24h"] = df_kline["close"].shift(-288)
    df_kline["ret_24h"] = (df_kline["close_24h"] - df_kline["close"]) / df_kline["close"] * 100
    df_kline["mfe_close_24h"] = (rolling_max_close_24h - df_kline["close"]) / df_kline["close"] * 100
    df_kline["mae_close_24h"] = (df_kline["close"] - rolling_min_close_24h) / df_kline["close"] * 100
    
    # 펀딩비 매핑
    df_merged = pd.merge_asof(
        df_kline,
        df_fr[["fundingTime", "fundingRate"]],
        left_on="datetime",
        right_on="fundingTime",
        direction="backward"
    )
    df_merged["fundingRate"] = df_merged["fundingRate"].ffill().fillna(0.0001)
    
    # 8시간 정산 시점 캔들 플래그 (00:00, 08:00, 16:00에 정확히 일치하는 5분봉)
    df_merged["is_funding_event_bar"] = df_merged["datetime"].isin(df_fr["fundingTime"])
    
    df_merged = df_merged.dropna(subset=["vol_ratio", "ret_4h", "ret_24h", "mfe_close_4h", "mfe_close_24h"]).reset_index(drop=True)
    return df_merged


def run_strict_comparison(df: pd.DataFrame):
    """엄격한 단일 진입 대조를 수행합니다."""
    # 펀딩비 분위수
    fr_p05 = df["fundingRate"].quantile(0.05)
    fr_p10 = df["fundingRate"].quantile(0.10)
    fr_p90 = df["fundingRate"].quantile(0.90)
    fr_p95 = df["fundingRate"].quantile(0.95)
    
    cond_lw = (df["vol_ratio"] >= 3.0) & (df["lower_wick_ratio"] >= 0.55) & (df["lower_wick"] >= 1.5 * df["body"])
    cond_uw = (df["vol_ratio"] >= 3.0) & (df["upper_wick_ratio"] >= 0.55) & (df["upper_wick"] >= 1.5 * df["body"])
    
    # 1. 펀딩비 단독: 8시간 정산 시점 1회만 진입!
    f10_events = df[df["is_funding_event_bar"] & (df["fundingRate"] <= fr_p10)]
    f05_events = df[df["is_funding_event_bar"] & (df["fundingRate"] <= fr_p05)]
    f90_events = df[df["is_funding_event_bar"] & (df["fundingRate"] >= fr_p90)]
    f95_events = df[df["is_funding_event_bar"] & (df["fundingRate"] >= fr_p95)]
    
    # 2. 결합 진입: 과열 중 청산 꼬리 시점에 진입!
    c10_events = df[(df["fundingRate"] <= fr_p10) & cond_lw]
    c05_events = df[(df["fundingRate"] <= fr_p05) & cond_lw]
    c90_events = df[(df["fundingRate"] >= fr_p90) & cond_uw]
    c95_events = df[(df["fundingRate"] >= fr_p95) & cond_uw]
    
    specs = [
        ("1. [숏과열 10%] 펀딩비 단독 (정산시점 1회)", f10_events, "LONG", 24, "24h"),
        ("2. [숏과열 10% + 아래꼬리] 결합 진입", c10_events, "LONG", 4, "4h"),
        ("3. [숏과열 5%] 펀딩비 단독 (정산시점 1회)", f05_events, "LONG", 24, "24h"),
        ("4. [숏과열 5% + 아래꼬리] 결합 진입", c05_events, "LONG", 4, "4h"),
        ("5. [롱과열 10%] 펀딩비 단독 (정산시점 1회)", f90_events, "SHORT", 24, "24h"),
        ("6. [롱과열 10% + 윗꼬리] 결합 진입", c90_events, "SHORT", 4, "4h"),
        ("7. [롱과열 5%] 펀딩비 단독 (정산시점 1회)", f95_events, "SHORT", 24, "24h"),
        ("8. [롱과열 5% + 윗꼬리] 결합 진입", c95_events, "SHORT", 4, "4h"),
    ]
    
    results = []
    for name, sub, direction, h_hours, h_type in specs:
        count = len(sub)
        if count == 0:
            continue
            
        if direction == "LONG":
            ret = sub[f"ret_{h_type}"].mean()
            win = (sub[f"ret_{h_type}"] > 0).mean() * 100
            mfe = sub[f"mfe_close_{h_type}"].mean()
            mae = sub[f"mae_close_{h_type}"].mean()
            worst_mae = sub[f"mae_close_{h_type}"].max()
            tp_1_5 = (sub[f"mfe_close_{h_type}"] >= 1.5).mean() * 100
        else:
            ret = -sub[f"ret_{h_type}"].mean()
            win = (sub[f"ret_{h_type}"] < 0).mean() * 100
            mfe = sub[f"mae_close_{h_type}"].mean()
            mae = sub[f"mfe_close_{h_type}"].mean()
            worst_mae = sub[f"mfe_close_{h_type}"].max()
            tp_1_5 = (sub[f"mae_close_{h_type}"] >= 1.5).mean() * 100
            
        rr = mfe / mae if mae > 0 else np.nan
        mfe_per_hour = mfe / h_hours
        
        results.append({
            "전략명": name,
            "Horizon": f"{h_hours}시간",
            "진짜표본수": count,
            "방향": direction,
            "만기수익": ret,
            "승률": win,
            "MFE": mfe,
            "MAE": mae,
            "손익비(MFE/MAE)": rr,
            "시간당MFE": mfe_per_hour,
            "최악낙폭(MAE)": worst_mae
        })
        
    return pd.DataFrame(results)


def print_strict_report(res_df: pd.DataFrame):
    """엄격 대조 보고서를 출력합니다."""
    print("=" * 125)
    print("[FLARE] [펀딩비 정산시점 1회 진입 (24h)] vs [청산 꼬리 진입 (4h)] 엄격 1:1 대조 보고서")
    print("=" * 125)
    
    header_fmt = "{:<36} | {:<5} | {:<5} | {:<5} | {:<8} | {:<7} | {:<8} | {:<8} | {:<7} | {:<8} | {:<8}"
    row_fmt = "{:<36} | {:<5} | {:<5} | {:<5} | {:>+7.2f}% | {:>6.1f}% | {:>+7.2f}% | {:>7.2f}% | {:>6.2f}x | {:>+7.2f}%/h | {:>7.2f}%"
    
    print(header_fmt.format(
        "전략 및 조건", "Hz", "표본", "방향", "만기수익", "승률", "종가MFE", "종가MAE", "손익비", "시간당MFE", "최악낙폭"
    ))
    print("-" * 125)
    
    for _, r in res_df.iterrows():
        print(row_fmt.format(
            r["전략명"],
            r["Horizon"],
            r["진짜표본수"],
            r["방향"],
            r["만기수익"],
            r["승률"],
            r["MFE"],
            r["MAE"],
            r["손익비(MFE/MAE)"],
            r["시간당MFE"],
            r["최악낙폭(MAE)"]
        ))
        
    print("=" * 125)


def main():
    klines_file = Path(__file__).resolve().parent.parent.parent / "data" / "BTCUSDT_5m_2022_2024.csv"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
    df = load_and_prepare_clean_data(klines_file, funding_file)
    res_df = run_strict_comparison(df)
    print_strict_report(res_df)


if __name__ == "__main__":
    main()
