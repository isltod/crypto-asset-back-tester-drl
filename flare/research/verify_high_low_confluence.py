"""
flare.research.verify_high_low_confluence

실전 지정가 TP/SL 관점:
5분봉 High(최고가) / Low(최저가) 극단 터치 기준 MFE/MAE 정밀 대조
1) [펀딩비 단독 24h] (정산시점 1회 진입)
2) [결합 전략 4h] (꼬리 캔들 진입 후 4시간 내 High/Low)
3) [결합 전략 24h] (꼬리 캔들 진입 후 24시간 내 High/Low)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")


def load_high_low_data(klines_file: Path, funding_file: Path):
    df_kline = pd.read_csv(klines_file)
    if "datetime" in df_kline.columns:
        df_kline["datetime"] = pd.to_datetime(df_kline["datetime"], format="ISO8601", utc=True)
    else:
        df_kline["datetime"] = pd.to_datetime(df_kline["timestamp"], unit="ms", utc=True)
    df_kline = df_kline.sort_values("datetime").reset_index(drop=True)
    
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    
    # 꼬리 계산
    df_kline["body"] = (df_kline["close"] - df_kline["open"]).abs()
    df_kline["total_range"] = df_kline["high"] - df_kline["low"]
    df_kline["lower_wick"] = np.minimum(df_kline["open"], df_kline["close"]) - df_kline["low"]
    df_kline["upper_wick"] = df_kline["high"] - np.maximum(df_kline["open"], df_kline["close"])
    
    safe_range = np.where(df_kline["total_range"] == 0, 1e-9, df_kline["total_range"])
    df_kline["lower_wick_ratio"] = df_kline["lower_wick"] / safe_range
    df_kline["upper_wick_ratio"] = df_kline["upper_wick"] / safe_range
    
    df_kline["vol_sma288"] = df_kline["volume"].rolling(window=288, min_periods=72).mean()
    df_kline["vol_ratio"] = df_kline["volume"] / df_kline["vol_sma288"]
    
    # 4시간 High/Low 롤링
    rolling_max_high_4h = df_kline["high"].iloc[::-1].rolling(window=48, min_periods=48).max().iloc[::-1]
    rolling_min_low_4h = df_kline["low"].iloc[::-1].rolling(window=48, min_periods=48).min().iloc[::-1]
    df_kline["mfe_hl_up_4h"] = (rolling_max_high_4h - df_kline["close"]) / df_kline["close"] * 100
    df_kline["mae_hl_down_4h"] = (df_kline["close"] - rolling_min_low_4h) / df_kline["close"] * 100
    
    # 24시간 High/Low 롤링
    rolling_max_high_24h = df_kline["high"].iloc[::-1].rolling(window=288, min_periods=288).max().iloc[::-1]
    rolling_min_low_24h = df_kline["low"].iloc[::-1].rolling(window=288, min_periods=288).min().iloc[::-1]
    df_kline["mfe_hl_up_24h"] = (rolling_max_high_24h - df_kline["close"]) / df_kline["close"] * 100
    df_kline["mae_hl_down_24h"] = (df_kline["close"] - rolling_min_low_24h) / df_kline["close"] * 100
    
    # 펀딩비 매핑
    df_merged = pd.merge_asof(
        df_kline,
        df_fr[["fundingTime", "fundingRate"]],
        left_on="datetime",
        right_on="fundingTime",
        direction="backward"
    )
    df_merged["fundingRate"] = df_merged["fundingRate"].ffill().fillna(0.0001)
    df_merged["is_funding_event_bar"] = df_merged["datetime"].isin(df_fr["fundingTime"])
    
    df_merged = df_merged.dropna(subset=["vol_ratio", "mfe_hl_up_4h", "mfe_hl_up_24h"]).reset_index(drop=True)
    return df_merged


def run_high_low_comparison(df: pd.DataFrame):
    fr_p05 = df["fundingRate"].quantile(0.05)
    fr_p10 = df["fundingRate"].quantile(0.10)
    fr_p90 = df["fundingRate"].quantile(0.90)
    fr_p95 = df["fundingRate"].quantile(0.95)
    
    cond_lw = (df["vol_ratio"] >= 3.0) & (df["lower_wick_ratio"] >= 0.55) & (df["lower_wick"] >= 1.5 * df["body"])
    cond_uw = (df["vol_ratio"] >= 3.0) & (df["upper_wick_ratio"] >= 0.55) & (df["upper_wick"] >= 1.5 * df["body"])
    
    f10_events = df[df["is_funding_event_bar"] & (df["fundingRate"] <= fr_p10)]
    f05_events = df[df["is_funding_event_bar"] & (df["fundingRate"] <= fr_p05)]
    f90_events = df[df["is_funding_event_bar"] & (df["fundingRate"] >= fr_p90)]
    f95_events = df[df["is_funding_event_bar"] & (df["fundingRate"] >= fr_p95)]
    
    c10_events = df[(df["fundingRate"] <= fr_p10) & cond_lw]
    c05_events = df[(df["fundingRate"] <= fr_p05) & cond_lw]
    c90_events = df[(df["fundingRate"] >= fr_p90) & cond_uw]
    c95_events = df[(df["fundingRate"] >= fr_p95) & cond_uw]
    
    specs = [
        # (이름, 데이터, 방향, horizon_hours, horizon_type)
        ("1. [숏과열 10%] 펀딩비 단독 (24h)", f10_events, "LONG", 24, "24h"),
        ("2. [숏과열 10% + 꼬리] 결합 (4h)", c10_events, "LONG", 4, "4h"),
        ("3. [숏과열 10% + 꼬리] 결합 (24h)", c10_events, "LONG", 24, "24h"),
        ("4. [숏과열 5%] 펀딩비 단독 (24h)", f05_events, "LONG", 24, "24h"),
        ("5. [숏과열 5% + 꼬리] 결합 (4h)", c05_events, "LONG", 4, "4h"),
        ("6. [숏과열 5% + 꼬리] 결합 (24h)", c05_events, "LONG", 24, "24h"),
        ("7. [롱과열 10%] 펀딩비 단독 (24h)", f90_events, "SHORT", 24, "24h"),
        ("8. [롱과열 10% + 꼬리] 결합 (4h)", c90_events, "SHORT", 4, "4h"),
        ("9. [롱과열 5%] 펀딩비 단독 (24h)", f95_events, "SHORT", 24, "24h"),
        ("10. [롱과열 5% + 꼬리] 결합 (4h)", c95_events, "SHORT", 4, "4h"),
    ]
    
    results = []
    for name, sub, direction, h_hours, h_type in specs:
        count = len(sub)
        if count == 0:
            continue
            
        if direction == "LONG":
            mfe = sub[f"mfe_hl_up_{h_type}"].mean()
            mae = sub[f"mae_hl_down_{h_type}"].mean()
            worst_mae = sub[f"mae_hl_down_{h_type}"].max()
            tp_1_5 = (sub[f"mfe_hl_up_{h_type}"] >= 1.5).mean() * 100
            tp_2_0 = (sub[f"mfe_hl_up_{h_type}"] >= 2.0).mean() * 100
        else:
            mfe = sub[f"mae_hl_down_{h_type}"].mean()
            mae = sub[f"mfe_hl_up_{h_type}"].mean()
            worst_mae = sub[f"mfe_hl_up_{h_type}"].max()
            tp_1_5 = (sub[f"mae_hl_down_{h_type}"] >= 1.5).mean() * 100
            tp_2_0 = (sub[f"mae_hl_down_{h_type}"] >= 2.0).mean() * 100
            
        rr = mfe / mae if mae > 0 else np.nan
        mfe_per_hour = mfe / h_hours
        
        results.append({
            "전략명": name,
            "Hz": f"{h_hours}h",
            "표본": count,
            "방향": direction,
            "HL_MFE(익절룸)": mfe,
            "HL_MAE(손실룸)": mae,
            "손익비(MFE/MAE)": rr,
            "시간당MFE": mfe_per_hour,
            "TP +1.5%확률": tp_1_5,
            "TP +2.0%확률": tp_2_0,
            "최악낙폭(MAE)": worst_mae
        })
        
    return pd.DataFrame(results)


def print_hl_report(res_df: pd.DataFrame):
    print("=" * 125)
    print("[FLARE] [실전 지정가 TP/SL 기준] 5분봉 High/Low 극단 진폭 MFE/MAE 정밀 대조 보고서")
    print("=" * 125)
    
    header_fmt = "{:<32} | {:<4} | {:<4} | {:<5} | {:<8} | {:<8} | {:<7} | {:<8} | {:<7} | {:<7} | {:<7}"
    row_fmt = "{:<32} | {:<4} | {:<4} | {:<5} | {:>+7.2f}% | {:>7.2f}% | {:>6.2f}x | {:>+7.2f}%/h | {:>6.1f}% | {:>6.1f}% | {:>6.2f}%"
    
    print(header_fmt.format(
        "전략 및 조건", "Hz", "표본", "방향", "HL_MFE", "HL_MAE", "손익비", "시간당MFE", "TP+1.5%", "TP+2.0%", "최악낙폭"
    ))
    print("-" * 125)
    
    for _, r in res_df.iterrows():
        print(row_fmt.format(
            r["전략명"],
            r["Hz"],
            r["표본"],
            r["방향"],
            r["HL_MFE(익절룸)"],
            r["HL_MAE(손실룸)"],
            r["손익비(MFE/MAE)"],
            r["시간당MFE"],
            r["TP +1.5%확률"],
            r["TP +2.0%확률"],
            r["최악낙폭(MAE)"]
        ))
        
    print("=" * 125)


def main():
    klines_file = Path(__file__).resolve().parent.parent.parent / "data" / "BTCUSDT_5m_2022_2024.csv"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
    df = load_high_low_data(klines_file, funding_file)
    res_df = run_high_low_comparison(df)
    print_hl_report(res_df)


if __name__ == "__main__":
    main()
