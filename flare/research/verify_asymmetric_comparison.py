"""
flare.research.verify_asymmetric_comparison

사용자 맞춤형 정밀 대조:
1) [펀딩비 단독 진입]: 24시간 Horizon의 Close 기준 MFE / MAE
2) [펀딩비 + 꼬리 결합 진입]: 4시간 Horizon의 Close 기준 MFE / MAE
두 전략의 MFE, MAE, 손익비(MFE/MAE), 시간 효율(시간당 수익), 익절 도달률 등을 1:1 비교
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")


def load_data_and_calc_horizons(klines_file: Path, funding_file: Path) -> pd.DataFrame:
    """5분봉 캔들과 펀딩비를 결합하고 4시간(48봉) 및 24시간(288봉) Close 기준 MFE/MAE를 계산합니다."""
    print(f"[*] 데이터 로드 및 4h / 24h 종가 기준 롤링 계산 중...")
    df = pd.read_csv(klines_file)
    
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        
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
    
    # 캔들 구조
    df["body"] = (df["close"] - df["open"]).abs()
    df["total_range"] = df["high"] - df["low"]
    df["lower_wick"] = np.minimum(df["open"], df["close"]) - df["low"]
    df["upper_wick"] = df["high"] - np.maximum(df["open"], df["close"])
    
    safe_range = np.where(df["total_range"] == 0, 1e-9, df["total_range"])
    df["lower_wick_ratio"] = df["lower_wick"] / safe_range
    df["upper_wick_ratio"] = df["upper_wick"] / safe_range
    
    df["vol_sma288"] = df["volume"].rolling(window=288, min_periods=72).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma288"]
    
    # 4시간(48개 봉) 종가 롤링
    rolling_max_close_4h = df["close"].iloc[::-1].rolling(window=48, min_periods=48).max().iloc[::-1]
    rolling_min_close_4h = df["close"].iloc[::-1].rolling(window=48, min_periods=48).min().iloc[::-1]
    df["close_4h"] = df["close"].shift(-48)
    df["ret_4h"] = (df["close_4h"] - df["close"]) / df["close"] * 100
    df["mfe_close_up_4h"] = (rolling_max_close_4h - df["close"]) / df["close"] * 100
    df["mae_close_down_4h"] = (df["close"] - rolling_min_close_4h) / df["close"] * 100
    
    # 24시간(288개 봉) 종가 롤링
    rolling_max_close_24h = df["close"].iloc[::-1].rolling(window=288, min_periods=288).max().iloc[::-1]
    rolling_min_close_24h = df["close"].iloc[::-1].rolling(window=288, min_periods=288).min().iloc[::-1]
    df["close_24h"] = df["close"].shift(-288)
    df["ret_24h"] = (df["close_24h"] - df["close"]) / df["close"] * 100
    df["mfe_close_up_24h"] = (rolling_max_close_24h - df["close"]) / df["close"] * 100
    df["mae_close_down_24h"] = (df["close"] - rolling_min_close_24h) / df["close"] * 100
    
    df = df.dropna(subset=["vol_ratio", "ret_4h", "ret_24h", "mfe_close_up_4h", "mfe_close_up_24h"]).reset_index(drop=True)
    print(f"[+] 총 {len(df):,}개 캔들 전처리 완료")
    return df


def analyze_custom_comparison(df: pd.DataFrame):
    """사용자가 지정한 단독(24h) vs 결합(4h)을 정밀 비교합니다."""
    fr_p05 = df["fundingRate"].quantile(0.05)
    fr_p10 = df["fundingRate"].quantile(0.10)
    fr_p90 = df["fundingRate"].quantile(0.90)
    fr_p95 = df["fundingRate"].quantile(0.95)
    
    cond_lw = (df["vol_ratio"] >= 3.0) & (df["lower_wick_ratio"] >= 0.55) & (df["lower_wick"] >= 1.5 * df["body"])
    cond_uw = (df["vol_ratio"] >= 3.0) & (df["upper_wick_ratio"] >= 0.55) & (df["upper_wick"] >= 1.5 * df["body"])
    
    # 1. 숏 과열 10% (LONG)
    sub_f10_24 = df[df["fundingRate"] <= fr_p10]
    sub_c10_4 = df[(df["fundingRate"] <= fr_p10) & cond_lw]
    
    # 2. 숏 과열 5% (LONG)
    sub_f05_24 = df[df["fundingRate"] <= fr_p05]
    sub_c05_4 = df[(df["fundingRate"] <= fr_p05) & cond_lw]
    
    # 3. 롱 과열 10% (SHORT)
    sub_f90_24 = df[df["fundingRate"] >= fr_p90]
    sub_c90_4 = df[(df["fundingRate"] >= fr_p90) & cond_uw]
    
    # 4. 롱 과열 5% (SHORT)
    sub_f95_24 = df[df["fundingRate"] >= fr_p95]
    sub_c95_4 = df[(df["fundingRate"] >= fr_p95) & cond_uw]
    
    specs = [
        # (이름, 데이터, 방향, horizon_hours, horizon_type)
        ("1. [숏과열 10%] 펀딩비 단독", sub_f10_24, "LONG", 24, "24h"),
        ("2. [숏과열 10% + 아래꼬리] 결합", sub_c10_4, "LONG", 4, "4h"),
        ("3. [숏과열 5%] 펀딩비 단독", sub_f05_24, "LONG", 24, "24h"),
        ("4. [숏과열 5% + 아래꼬리] 결합", sub_c05_4, "LONG", 4, "4h"),
        ("5. [롱과열 10%] 펀딩비 단독", sub_f90_24, "SHORT", 24, "24h"),
        ("6. [롱과열 10% + 윗꼬리] 결합", sub_c90_4, "SHORT", 4, "4h"),
        ("7. [롱과열 5%] 펀딩비 단독", sub_f95_24, "SHORT", 24, "24h"),
        ("8. [롱과열 5% + 윗꼬리] 결합", sub_c95_4, "SHORT", 4, "4h"),
    ]
    
    results = []
    for name, sub, direction, h_hours, h_type in specs:
        count = len(sub)
        if count == 0:
            continue
            
        if direction == "LONG":
            ret = sub[f"ret_{h_type}"].mean()
            win = (sub[f"ret_{h_type}"] > 0).mean() * 100
            mfe = sub[f"mfe_close_up_{h_type}"].mean()
            mae = sub[f"mae_close_down_{h_type}"].mean()
            tp_1_5 = (sub[f"mfe_close_up_{h_type}"] >= 1.5).mean() * 100
        else:
            ret = -sub[f"ret_{h_type}"].mean()
            win = (sub[f"ret_{h_type}"] < 0).mean() * 100
            mfe = sub[f"mae_close_down_{h_type}"].mean()
            mae = sub[f"mfe_close_up_{h_type}"].mean()
            tp_1_5 = (sub[f"mae_close_down_{h_type}"] >= 1.5).mean() * 100
            
        rr = mfe / mae if mae > 0 else np.nan
        # 시간당 효율: (평균 MFE / 보유 시간)
        mfe_per_hour = mfe / h_hours
        
        results.append({
            "전략명": name,
            "Horizon": f"{h_hours}시간",
            "표본수": count,
            "방향": direction,
            "만기종가수익": ret,
            "종가승률": win,
            "종가MFE": mfe,
            "종가MAE": mae,
            "손익비(MFE/MAE)": rr,
            "시간당MFE속도": mfe_per_hour,
            "TP +1.5%확률": tp_1_5
        })
        
    return pd.DataFrame(results)


def print_custom_report(res_df: pd.DataFrame):
    """결과를 깔끔하게 출력합니다."""
    print("=" * 125)
    print("[FLARE] [펀딩비 단독 24h] vs [펀딩비 + 꼬리 결합 4h] 사용자 정의 MFE/MAE 정밀 대조 보고서")
    print("=" * 125)
    
    header_fmt = "{:<32} | {:<6} | {:<5} | {:<5} | {:<8} | {:<7} | {:<8} | {:<8} | {:<7} | {:<8} | {:<8}"
    row_fmt = "{:<32} | {:<6} | {:<5} | {:<5} | {:>+7.2f}% | {:>6.1f}% | {:>+7.2f}% | {:>7.2f}% | {:>6.2f}x | {:>+7.2f}%/h | {:>7.1f}%"
    
    print(header_fmt.format(
        "전략 및 조건", "Horizon", "표본", "방향", "만기수익", "승률", "종가MFE", "종가MAE", "손익비", "시간당MFE", "TP+1.5%"
    ))
    print("-" * 125)
    
    for _, r in res_df.iterrows():
        print(row_fmt.format(
            r["전략명"],
            r["Horizon"],
            r["표본수"],
            r["방향"],
            r["만기종가수익"],
            r["종가승률"],
            r["종가MFE"],
            r["종가MAE"],
            r["손익비(MFE/MAE)"],
            r["시간당MFE속도"],
            r["TP +1.5%확률"]
        ))
        
    print("=" * 125)


def main():
    klines_file = Path(__file__).resolve().parent.parent.parent / "data" / "BTCUSDT_5m_2022_2024.csv"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
    df = load_data_and_calc_horizons(klines_file, funding_file)
    res_df = analyze_custom_comparison(df)
    print_custom_report(res_df)


if __name__ == "__main__":
    main()
