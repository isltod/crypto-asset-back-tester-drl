"""
flare.research.verify_liquidation_proxy

5분봉 OHLCV 데이터와 펀딩비 데이터를 결합하여:
1) 단독 5분봉 청산 꼬리 캔들의 성과
2) [펀딩비 극단값 + 5분봉 청산 꼬리 캔들] 결합(Confluence) 시의 성과를 교차 검증하는 모듈
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")


def load_and_preprocess_combined_data(klines_file: Path, funding_file: Path) -> pd.DataFrame:
    """5분봉 캔들과 펀딩비 데이터를 결합합니다."""
    print(f"[*] 5분봉 데이터 로드 중: {klines_file.name}...")
    df = pd.read_csv(klines_file)
    
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # 펀딩비 결합
    if funding_file.exists():
        df_fr = pd.read_csv(funding_file)
        df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
        df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
        
        # 5분봉 시점에 직전 최신 펀딩비 매핑 (backward)
        df = pd.merge_asof(
            df,
            df_fr[["fundingTime", "fundingRate"]],
            left_on="datetime",
            right_on="fundingTime",
            direction="backward"
        )
        df["fundingRate"] = df["fundingRate"].ffill().fillna(0.0001)
    else:
        df["fundingRate"] = 0.0001
        
    # 캔들 구조 계산
    df["body"] = (df["close"] - df["open"]).abs()
    df["total_range"] = df["high"] - df["low"]
    
    # 아래꼬리, 윗꼬리
    df["lower_wick"] = np.minimum(df["open"], df["close"]) - df["low"]
    df["upper_wick"] = df["high"] - np.maximum(df["open"], df["close"])
    
    safe_range = np.where(df["total_range"] == 0, 1e-9, df["total_range"])
    df["lower_wick_ratio"] = df["lower_wick"] / safe_range
    df["upper_wick_ratio"] = df["upper_wick"] / safe_range
    
    # 거래량 이동평균 (24H = 288개 봉)
    df["vol_sma288"] = df["volume"].rolling(window=288, min_periods=72).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma288"]
    
    # 2시간(24개 봉) MFE / MAE
    rolling_max_high_2h = df["high"].iloc[::-1].rolling(window=24, min_periods=24).max().iloc[::-1]
    rolling_min_low_2h = df["low"].iloc[::-1].rolling(window=24, min_periods=24).min().iloc[::-1]
    
    df["max_high_2h"] = rolling_max_high_2h
    df["min_low_2h"] = rolling_min_low_2h
    df["close_2h"] = df["close"].shift(-24)
    df["ret_2h"] = (df["close_2h"] - df["close"]) / df["close"] * 100
    df["max_up_2h_pct"] = (df["max_high_2h"] - df["close"]) / df["close"] * 100
    df["max_down_2h_pct"] = (df["min_low_2h"] - df["close"]) / df["close"] * 100
    
    # 4시간(48개 봉) MFE / MAE
    rolling_max_high_4h = df["high"].iloc[::-1].rolling(window=48, min_periods=48).max().iloc[::-1]
    rolling_min_low_4h = df["low"].iloc[::-1].rolling(window=48, min_periods=48).min().iloc[::-1]
    df["max_up_4h_pct"] = (rolling_max_high_4h - df["close"]) / df["close"] * 100
    df["max_down_4h_pct"] = (rolling_min_low_4h - df["close"]) / df["close"] * 100
    
    df = df.dropna(subset=["vol_ratio", "ret_2h", "max_up_4h_pct", "max_down_4h_pct"]).reset_index(drop=True)
    print(f"[+] 총 {len(df):,}개 5분봉 캔들 전처리 완료 ({df['datetime'].min()} ~ {df['datetime'].max()})")
    return df


def analyze_confluence_signals(df: pd.DataFrame) -> dict:
    """단독 꼬리 캔들과 [펀딩비 + 꼬리 캔들] 결합 시그널의 성과를 대조 분석합니다."""
    # 펀딩비 분위수
    fr_p05 = df["fundingRate"].quantile(0.05)  # 숏 과열 기준 (약 -0.005%)
    fr_p10 = df["fundingRate"].quantile(0.10)
    fr_p90 = df["fundingRate"].quantile(0.90)  # 롱 과열 기준 (약 +0.02%)
    fr_p95 = df["fundingRate"].quantile(0.95)
    
    # 꼬리 조건
    cond_lw = (df["vol_ratio"] >= 3.0) & (df["lower_wick_ratio"] >= 0.55) & (df["lower_wick"] >= 1.5 * df["body"])
    cond_uw = (df["vol_ratio"] >= 3.0) & (df["upper_wick_ratio"] >= 0.55) & (df["upper_wick"] >= 1.5 * df["body"])
    
    normal_mask = (df["vol_ratio"] <= 1.5)
    normal_2h_returns = df.loc[normal_mask, "ret_2h"]
    
    categories = [
        # 단독 꼬리
        ("1. [단독] 롱청산빔 (아래꼬리 55%+)", cond_lw, "LONG"),
        ("2. [단독] 숏청산빔 (윗꼬리 55%+)", cond_uw, "SHORT"),
        ("3. [기준] 평소 정상 캔들 (Vol <= 1.5x)", normal_mask, "NEUTRAL"),
        # 🟢 숏과열(공포) + 롱청산빔 결합 (LONG Confluence)
        ("4. [결합★] 숏과열 10% + 아래꼬리 55%", cond_lw & (df["fundingRate"] <= fr_p10), "LONG"),
        ("5. [결합★★] 숏과열 5% + 아래꼬리 55%", cond_lw & (df["fundingRate"] <= fr_p05), "LONG"),
        ("6. [결합★★★] 숏패닉(FR<=0) + 아래꼬리 55%", cond_lw & (df["fundingRate"] <= 0.0), "LONG"),
        # 🔴 롱과열(탐욕) + 숏청산빔 결합 (SHORT Confluence)
        ("7. [결합★] 롱과열 10% + 윗꼬리 55%", cond_uw & (df["fundingRate"] >= fr_p90), "SHORT"),
        ("8. [결합★★] 롱과열 5% + 윗꼬리 55%", cond_uw & (df["fundingRate"] >= fr_p95), "SHORT"),
    ]
    
    results = []
    
    for name, mask, direction in categories:
        sub = df[mask]
        count = len(sub)
        if count == 0:
            continue
            
        if direction == "LONG":
            ret_2h = sub["ret_2h"].mean()
            win_rate_2h = (sub["ret_2h"] > 0).mean() * 100
            mfe_2h = sub["max_up_2h_pct"].mean()
            mae_2h = sub["max_down_2h_pct"].abs().mean()
            mfe_4h = sub["max_up_4h_pct"].mean()
            mae_4h = sub["max_down_4h_pct"].abs().mean()
            tp_1pct = (sub["max_up_2h_pct"] >= 1.0).mean() * 100
            t_stat, p_val = stats.ttest_ind(sub["ret_2h"], normal_2h_returns, equal_var=False)
        elif direction == "SHORT":
            ret_2h = -sub["ret_2h"].mean()
            win_rate_2h = (sub["ret_2h"] < 0).mean() * 100
            mfe_2h = sub["max_down_2h_pct"].abs().mean()
            mae_2h = sub["max_up_2h_pct"].mean()
            mfe_4h = sub["max_down_4h_pct"].abs().mean()
            mae_4h = sub["max_up_4h_pct"].mean()
            tp_1pct = (sub["max_down_2h_pct"].abs() >= 1.0).mean() * 100
            t_stat, p_val = stats.ttest_ind(-sub["ret_2h"], -normal_2h_returns, equal_var=False)
        else:
            ret_2h = sub["ret_2h"].mean()
            win_rate_2h = np.nan
            mfe_2h = sub["max_up_2h_pct"].mean()
            mae_2h = sub["max_down_2h_pct"].abs().mean()
            mfe_4h = sub["max_up_4h_pct"].mean()
            mae_4h = sub["max_down_4h_pct"].abs().mean()
            tp_1pct = np.nan
            p_val = np.nan
            
        rr_2h = mfe_2h / mae_2h if mae_2h > 0 else np.nan
        rr_4h = mfe_4h / mae_4h if mae_4h > 0 else np.nan
        
        results.append({
            "시그널명": name,
            "표본수": count,
            "방향": direction,
            "2h수익": ret_2h,
            "2h승률": win_rate_2h,
            "2h MFE": mfe_2h,
            "2h MAE": mae_2h,
            "2h손익비": rr_2h,
            "4h손익비": rr_4h,
            "+1.0%도달": tp_1pct,
            "p-val": p_val
        })
        
    return {
        "results": pd.DataFrame(results),
        "total_samples": len(df),
        "start_date": df["datetime"].min(),
        "end_date": df["datetime"].max(),
    }


def print_confluence_report(summary: dict):
    """결과를 마크다운 표 및 콘솔로 출력합니다."""
    print("=" * 120)
    print("[FLARE] 단독 꼬리 캔들 vs [펀딩비 + 5분봉 꼬리 캔들] 결합(Confluence) 검증 보고서")
    print("=" * 120)
    print(f"[*] 분석 기간: {summary['start_date'].strftime('%Y-%m-%d')} ~ {summary['end_date'].strftime('%Y-%m-%d')} (총 {summary['total_samples']:,}개 5분봉 캔들)")
    print("-" * 120)
    
    res_df = summary["results"]
    
    header_fmt = "{:<36} | {:<5} | {:<5} | {:<8} | {:<7} | {:<8} | {:<8} | {:<7} | {:<7} | {:<8} | {:<7}"
    row_fmt = "{:<36} | {:<5} | {:<5} | {:>+7.2f}% | {:>6} | {:>+7.2f}% | {:>7.2f}% | {:>6} | {:>6} | {:>7} | {:>7}"
    
    print(header_fmt.format(
        "시그널 및 결합 조건", "표본", "방향", "2h수익", "2h승률", "2h MFE", "2h MAE", "2h손익비", "4h손익비", "+1%도달", "p-val"
    ))
    print("-" * 120)
    
    for _, row in res_df.iterrows():
        win_str = f"{row['2h승률']:.1f}%" if pd.notnull(row['2h승률']) else "-"
        tp_str = f"{row['+1.0%도달']:.1f}%" if pd.notnull(row['+1.0%도달']) else "-"
        p_str = f"{row['p-val']:.4f}" if pd.notnull(row['p-val']) else "-"
        rr2_str = f"{row['2h손익비']:.2f}x" if pd.notnull(row['2h손익비']) else "-"
        rr4_str = f"{row['4h손익비']:.2f}x" if pd.notnull(row['4h손익비']) else "-"
        
        print(row_fmt.format(
            row['시그널명'],
            row['표본수'],
            row['방향'],
            row['2h수익'],
            win_str,
            row['2h MFE'],
            row['2h MAE'],
            rr2_str,
            rr4_str,
            tp_str,
            p_str
        ))
        
    print("=" * 120)


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    klines_file = data_dir / "BTCUSDT_5m_2022_2024.csv"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
    df = load_and_preprocess_combined_data(klines_file, funding_file)
    summary = analyze_confluence_signals(df)
    print_confluence_report(summary)


if __name__ == "__main__":
    main()
