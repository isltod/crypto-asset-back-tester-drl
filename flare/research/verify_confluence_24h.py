"""
flare.research.verify_confluence_24h

[검증 1: 펀딩비 단독 24h] vs [검증 2: 펀딩비 + 5분봉 꼬리 결합 24h]를 동일한 조건(24시간 Horizon)에서 
3가지 가격 평가 기준:
1) 24h 최종 종가 (Close)
2) 24h 내 5분봉 종가 기준 최대 진폭 (Max/Min Close)
3) 24h 내 5분봉 극단 진폭 (Max High / Min Low)
으로 1:1 정밀 대조 비교하는 모듈
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")


def load_and_prepare_24h_data(klines_file: Path, funding_file: Path) -> pd.DataFrame:
    """5분봉 캔들과 펀딩비를 결합하고 향후 24시간(288개 5분봉) 롤링 지표를 계산합니다."""
    print(f"[*] 5분봉 데이터 로드 및 24시간 롤링 계산 중: {klines_file.name}...")
    df = pd.read_csv(klines_file)
    
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # 펀딩비 결합
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
    
    # 꼬리 및 거래량 계산
    df["body"] = (df["close"] - df["open"]).abs()
    df["total_range"] = df["high"] - df["low"]
    df["lower_wick"] = np.minimum(df["open"], df["close"]) - df["low"]
    df["upper_wick"] = df["high"] - np.maximum(df["open"], df["close"])
    
    safe_range = np.where(df["total_range"] == 0, 1e-9, df["total_range"])
    df["lower_wick_ratio"] = df["lower_wick"] / safe_range
    df["upper_wick_ratio"] = df["upper_wick"] / safe_range
    
    df["vol_sma288"] = df["volume"].rolling(window=288, min_periods=72).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma288"]
    
    # 24시간(288개 5분봉) 롤링 미래 지표 계산
    # 1) 24h 뒤 종가
    df["close_24h"] = df["close"].shift(-288)
    df["final_ret_24h"] = (df["close_24h"] - df["close"]) / df["close"] * 100
    
    # 2) 24h 내 5분봉 종가 기준 최고/최저 (Max Close / Min Close)
    rolling_max_close_24h = df["close"].iloc[::-1].rolling(window=288, min_periods=288).max().iloc[::-1]
    rolling_min_close_24h = df["close"].iloc[::-1].rolling(window=288, min_periods=288).min().iloc[::-1]
    df["max_close_up_pct"] = (rolling_max_close_24h - df["close"]) / df["close"] * 100
    df["min_close_down_pct"] = (rolling_min_close_24h - df["close"]) / df["close"] * 100
    
    # 3) 24h 내 High 최고가 / Low 최저가 (Max High / Min Low)
    rolling_max_high_24h = df["high"].iloc[::-1].rolling(window=288, min_periods=288).max().iloc[::-1]
    rolling_min_low_24h = df["low"].iloc[::-1].rolling(window=288, min_periods=288).min().iloc[::-1]
    df["max_high_up_pct"] = (rolling_max_high_24h - df["close"]) / df["close"] * 100
    df["min_low_down_pct"] = (rolling_min_low_24h - df["close"]) / df["close"] * 100
    
    df = df.dropna(subset=["vol_ratio", "final_ret_24h", "max_close_up_pct", "max_high_up_pct"]).reset_index(drop=True)
    print(f"[+] 총 {len(df):,}개 5분봉 캔들 24h 롤링 전처리 완료")
    return df


def evaluate_24h_metrics(sub_df: pd.DataFrame, direction: str) -> dict:
    """하위 데이터셋에 대해 3가지 가격 평가 기준 지표를 산출합니다."""
    count = len(sub_df)
    if count == 0:
        return {}
        
    if direction == "LONG":
        final_ret = sub_df["final_ret_24h"].mean()
        final_win = (sub_df["final_ret_24h"] > 0).mean() * 100
        
        # 종가 기준 MFE / MAE
        mfe_close = sub_df["max_close_up_pct"].mean()
        mae_close = sub_df["min_close_down_pct"].abs().mean()
        
        # High/Low 기준 MFE / MAE
        mfe_hl = sub_df["max_high_up_pct"].mean()
        mae_hl = sub_df["min_low_down_pct"].abs().mean()
        p90_mfe_hl = sub_df["max_high_up_pct"].quantile(0.90)
        worst_mae_hl = sub_df["min_low_down_pct"].abs().max()
        tp_1_5pct = (sub_df["max_high_up_pct"] >= 1.5).mean() * 100
        tp_2_0pct = (sub_df["max_high_up_pct"] >= 2.0).mean() * 100
    else:
        final_ret = -sub_df["final_ret_24h"].mean()
        final_win = (sub_df["final_ret_24h"] < 0).mean() * 100
        
        # 종가 기준 MFE / MAE
        mfe_close = sub_df["min_close_down_pct"].abs().mean()
        mae_close = sub_df["max_close_up_pct"].mean()
        
        # High/Low 기준 MFE / MAE
        mfe_hl = sub_df["min_low_down_pct"].abs().mean()
        mae_hl = sub_df["max_high_up_pct"].mean()
        p90_mfe_hl = sub_df["min_low_down_pct"].abs().quantile(0.90)
        worst_mae_hl = sub_df["max_high_up_pct"].max()
        tp_1_5pct = (sub_df["min_low_down_pct"].abs() >= 1.5).mean() * 100
        tp_2_0pct = (sub_df["min_low_down_pct"].abs() >= 2.0).mean() * 100
        
    return {
        "표본수": count,
        "24h종가수익": final_ret,
        "24h종가승률": final_win,
        "종가MFE": mfe_close,
        "종가MAE": mae_close,
        "종가손익비": mfe_close / mae_close if mae_close > 0 else np.nan,
        "HL_MFE": mfe_hl,
        "HL_MAE": mae_hl,
        "HL손익비": mfe_hl / mae_hl if mae_hl > 0 else np.nan,
        "상위10%_MFE": p90_mfe_hl,
        "역대최악_MAE": worst_mae_hl,
        "TP_1.5%확률": tp_1_5pct,
        "TP_2.0%확률": tp_2_0pct,
    }


def compare_funding_vs_confluence_24h(df: pd.DataFrame):
    """펀딩비 단독 vs [펀딩비+꼬리] 결합의 24시간 성과를 정밀 비교합니다."""
    fr_p05 = df["fundingRate"].quantile(0.05)
    fr_p10 = df["fundingRate"].quantile(0.10)
    fr_p90 = df["fundingRate"].quantile(0.90)
    fr_p95 = df["fundingRate"].quantile(0.95)
    
    cond_lw = (df["vol_ratio"] >= 3.0) & (df["lower_wick_ratio"] >= 0.55) & (df["lower_wick"] >= 1.5 * df["body"])
    cond_uw = (df["vol_ratio"] >= 3.0) & (df["upper_wick_ratio"] >= 0.55) & (df["upper_wick"] >= 1.5 * df["body"])
    
    comparisons = [
        ("🟢 [숏과열 10%] 펀딩비 단독 진입", df["fundingRate"] <= fr_p10, "LONG"),
        ("🌟 [숏과열 10% + 아래꼬리] 결합 진입", (df["fundingRate"] <= fr_p10) & cond_lw, "LONG"),
        ("🟢 [숏과열 5%] 펀딩비 단독 진입", df["fundingRate"] <= fr_p05, "LONG"),
        ("🏆 [숏과열 5% + 아래꼬리] 결합 진입", (df["fundingRate"] <= fr_p05) & cond_lw, "LONG"),
        ("💎 [숏패닉 FR<=0] 펀딩비 단독 진입", df["fundingRate"] <= 0.0, "LONG"),
        ("👑 [숏패닉 FR<=0 + 아래꼬리] 결합 진입", (df["fundingRate"] <= 0.0) & cond_lw, "LONG"),
        ("🔴 [롱과열 10%] 펀딩비 단독 진입", df["fundingRate"] >= fr_p90, "SHORT"),
        ("⚠️ [롱과열 10% + 윗꼬리] 결합 진입", (df["fundingRate"] >= fr_p90) & cond_uw, "SHORT"),
        ("🔴 [롱과열 5%] 펀딩비 단독 진입", df["fundingRate"] >= fr_p95, "SHORT"),
        ("🔥 [롱과열 5% + 윗꼬리] 결합 진입", (df["fundingRate"] >= fr_p95) & cond_uw, "SHORT"),
    ]
    
    results = []
    for name, mask, direction in comparisons:
        sub = df[mask]
        res = evaluate_24h_metrics(sub, direction)
        res["비교대상"] = name
        res["방향"] = direction
        results.append(res)
        
    return pd.DataFrame(results)


def print_comparison_report(res_df: pd.DataFrame):
    """비교 결과를 깔끔한 마크다운 및 콘솔 표로 출력합니다."""
    print("=" * 125)
    print("[FLARE] [펀딩비 단독 24h] vs [펀딩비 + 꼬리 결합 24h] 3대 가격 기준 정밀 대조 보고서")
    print("=" * 125)
    print("-" * 125)
    
    header_fmt = "{:<32} | {:<5} | {:<5} | {:<8} | {:<7} | {:<8} | {:<8} | {:<7} | {:<8} | {:<7}"
    row_fmt = "{:<32} | {:<5} | {:<5} | {:>+7.2f}% | {:>6.1f}% | {:>+7.2f}% | {:>+7.2f}% | {:>6.2f}x | {:>7.1f}% | {:>6.2f}%"
    
    print(header_fmt.format(
        "비교 대상 전략 (24h 기준)", "표본", "방향", "24h종가수익", "종가승률", "종가MFE", "HL_MFE(극단)", "HL손익비", "+2.0%터치", "최악MAE"
    ))
    print("-" * 125)
    
    for _, r in res_df.iterrows():
        print(row_fmt.format(
            r["비교대상"],
            r["표본수"],
            r["방향"],
            r["24h종가수익"],
            r["24h종가승률"],
            r["종가MFE"],
            r["HL_MFE"],
            r["HL손익비"],
            r["TP_2.0%확률"],
            r["역대최악_MAE"]
        ))
        
    print("=" * 125)


def main():
    klines_file = Path(__file__).resolve().parent.parent.parent / "data" / "BTCUSDT_5m_2022_2024.csv"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
    df = load_and_prepare_24h_data(klines_file, funding_file)
    res_df = compare_funding_vs_confluence_24h(df)
    print_comparison_report(res_df)


if __name__ == "__main__":
    main()
