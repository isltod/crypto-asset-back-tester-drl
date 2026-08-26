"""
flare.research.test_funding_absolute_levels

펀딩비 절대값 임계치(Absolute Threshold Levels) 정밀 검증
- 조건: FR <= 0.0000 (음수) vs -0.005% vs -0.010% vs -0.020% vs -0.030%
- 검증 구간: 2021년 불장(OOS) 및 2022~2024년(인샘플) 전체 4개년
- 각 절대값 수준별 거래수, 승률, 누적수익률, MDD 비교
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.data.features import generate_all_features
from flare.backtest.engine import TripleBarrierEngine


def run_threshold_grid_on_dataset(klines_path: Path, dataset_name: str):
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
    df = pd.read_csv(klines_path)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    df = pd.merge_asof(df, df_fr[["fundingTime", "fundingRate"]], left_on="datetime", right_on="fundingTime", direction="backward")
    df["fundingRate"] = df["fundingRate"].ffill().fillna(0.0001)
    
    df, _ = generate_all_features(df)
    eval_df = df.iloc[8640:].reset_index(drop=True)
    
    is_settle_bar = eval_df["datetime"].dt.minute == 0
    is_settle_hour = eval_df["datetime"].dt.hour.isin([0, 8, 16])
    
    thresholds = [
        ("01. [기존 상대만] RSI <= 5% (절대값 없음)", (eval_df["feat_funding_rsi_30d"] <= 0.05)),
        ("02. [RSI + 단순음수] RSI <= 5% & FR <= 0.0%", (eval_df["feat_funding_rsi_30d"] <= 0.05) & (eval_df["fundingRate"] <= 0.0)),
        ("03. [RSI + 약과열]   RSI <= 5% & FR <= -0.005%", (eval_df["feat_funding_rsi_30d"] <= 0.05) & (eval_df["fundingRate"] <= -0.00005)),
        ("04. [RSI + 진성과열] RSI <= 5% & FR <= -0.010%", (eval_df["feat_funding_rsi_30d"] <= 0.05) & (eval_df["fundingRate"] <= -0.00010)),
        ("05. [RSI + 극단과열] RSI <= 5% & FR <= -0.020%", (eval_df["feat_funding_rsi_30d"] <= 0.05) & (eval_df["fundingRate"] <= -0.00020)),
        ("06. [순수 절대과열] RSI 무관 & FR <= -0.010%", (eval_df["fundingRate"] <= -0.00010)),
        ("07. [순수 극단과열] RSI 무관 & FR <= -0.020%", (eval_df["fundingRate"] <= -0.00020))
    ]
    
    engine = TripleBarrierEngine(fee_maker_pct=0.02, fee_taker_pct=0.05, slippage_pct=0.02)
    
    print(f"\n=========================================================================================")
    print(f"🔬 [{dataset_name}] 펀딩비 절대값 임계치별 Mode 2.1 스윙 백테스트 (SL -4%, No TP, 24h)")
    print(f"=========================================================================================")
    header = "{:<44} | {:<6} | {:<7} | {:<11} | {:<7} | {:<8}"
    row = "{:<44} | {:>6} | {:>6.1f}% | {:>10.2f}% | {:>7.2f} | {:>7.2f}%"
    print(header.format("펀딩비 진입 조건", "거래수", "승률", "누적수익률", "손익비", "최대낙폭(MDD)"))
    print("-" * 95)
    
    for label, cond in thresholds:
        sig = is_settle_bar & is_settle_hour & cond
        _, m = engine.run_backtest(eval_df, sig, tp_pct=999.0, sl_pct=4.0, max_horizon_bars=288)
        print(row.format(
            label,
            f"{m['total_trades']}회",
            m["win_rate"],
            m["cumulative_return"],
            m["profit_factor"],
            m["mdd"]
        ))


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    # 1. 2021년 OOS
    run_threshold_grid_on_dataset(data_dir / "BTCUSDT_5m_2021.csv", "2021년 불장 OOS 구간")
    # 2. 2022~2024년 인샘플
    run_threshold_grid_on_dataset(data_dir / "BTCUSDT_5m_2022_2024.csv", "2022~2024년 약세/횡보 인샘플 구간")


if __name__ == "__main__":
    main()
