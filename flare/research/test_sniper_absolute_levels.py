"""
flare.research.test_sniper_absolute_levels

Mode 1.1 (FLARE-Sniper-Pure, 4h)에 대해
펀딩비 절대값 임계치(FR Absolute Thresholds) 구간별 정밀 백테스트
- 조건: 펀딩비 절대값 조건 & 5분봉 청산 꼬리 (거래량 3x + 아래꼬리 55%+)
- 청산 룰: SL -3.0% / No TP / 4시간 만기 종가 청산
- 검증 구간: 2022~2024년 (2.5년 인샘플) 및 2021년 (1년 OOS)
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


def run_sniper_threshold_test(klines_path: Path, dataset_label: str):
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
    
    is_wick_spike = (eval_df["feat_is_lower_wick_spike"] == 1.0)
    
    thresholds = [
        ("01. [기존 상대만] RSI <= 10% (절대값 없음)", (eval_df["feat_funding_rsi_30d"] <= 0.10)),
        ("02. [단순 음수]   FR <= 0.0000%", (eval_df["fundingRate"] <= 0.0)),
        ("03. [미세 숏쏠림] FR <= -0.0025%", (eval_df["fundingRate"] <= -0.000025)),
        ("04. [약한 과열]   FR <= -0.0050%", (eval_df["fundingRate"] <= -0.000050)),
        ("05. [중간 과열]   FR <= -0.0075%", (eval_df["fundingRate"] <= -0.000075)),
        ("06. [진성 과열]   FR <= -0.0100%", (eval_df["fundingRate"] <= -0.000100)),
        ("07. [극단 과열]   FR <= -0.0200%", (eval_df["fundingRate"] <= -0.000200))
    ]
    
    engine = TripleBarrierEngine(fee_maker_pct=0.02, fee_taker_pct=0.05, slippage_pct=0.02)
    
    print(f"\n===================================================================================================")
    print(f"🔬 [{dataset_label}] Mode 1.1 스나이퍼 펀딩비 절대값 구간별 백테스트 (SL -3%, No TP, 4h)")
    print(f"===================================================================================================")
    header = "{:<44} | {:<6} | {:<7} | {:<11} | {:<7} | {:<8} | {:<15}"
    row = "{:<44} | {:>6} | {:>6.1f}% | {:>10.2f}% | {:>7.2f} | {:>7.2f}% | TP:{:<2} SL:{:<2} TO:{:<2}"
    print(header.format("펀딩비 진입 조건", "거래수", "승률", "누적수익률", "손익비", "최대낙폭(MDD)", "청산 분포"))
    print("-" * 105)
    
    for label, cond in thresholds:
        sig = cond & is_wick_spike
        _, m = engine.run_backtest(eval_df, sig, tp_pct=999.0, sl_pct=3.0, max_horizon_bars=48)
        print(row.format(
            label,
            f"{m['total_trades']}회",
            m["win_rate"],
            m["cumulative_return"],
            m["profit_factor"],
            m["mdd"],
            m["tp_count"],
            m["sl_count"],
            m["timeout_count"]
        ))


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    # 1. 원래 백테스트를 하던 2022~2024년 2.5년 인샘플
    run_sniper_threshold_test(data_dir / "BTCUSDT_5m_2022_2024.csv", "2022~2024년 (2.5년 메인 인샘플)")
    # 2. 2021년 불장 OOS
    run_sniper_threshold_test(data_dir / "BTCUSDT_5m_2021.csv", "2021년 불장 (1년 OOS)")


if __name__ == "__main__":
    main()
