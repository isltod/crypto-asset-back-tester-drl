"""
flare.research.test_oos_with_absolute_guard

상대 순위(RSI <= 0.05)에 '진짜 음수 펀딩비(fundingRate < 0)' 절대값 가드레일을 결합했을 때
2021년 OOS (1년) 및 2022~2024년 (2.5년) 백테스트 성과 검증
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


def test_year(klines_path: Path, year_label: str):
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
    
    # 1. 기존 조건 (상대 순위만 적용)
    sig_old = is_settle_bar & is_settle_hour & (eval_df["feat_funding_rsi_30d"] <= 0.05)
    
    # 2. 개선 조건 (상대 순위 하위 5% + 진짜 음수 펀딩비 fundingRate < 0)
    sig_guarded = is_settle_bar & is_settle_hour & (eval_df["feat_funding_rsi_30d"] <= 0.05) & (eval_df["fundingRate"] < 0)
    
    engine = TripleBarrierEngine(fee_maker_pct=0.02, fee_taker_pct=0.05, slippage_pct=0.02)
    
    _, m_old = engine.run_backtest(eval_df, sig_old, tp_pct=999.0, sl_pct=4.0, max_horizon_bars=288)
    _, m_guarded = engine.run_backtest(eval_df, sig_guarded, tp_pct=999.0, sl_pct=4.0, max_horizon_bars=288)
    
    print(f"\n📊 [{year_label}] Mode 2.1 스윙 백테스트 비교 (SL -4%, No TP, 24h)")
    print("-" * 85)
    print(f"1) 기존 (상대 순위만) ➔ 거래수: {m_old['total_trades']:>3}회 | 승률: {m_old['win_rate']:>5.1f}% | 수익률: {m_old['cumulative_return']:>+6.2f}% | MDD: {m_old['mdd']:>5.2f}%")
    print(f"2) 가드레일 (진짜음수) ➔ 거래수: {m_guarded['total_trades']:>3}회 | 승률: {m_guarded['win_rate']:>5.1f}% | 수익률: {m_guarded['cumulative_return']:>+6.2f}% | MDD: {m_guarded['mdd']:>5.2f}% 🛡️")


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    print("=" * 85)
    print("🔬 [가드레일 검증] 상대 순위(RSI) + '진짜 음수 펀딩비(FR < 0)' 결합 효과 검증")
    print("=" * 85)
    
    test_year(data_dir / "BTCUSDT_5m_2021.csv", "2021년 불장 (미지의 OOS 구간)")
    test_year(data_dir / "BTCUSDT_5m_2022_2024.csv", "2022~2024년 (약세/횡보 인샘플 구간)")
    print("=" * 85)


if __name__ == "__main__":
    main()
