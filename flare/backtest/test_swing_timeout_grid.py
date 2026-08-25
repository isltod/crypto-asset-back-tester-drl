"""
flare.backtest.test_swing_timeout_grid

Mode 2.1 베이스라인(No TP + SL -4.0%)에 대해
Timeout 만기 시간을 [8시간, 16시간, 24시간, 32시간]으로 변경하며
거래수, 승률, 누적수익률, 손익비, MDD, 샤프지수를 정밀 대조하는 스크립트
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.data.features import generate_all_features
from flare.backtest.engine import TripleBarrierEngine


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    klines_file = data_dir / "BTCUSDT_5m_2022_2024.csv"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
    print(f"[*] 5분봉 데이터 로드: {klines_file.name}...")
    df = pd.read_csv(klines_file)
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
    sig_swing_pure = (
        is_settle_bar & is_settle_hour & 
        (eval_df["feat_funding_rsi_30d"] <= 0.05)
    )
    
    engine = TripleBarrierEngine(fee_maker_pct=0.02, fee_taker_pct=0.05, slippage_pct=0.02)
    
    timeout_hours = [8, 16, 24, 32]
    results = []
    
    print("=" * 115)
    print("🔬 [Mode 2.1: FLARE-Swing 베이스라인] Timeout 만기 시간 [8h, 16h, 24h, 32h] 백테스트 대조")
    print("=" * 115)
    
    for h in timeout_hours:
        bars = h * 12 # 5분봉 개수 (1시간당 12봉)
        trades, m = engine.run_backtest(
            eval_df,
            sig_swing_pure,
            tp_pct=999.0, # No TP
            sl_pct=4.0,   # SL -4.0%
            max_horizon_bars=bars
        )
        results.append({
            "timeout_str": f"{h}시간 ({bars}봉)",
            "hours": h,
            **m
        })
        
    res_df = pd.DataFrame(results)
    
    header_fmt = "{:<16} | {:<6} | {:<8} | {:<12} | {:<7} | {:<8} | {:<8} | {:<8} | {:<15}"
    row_fmt = "{:<16} | {:>6} | {:>7.1f}% | {:>11.2f}% | {:>7.2f} | {:>7.2f}% | {:>8.2f} | {:>+7.2f}% | TP:{:<2} SL:{:<2} TO:{:<2}"
    
    print(header_fmt.format("만기 보유 시간", "거래수", "승률", "총 누적수익률", "손익비", "최대낙폭(MDD)", "샤프지수", "건당평균", "청산사유 분포"))
    print("-" * 115)
    for _, r in res_df.iterrows():
        print(row_fmt.format(
            r["timeout_str"],
            f"{r['total_trades']}회",
            r["win_rate"],
            r["cumulative_return"],
            r["profit_factor"],
            r["mdd"],
            r["sharpe_ratio"],
            r["avg_return_per_trade"],
            r["tp_count"],
            r["sl_count"],
            r["timeout_count"]
        ))
    print("=" * 115)


if __name__ == "__main__":
    main()
