"""
flare.backtest.test_swing_2d_grid

Mode 2.1 (FLARE-Swing-Pure, 24h)에 대해
익절 TP [1, 4, 8, 12, 16]% x 손절 SL [1, 4, 8, 12, 16]%
총 25개 조합 2D 그리드 서치 백테스트 및 성과 매트릭스 산출 모듈
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
    
    # 8시간 정산 시점 (00:00, 08:00, 16:00 UTC) 하위 5% 과열 신호
    is_settle_bar = eval_df["datetime"].dt.minute == 0
    is_settle_hour = eval_df["datetime"].dt.hour.isin([0, 8, 16])
    sig_swing_pure = (
        is_settle_bar & is_settle_hour & 
        (eval_df["feat_funding_rsi_30d"] <= 0.05)
    )
    
    engine = TripleBarrierEngine(fee_maker_pct=0.02, fee_taker_pct=0.05, slippage_pct=0.02)
    
    tp_levels = [1.0, 4.0, 8.0, 12.0, 16.0]
    sl_levels = [1.0, 4.0, 8.0, 12.0, 16.0]
    
    records = []
    
    print("=" * 115)
    print("🔬 [Mode 2.1: FLARE-Swing-Pure] TP x SL (5x5 = 25개 조합) 2D 그리드 서치 실행 중...")
    print("=" * 115)
    
    for tp in tp_levels:
        for sl in sl_levels:
            trades, metrics = engine.run_backtest(
                eval_df,
                sig_swing_pure,
                tp_pct=tp,
                sl_pct=sl,
                max_horizon_bars=288 # 24시간
            )
            records.append({
                "tp": tp,
                "sl": sl,
                "total_trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "return_pct": metrics["cumulative_return"],
                "profit_factor": metrics["profit_factor"],
                "mdd": metrics["mdd"],
                "sharpe": metrics["sharpe_ratio"],
                "tp_count": metrics["tp_count"],
                "sl_count": metrics["sl_count"],
                "timeout_count": metrics["timeout_count"]
            })
            
    res_df = pd.DataFrame(records)
    
    # 1. 2D 총 누적수익률 (%) 피벗 매트릭스
    pnl_pivot = res_df.pivot(index="sl", columns="tp", values="return_pct")
    win_pivot = res_df.pivot(index="sl", columns="tp", values="win_rate")
    mdd_pivot = res_df.pivot(index="sl", columns="tp", values="mdd")
    trades_pivot = res_df.pivot(index="sl", columns="tp", values="total_trades")
    
    print("\n📈 [1] 총 누적수익률 (%) 매트릭스 (행: SL 손절폭 / 열: TP 익절폭):")
    print("-" * 75)
    print(pnl_pivot.round(2).to_string())
    
    print("\n🎯 [2] 승률 (%) 매트릭스:")
    print("-" * 75)
    print(win_pivot.round(1).to_string())
    
    print("\n🛡️ [3] 최대낙폭 (MDD %) 매트릭스 (작을수록 우수):")
    print("-" * 75)
    print(mdd_pivot.round(2).to_string())

    print("\n📊 [4] 총 거래 횟수 매트릭스:")
    print("-" * 75)
    print(trades_pivot.to_string())
    
    print("\n" + "=" * 115)
    print("🏆 [Top 5 최고 수익률 조합]")
    print("-" * 115)
    top5 = res_df.sort_values("return_pct", ascending=False).head(5)
    for i, r in top5.reset_index(drop=True).iterrows():
        print(f"{i+1}위: TP +{r['tp']:.0f}% / SL -{r['sl']:.0f}% ➔ 누적수익률: {r['return_pct']:>+6.2f}% | 승률: {r['win_rate']:>4.1f}% | MDD: {r['mdd']:>5.2f}% | 손익비: {r['profit_factor']:>4.2f} | 거래수: {r['total_trades']}회 (TP:{r['tp_count']} SL:{r['sl_count']} TO:{r['timeout_count']})")
    print("=" * 115)


if __name__ == "__main__":
    main()
