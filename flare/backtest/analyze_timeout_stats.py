"""
flare.backtest.analyze_timeout_stats

Mode 2.1 (SL -4.0% 고정) 조건에서
각 TP 수준별로 '24시간 만기(Timeout) 종가 청산된 거래'들만의
총 횟수, 이익 횟수, 손실 횟수, 승률, 평균 손익(%), 누적 손익(%)을 정밀 분석하는 스크립트
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
    
    tp_levels = [1.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0]
    fixed_sl = 4.0
    
    results = []
    
    print("=" * 115)
    print("🔬 [SL -4.0% 고정] TP 수준별 '24시간 만기(Timeout) 종가 청산 거래' 정밀 분석")
    print("=" * 115)
    
    for tp in tp_levels:
        trade_df, metrics = engine.run_backtest(
            eval_df,
            sig_swing_pure,
            tp_pct=tp,
            sl_pct=fixed_sl,
            max_horizon_bars=288 # 24시간
        )
        
        # Timeout 거래만 필터링
        to_trades = trade_df[trade_df["exit_reason"] == "TIMEOUT"]
        
        to_count = len(to_trades)
        if to_count > 0:
            rets = to_trades["return_pct"]
            wins = rets[rets > 0]
            losses = rets[rets <= 0]
            
            win_cnt = len(wins)
            loss_cnt = len(losses)
            win_rate = (win_cnt / to_count) * 100.0
            avg_pnl = rets.mean()
            total_pnl = rets.sum()
            avg_win = wins.mean() if win_cnt > 0 else 0.0
            avg_loss = losses.mean() if loss_cnt > 0 else 0.0
        else:
            win_cnt = 0
            loss_cnt = 0
            win_rate = 0.0
            avg_pnl = 0.0
            total_pnl = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            
        results.append({
            "tp_pct": tp,
            "total_trades": metrics["total_trades"],
            "tp_count": metrics["tp_count"],
            "sl_count": metrics["sl_count"],
            "to_count": to_count,
            "to_win_cnt": win_cnt,
            "to_loss_cnt": loss_cnt,
            "to_win_rate": win_rate,
            "to_avg_pnl": avg_pnl,
            "to_total_pnl": total_pnl,
            "to_avg_win": avg_win,
            "to_avg_loss": avg_loss
        })
        
    res_df = pd.DataFrame(results)
    
    header_fmt = "{:<10} | {:<6} | {:<12} | {:<10} | {:<10} | {:<10} | {:<12} | {:<12}"
    row_fmt = "TP +{:<4.0f}%  | {:>4}회 | TP:{:<2} SL:{:<2} TO:{:<2} | {:>4}회 ({:>4.1f}%) | {:>4}회 ({:>4.1f}%) | {:>5.1f}%     | {:>+10.2f}% | {:>+10.2f}%"
    
    print(header_fmt.format("TP 수준", "총거래", "전체 청산분포", "만기 이익수", "만기 손실수", "만기 승률", "만기 평균손익", "만기 누적손익"))
    print("-" * 115)
    
    for _, r in res_df.iterrows():
        to_win_pct = (r["to_win_cnt"] / r["to_count"] * 100.0) if r["to_count"] > 0 else 0.0
        to_loss_pct = (r["to_loss_cnt"] / r["to_count"] * 100.0) if r["to_count"] > 0 else 0.0
        
        print(row_fmt.format(
            r["tp_pct"],
            r["total_trades"],
            f"{r['tp_count']}", f"{r['sl_count']}", f"{r['to_count']}",
            r["to_win_cnt"], to_win_pct,
            r["to_loss_cnt"], to_loss_pct,
            r["to_win_rate"],
            r["to_avg_pnl"],
            r["to_total_pnl"]
        ))
    print("=" * 115)


if __name__ == "__main__":
    main()
