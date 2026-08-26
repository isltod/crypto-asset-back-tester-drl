"""
flare.research.inspect_two_periods

사용자가 지목한 2대 구간의 실제 거래 데이터 정밀 분석
1. [2022년 11월 ~ 2023년 1월]: FTX 파산 사태 대폭락 구간의 체결 내역, 연속 손절 및 MDD 원인
2. [2023년 11월 ~ 2024년 12월]: 현물 ETF 불장 구간의 거래 빈도, 펀딩비 환경 및 플랫 횡보 원인
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.backtest.test_multi_position_equal_weight import run_equal_weight_multi_position


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    
    # 2.5배 레버리지 기준 실전 거래 데이터 추출
    res = run_equal_weight_multi_position(symbols, data_dir, initial_capital=1_000_000.0, leverage=2.5, allocation_ratio=0.80)
    trades_df = res["trades_df"].copy()
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
    
    # ==========================================
    # 1. 2022년 11월 ~ 2023년 1월 분석 (FTX 사태)
    # ==========================================
    p1_start = pd.Timestamp("2022-11-01 00:00:00", tz="UTC")
    p1_end = pd.Timestamp("2023-01-31 23:59:59", tz="UTC")
    p1_trades = trades_df[(trades_df["entry_time"] >= p1_start) & (trades_df["entry_time"] <= p1_end)].copy()
    
    print("=" * 115)
    print("🔬 [구간 1 정밀 분석] 2022년 11월 ~ 2023년 1월 (FTX 파산 및 크립토 윈터 최악의 투매장)")
    print("=" * 115)
    print(f"[*] 총 거래 횟수: {len(p1_trades)}회")
    print(f"[*] 승률: {(p1_trades['net_pnl']>0).mean()*100:.1f}% ({len(p1_trades[p1_trades['net_pnl']>0])}승 {len(p1_trades[p1_trades['net_pnl']<=0])}패)")
    print(f"[*] 구간 누적 실현 손익: ₩{p1_trades['net_pnl'].sum():>+10,.0f}")
    print(f"[*] 손절(SL) 발생 횟수: {len(p1_trades[p1_trades['exit_reason']=='SL'])}회 / 만기(TIMEOUT): {len(p1_trades[p1_trades['exit_reason']=='TIMEOUT'])}회")
    print("-" * 115)
    print("📊 [종목별 손익 상세]")
    for sym, grp in p1_trades.groupby("symbol"):
        wr = (grp["net_pnl"] > 0).mean() * 100.0
        sl_cnt = (grp["exit_reason"] == "SL").sum()
        print(f"    • {sym:<8}: {len(grp):>2}회 거래 | 승률 {wr:>5.1f}% | SL 손절 {sl_cnt:>2}회 | 누적 손익 ₩{grp['net_pnl'].sum():>+10,.0f}")
    print("-" * 115)
    print("📋 [2022년 11월 FTX 주간(11/06~11/15) 주요 거래 내역]")
    ftx_week = p1_trades[(p1_trades["entry_time"] >= "2022-11-06") & (p1_trades["entry_time"] <= "2022-11-20")]
    for _, r in ftx_week.iterrows():
        print(f"    - [{r['entry_time'].strftime('%Y-%m-%d %H:%M')}] {r['symbol']:<8} | 사유: {r['exit_reason']:<7} | 손익률: {r['return_pct']:>+6.2f}% | 손익금: ₩{r['net_pnl']:>+9,.0f}")
        
    # ==========================================
    # 2. 2023년 11월 ~ 2024년 12월 분석 (ETF 불장)
    # ==========================================
    p2_start = pd.Timestamp("2023-11-01 00:00:00", tz="UTC")
    p2_end = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
    p2_trades = trades_df[(trades_df["entry_time"] >= p2_start) & (trades_df["entry_time"] <= p2_end)].copy()
    
    print("\n" + "=" * 115)
    print("🔬 [구간 2 정밀 분석] 2023년 11월 ~ 2024년 12월 (비트코인 현물 ETF 승인 및 대세 불장)")
    print("=" * 115)
    print(f"[*] 14개월간 총 거래 횟수: {len(p2_trades)}회 (월평균 겨우 {len(p2_trades)/14.0:.1f}회!)")
    print(f"[*] 승률: {(p2_trades['net_pnl']>0).mean()*100:.1f}% ({len(p2_trades[p2_trades['net_pnl']>0])}승 {len(p2_trades[p2_trades['net_pnl']<=0])}패)")
    print(f"[*] 구간 누적 실현 손익: ₩{p2_trades['net_pnl'].sum():>+10,.0f}")
    print("-" * 115)
    print("📊 [종목별 손익 및 거래 횟수]")
    for sym, grp in p2_trades.groupby("symbol"):
        wr = (grp["net_pnl"] > 0).mean() * 100.0
        print(f"    • {sym:<8}: 14개월간 총 {len(grp):>2}회 거래 (월평균 {len(grp)/14.0:.1f}회) | 승률 {wr:>5.1f}% | 누적 손익 ₩{grp['net_pnl'].sum():>+10,.0f}")
    print("=" * 115)


if __name__ == "__main__":
    main()
