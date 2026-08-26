"""
flare.backtest.analyze_overlap

Mode 2.1 (FLARE-Swing, 85회)과 Mode 1.1 (FLARE-Sniper, 101회) 간의
포지션 보유 시간 [entry_time ~ exit_time] 1:1 시간대 중첩(Overlap) 정밀 분석
- 완전 중첩된 거래 수 (Overlap Trades)
- 스윙 없이 스나이퍼만 단독 발생한 거래 수 (Independent Sniper)
- 스나이퍼 없이 스윙만 단독 발생한 거래 수 (Independent Swing)
- 두 모드 통합 시 실질 순수 총 거래 기회 (Total Unique Trades)
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
    
    engine = TripleBarrierEngine(fee_maker_pct=0.02, fee_taker_pct=0.05, slippage_pct=0.02)
    
    # 1. Mode 2.1 Swing 거래 실행 (SL -4%, No TP, 24h)
    is_settle_bar = eval_df["datetime"].dt.minute == 0
    is_settle_hour = eval_df["datetime"].dt.hour.isin([0, 8, 16])
    sig_swing = is_settle_bar & is_settle_hour & (eval_df["feat_funding_rsi_30d"] <= 0.05)
    trades_swing_df, m_swing = engine.run_backtest(eval_df, sig_swing, tp_pct=999.0, sl_pct=4.0, max_horizon_bars=288)
    
    # 2. Mode 1.1 Sniper 거래 실행 (SL -3%, No TP, 4h)
    sig_sniper = (eval_df["feat_funding_rsi_30d"] <= 0.10) & (eval_df["feat_is_lower_wick_spike"] == 1.0)
    trades_sniper_df, m_sniper = engine.run_backtest(eval_df, sig_sniper, tp_pct=999.0, sl_pct=3.0, max_horizon_bars=48)
    
    # 3. 시간대 중첩(Overlap) 정밀 대조
    # Sniper 거래가 Swing 거래 보유 구간 [entry, exit]에 포함되는지 체크
    sniper_overlap_flags = []
    
    for _, sn_trade in trades_sniper_df.iterrows():
        sn_start = sn_trade["entry_time"]
        sn_end = sn_trade["exit_time"]
        
        # 스윙 거래 중 겹치는 거래가 있는지 확인
        # (sn_start <= sw_end) and (sn_end >= sw_start)
        overlaps = trades_swing_df[
            (sn_start <= trades_swing_df["exit_time"]) & 
            (sn_end >= trades_swing_df["entry_time"])
        ]
        sniper_overlap_flags.append(len(overlaps) > 0)
        
    trades_sniper_df["is_overlap_with_swing"] = sniper_overlap_flags
    
    # Swing 거래 중 Sniper와 겹친 거래 확인
    swing_overlap_flags = []
    for _, sw_trade in trades_swing_df.iterrows():
        sw_start = sw_trade["entry_time"]
        sw_end = sw_trade["exit_time"]
        overlaps = trades_sniper_df[
            (sw_start <= trades_sniper_df["exit_time"]) & 
            (sw_end >= trades_sniper_df["entry_time"])
        ]
        swing_overlap_flags.append(len(overlaps) > 0)
        
    trades_swing_df["is_overlap_with_sniper"] = swing_overlap_flags
    
    # 통계 산출
    total_swing = len(trades_swing_df)
    total_sniper = len(trades_sniper_df)
    
    sniper_overlap_cnt = trades_sniper_df["is_overlap_with_swing"].sum()
    sniper_independent_cnt = total_sniper - sniper_overlap_cnt
    
    swing_overlap_cnt = trades_swing_df["is_overlap_with_sniper"].sum()
    swing_independent_cnt = total_swing - swing_overlap_cnt
    
    # 독립된 총 고유 거래 기회 (스윙 85회 + 스윙과 안 겹친 순수 스나이퍼 거래)
    total_unique_events = total_swing + sniper_independent_cnt
    
    print("=" * 95)
    print("🔬 [FLARE] Mode 2.1(스윙 85회) vs Mode 1.1(스나이퍼 101회) 포지션 중첩 정밀 대조 보고서")
    print("=" * 95)
    print(f"[*] Mode 2.1 (Swing 24h) 총 거래수   : {total_swing}회")
    print(f"[*] Mode 1.1 (Sniper 4h) 총 거래수  : {total_sniper}회")
    print("-" * 95)
    print("📊 [1] 스나이퍼(Mode 1.1, 총 101회) 관점:")
    print(f"    - 🔴 스윙 포지션과 겹쳐서 발생한 거래   : {sniper_overlap_cnt}회 ({sniper_overlap_cnt/total_sniper*100:.1f}%)")
    print(f"    - 🟢 스윙 없이 '단독'으로 발생한 순수 거래: {sniper_independent_cnt}회 ({sniper_independent_cnt/total_sniper*100:.1f}%) 🏆")
    print("-" * 95)
    print("📊 [2] 스윙(Mode 2.1, 총 85회) 관점:")
    print(f"    - 🔴 스나이퍼와 겹친 스윙 거래         : {swing_overlap_cnt}회 ({swing_overlap_cnt/total_swing*100:.1f}%)")
    print(f"    - 🟢 스나이퍼 없이 '단독'으로 진행된 스윙: {swing_independent_cnt}회 ({swing_independent_cnt/total_swing*100:.1f}%) 🏆")
    print("-" * 95)
    print(f"💎 [3] 두 모드를 통합했을 때의 '순수 고유 거래 기회' (Total Unique Events):")
    print(f"    ➔ 총 {total_unique_events}회 (월평균 약 {total_unique_events/28.5:.1f}회 / 2.5년간)")
    print("=" * 95)
    
    # 독립 발생 스나이퍼 거래의 성과 출력
    ind_sniper_trades = trades_sniper_df[~trades_sniper_df["is_overlap_with_swing"]]
    if len(ind_sniper_trades) > 0:
        ind_sniper_pnl = ind_sniper_trades["return_pct"].sum()
        ind_sniper_wr = (ind_sniper_trades["return_pct"] > 0).mean() * 100
        print(f"[*] 🌟 스윙과 겹치지 않은 '순수 독립 스나이퍼 49회'만의 성과:")
        print(f"    - 승률: {ind_sniper_wr:.1f}% | 총 누적수익률: {ind_sniper_pnl:>+6.2f}% | 건당 평균: {ind_sniper_trades['return_pct'].mean():>+5.2f}%")
        print("=" * 95)


if __name__ == "__main__":
    main()
