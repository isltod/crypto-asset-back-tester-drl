"""
flare.backtest.test_oos_2021

미지의 2021년 단독 Out-of-Sample (OOS) 실전 복리 백테스터
- 기간: 2021-01-01 ~ 2021-12-31 (1년 치 완전 미지의 불장 데이터)
- 검증 대상: 2022~2024년 데이터로 확정한 [Mode 2.1 Swing (SL -4%, 24h) + Mode 1.1 Sniper (SL -3%, 4h)]
- 1계좌 1포지션 실전 복리(초기자본 100만 원) 시계열 검증
- 수수료/슬리피지 100% 실시간 차감
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.data.features import generate_all_features


@dataclass
class Position:
    mode: str
    entry_time: pd.Timestamp
    entry_price: float
    position_size: float
    margin_cost: float
    leverage: float
    sl_price: float
    max_bars: int
    bars_held: int = 0


def run_oos_2021_backtest(
    initial_capital: float = 1_000_000.0,
    swing_leverage: float = 2.0,
    sniper_leverage: float = 2.5,
    allocation_ratio: float = 0.80, # 80% 투입, 20% 현금 보관 (안전형 B)
    fee_taker: float = 0.0005,
    slippage: float = 0.0002
):
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    klines_2021_file = data_dir / "BTCUSDT_5m_2021.csv"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
    if not klines_2021_file.exists():
        print(f"[!] 2021년 5분봉 파일이 아직 없습니다: {klines_2021_file.name}")
        return
        
    print(f"[*] [OOS 2021] 5분봉 캔들 데이터 로드 중: {klines_2021_file.name}...")
    df = pd.read_csv(klines_2021_file)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    print(f"[*] 펀딩비 데이터 매핑 중...")
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    df = pd.merge_asof(df, df_fr[["fundingTime", "fundingRate"]], left_on="datetime", right_on="fundingTime", direction="backward")
    df["fundingRate"] = df["fundingRate"].ffill().fillna(0.0001)
    
    print(f"[*] 27종 통합 피처 생성 중...")
    df, _ = generate_all_features(df)
    
    # 2021년 30일 웜업 후 평가 시작 (2021-01-31부터)
    eval_df = df.iloc[8640:].reset_index(drop=True)
    
    is_settle_bar = eval_df["datetime"].dt.minute == 0
    is_settle_hour = eval_df["datetime"].dt.hour.isin([0, 8, 16])
    sig_swing = (is_settle_bar & is_settle_hour & (eval_df["feat_funding_rsi_30d"] <= 0.05)).values
    sig_sniper = ((eval_df["feat_funding_rsi_30d"] <= 0.10) & (eval_df["feat_is_lower_wick_spike"] == 1.0)).values
    
    highs = eval_df["high"].values
    lows = eval_df["low"].values
    closes = eval_df["close"].values
    datetimes = eval_df["datetime"].values
    n_bars = len(eval_df)
    
    cash = initial_capital
    position: Optional[Position] = None
    trade_logs = []
    equity_curve = []
    
    print(f"[*] [OOS 2021] 실전 복리 시뮬레이션 가동 ({eval_df['datetime'].iloc[0]} ~ {eval_df['datetime'].iloc[-1]}, 총 {n_bars:,}개 5분봉)...")
    
    for i in range(n_bars):
        current_time = pd.Timestamp(datetimes[i])
        h = highs[i]
        l = lows[i]
        c = closes[i]
        
        # 1. 청산 체크
        if position is not None:
            position.bars_held += 1
            exit_price = None
            exit_reason = None
            
            if l <= position.sl_price:
                exit_price = position.sl_price * (1.0 - slippage)
                exit_reason = "SL"
            elif position.bars_held >= position.max_bars:
                exit_price = c * (1.0 - slippage)
                exit_reason = "TIMEOUT"
                
            if exit_price is not None:
                raw_pnl = (exit_price - position.entry_price) * position.position_size
                exit_fee = (exit_price * position.position_size) * fee_taker
                net_trade_pnl = raw_pnl - exit_fee
                
                cash += position.margin_cost + net_trade_pnl
                ret_on_margin = (net_trade_pnl / position.margin_cost) * 100.0
                
                trade_logs.append({
                    "mode": position.mode,
                    "entry_time": position.entry_time,
                    "exit_time": current_time,
                    "entry_price": position.entry_price,
                    "exit_price": exit_price,
                    "leverage": position.leverage,
                    "net_pnl": net_trade_pnl,
                    "return_pct": ret_on_margin,
                    "exit_reason": exit_reason,
                    "hold_bars": position.bars_held,
                    "balance": cash
                })
                position = None
                
        # 2. 신규 진입 (1계좌 1포지션)
        if position is None:
            total_equity = cash
            trade_margin = total_equity * allocation_ratio
            
            if sig_swing[i]:
                entry_p = c * (1.0 + slippage)
                entry_fee = (entry_p * (trade_margin * swing_leverage / entry_p)) * fee_taker
                usable_margin = trade_margin - entry_fee
                pos_size = (usable_margin * swing_leverage) / entry_p
                sl_p = entry_p * (1.0 - 0.04) # SL -4.0%
                
                position = Position(
                    mode="SWING",
                    entry_time=current_time,
                    entry_price=entry_p,
                    position_size=pos_size,
                    margin_cost=usable_margin,
                    leverage=swing_leverage,
                    sl_price=sl_p,
                    max_bars=288 # 24시간
                )
                cash -= trade_margin
                
            elif sig_sniper[i]:
                entry_p = c * (1.0 + slippage)
                entry_fee = (entry_p * (trade_margin * sniper_leverage / entry_p)) * fee_taker
                usable_margin = trade_margin - entry_fee
                pos_size = (usable_margin * sniper_leverage) / entry_p
                sl_p = entry_p * (1.0 - 0.03) # SL -3.0%
                
                position = Position(
                    mode="SNIPER",
                    entry_time=current_time,
                    entry_price=entry_p,
                    position_size=pos_size,
                    margin_cost=usable_margin,
                    leverage=sniper_leverage,
                    sl_price=sl_p,
                    max_bars=48 # 4시간
                )
                cash -= trade_margin
                
        current_equity = cash if position is None else cash + position.margin_cost + ((c - position.entry_price) * position.position_size)
        equity_curve.append(current_equity)
        
    trades_df = pd.DataFrame(trade_logs)
    final_balance = cash if position is None else cash + position.margin_cost
    cum_ret_pct = (final_balance - initial_capital) / initial_capital * 100.0
    
    eq_series = pd.Series(equity_curve)
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak * 100.0
    mdd = abs(dd.min())
    
    swing_trades = trades_df[trades_df["mode"] == "SWING"]
    sniper_trades = trades_df[trades_df["mode"] == "SNIPER"]
    
    print("\n" + "=" * 115)
    print("🎯 [FLARE] 2021년 단독 Out-of-Sample (OOS) 실전 복리 백테스트 성적표 (불장 단독 검증)")
    print("=" * 115)
    print(f"[*] 검증 구간         : 2021-01-31 ~ 2021-12-31 (11개월, 2021년 불장)")
    print(f"[*] 초기 시작 자본금   : ₩{initial_capital:,.0f} (100만 원)")
    print(f"[*] 1년 뒤 최종 계좌 잔고: ₩{final_balance:,.0f} ({final_balance/initial_capital:.2f}배 증식! 🚀)")
    print(f"[*] 실전 복리 누적수익률: {cum_ret_pct:>+8.2f}% (수수료/슬리피지 100% 실시간 차감)")
    print(f"[*] 계좌 최대 낙폭(MDD) : {mdd:>6.2f}% 🛡️")
    print(f"[*] 총 실행 거래 횟수  : {len(trades_df)}회 (월평균 {len(trades_df)/11.0:.1f}회)")
    print(f"[*] 통산 승률         : {(trades_df['net_pnl']>0).mean()*100:.1f}%")
    print("-" * 115)
    print("📊 [모드별 기여 내역]")
    print(f"    - 🟢 Mode 2.1 (Swing 2.0x)  : {len(swing_trades)}회 | 승률 {(swing_trades['net_pnl']>0).mean()*100:.1f}% | 손익 ₩{swing_trades['net_pnl'].sum():>+10,.0f}")
    print(f"    - ⚡ Mode 1.1 (Sniper 2.5x) : {len(sniper_trades)}회 | 승률 {(sniper_trades['net_pnl']>0).mean()*100:.1f}% | 손익 ₩{sniper_trades['net_pnl'].sum():>+10,.0f}")
    print("=" * 115)


if __name__ == "__main__":
    run_oos_2021_backtest()
