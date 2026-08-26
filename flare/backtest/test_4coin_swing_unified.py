"""
flare.backtest.test_4coin_swing_unified

DOGE 제외 4대 메이저 코인(BTC, ETH, SOL, XRP) Mode 2.1 스윙 전용
1계좌 1포지션 4개년 실전 복리 시계열 백테스트 (2021~2024)
- 초기 자본금: 100만 원
- 레버리지: 2.0x (80% 투입, 20% 현금 버퍼)
- 룰: SL -4.0% (SOL은 SL -6.0%) / No TP / 24시간 만기 종가 청산
- 수수료/슬리피지 100% 실시간 차감
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.backtest.test_multicoin_unified_account import load_coin_events, ActivePosition


def run_4coin_swing_unified():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"] # DOGE 완전 제외!
    
    print("[*] 4대 코인(BTC, ETH, SOL, XRP) 4개년 데이터 로드 중...")
    dfs = [load_coin_events(sym, data_dir) for sym in symbols]
    coin_dict = {df["symbol"].iloc[0]: df.set_index("datetime") for df in dfs}
    
    master_timeline = pd.date_range("2021-01-01 00:00:00+00:00", "2024-12-31 23:00:00+00:00", freq="1h", tz="UTC")
    
    initial_capital = 1_000_000.0
    leverage = 2.0
    allocation_ratio = 0.80
    fee_taker = 0.0005
    slippage = 0.0002
    
    cash = initial_capital
    active_pos: Optional[ActivePosition] = None
    trade_logs = []
    equity_curve = []
    
    print(f"[*] [DOGE 제외 4대 코인 스윙] 4개년 1계좌 1포지션 복리 시뮬레이션 가동...")
    
    for current_time in master_timeline:
        # 1. 청산 체크
        if active_pos is not None:
            sym_data = coin_dict[active_pos.symbol]
            if current_time in sym_data.index:
                row = sym_data.loc[current_time]
                h = row["high"]
                l = row["low"]
                c = row["close"]
                
                active_pos.bars_held += 1
                exit_price = None
                exit_reason = None
                
                if l <= active_pos.sl_price:
                    exit_price = active_pos.sl_price * (1.0 - slippage)
                    exit_reason = "SL"
                elif active_pos.bars_held >= active_pos.max_bars:
                    exit_price = c * (1.0 - slippage)
                    exit_reason = "TIMEOUT"
                    
                if exit_price is not None:
                    raw_pnl = (exit_price - active_pos.entry_price) * active_pos.position_size
                    exit_fee = (exit_price * active_pos.position_size) * fee_taker
                    net_trade_pnl = raw_pnl - exit_fee
                    
                    cash += active_pos.margin_cost + net_trade_pnl
                    ret_pct = (net_trade_pnl / active_pos.margin_cost) * 100.0
                    
                    trade_logs.append({
                        "symbol": active_pos.symbol,
                        "entry_time": active_pos.entry_time,
                        "exit_time": current_time,
                        "entry_price": active_pos.entry_price,
                        "exit_price": exit_price,
                        "net_pnl": net_trade_pnl,
                        "return_pct": ret_pct,
                        "exit_reason": exit_reason,
                        "balance": cash
                    })
                    active_pos = None
                    
        # 2. 무포지션 상태일 때만 4개 코인의 스윙 신호 탐색
        if active_pos is None:
            total_equity = cash
            trade_margin = total_equity * allocation_ratio
            
            for sym in symbols:
                sym_data = coin_dict[sym]
                if current_time in sym_data.index:
                    row = sym_data.loc[current_time]
                    if row["sig_swing"]:
                        c = row["close"]
                        entry_p = c * (1.0 + slippage)
                        entry_fee = (entry_p * (trade_margin * leverage / entry_p)) * fee_taker
                        usable_margin = trade_margin - entry_fee
                        pos_size = (usable_margin * leverage) / entry_p
                        sl_rate = 0.06 if sym == "SOLUSDT" else 0.04
                        sl_p = entry_p * (1.0 - sl_rate)
                        
                        active_pos = ActivePosition(
                            symbol=sym,
                            mode="SWING",
                            entry_time=current_time,
                            entry_price=entry_p,
                            position_size=pos_size,
                            margin_cost=usable_margin,
                            leverage=leverage,
                            sl_price=sl_p,
                            max_bars=24
                        )
                        cash -= trade_margin
                        break
                        
        current_eq = cash if active_pos is None else cash + active_pos.margin_cost
        equity_curve.append(current_eq)
        
    trades_df = pd.DataFrame(trade_logs)
    final_balance = cash if active_pos is None else cash + active_pos.margin_cost
    total_ret_pct = (final_balance - initial_capital) / initial_capital * 100.0
    
    eq_series = pd.Series(equity_curve)
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak * 100.0
    mdd = abs(dd.min())
    
    print("\n" + "=" * 115)
    print("🏆 [DOGE 제외 4대 코인] Mode 2.1 스윙 전용 1계좌 1포지션 실전 복리 백테스트 성적표 (2021~2024, 4개년)")
    print("   • 대상 종목: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT (DOGE 완전 제외 🛡️)")
    print("   • 조건: 안전 2배 레버리지 (2.0x) | 잔고의 80% 투입 | No TP | 24시간 만기 종가 청산")
    print("=" * 115)
    print(f"[*] 초기 시작 자본금   : ₩{initial_capital:,.0f} (100만 원)")
    print(f"[*] 4년 뒤 최종 계좌 잔고: ₩{final_balance:,.0f} (약 {final_balance/initial_capital:.2f}배 증식! 🚀)")
    print(f"[*] 실전 복리 총수익률 : {total_ret_pct:>+8.2f}% (수수료/슬리피지 100% 실시간 차감)")
    print(f"[*] 계좌 최대 낙폭(MDD) : {mdd:>6.2f}% 🛡️")
    print(f"[*] 실질 체결 총 거래수: {len(trades_df)}회 (월평균 약 {len(trades_df)/48.0:.1f}회 / 연평균 {len(trades_df)/4.0:.1f}회)")
    print(f"[*] 통산 실전 승률     : {(trades_df['net_pnl']>0).mean()*100:.1f}% (총 {len(trades_df)}전 {(trades_df['net_pnl']>0).sum()}승)")
    print("-" * 115)
    print("📊 [종목별 실질 체결 손익 기여도]")
    for sym, group in trades_df.groupby("symbol"):
        wr = (group["net_pnl"] > 0).mean() * 100.0
        print(f"    • {sym:<8}: {len(group):>3}회 체결 | 승률 {wr:>5.1f}% | 누적 기여 손익 ₩{group['net_pnl'].sum():>+10,.0f}")
    print("=" * 115)


if __name__ == "__main__":
    run_4coin_swing_unified()
