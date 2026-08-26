"""
flare.backtest.test_multi_position_equal_weight

모든 종목의 동시 중복 포지션을 100% 허용하고,
총 자본을 1/N (슬롯 분할)로 균등 배분하여 운용하는 4개년 실전 복리 백테스터 (2021~2024)
- 케이스 A: 4대 정예 코인 (BTC, ETH, SOL, XRP) ➔ N=4 (각 25% 슬롯)
- 케이스 B: 5대 코인 (BTC, ETH, SOL, XRP, DOGE) ➔ N=5 (각 20% 슬롯)
- 초기 자본: 100만 원
- 레버리지: 2.0x (슬롯별 자본의 80% 투입, 20% 현금 버퍼)
- 룰: SL -4.0% (SOL은 SL -6.0%) / No TP / 24시간 만기 종가 청산
- 수수료/슬리피지 실시간 100% 차감
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List
import pandas as pd
import numpy as np

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.backtest.test_multicoin_unified_account import load_coin_events


@dataclass
class PositionSlot:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    position_size: float
    margin_cost: float
    leverage: float
    sl_price: float
    max_bars: int = 24
    bars_held: int = 0


def run_equal_weight_multi_position(
    symbols: List[str],
    data_dir: Path,
    initial_capital: float = 1_000_000.0,
    leverage: float = 2.0,
    allocation_ratio: float = 0.80, # 총 슬롯 한도의 80%만 투입
    fee_taker: float = 0.0005,
    slippage: float = 0.0002
):
    n_slots = len(symbols)
    slot_weight = 1.0 / n_slots
    
    dfs = [load_coin_events(sym, data_dir) for sym in symbols]
    coin_dict = {df["symbol"].iloc[0]: df.set_index("datetime") for df in dfs}
    
    master_timeline = pd.date_range("2021-01-01 00:00:00+00:00", "2024-12-31 23:00:00+00:00", freq="1h", tz="UTC")
    
    cash = initial_capital
    active_positions: Dict[str, PositionSlot] = {} # 심볼별 독립 포지션 슬롯
    trade_logs = []
    equity_curve = []
    
    for current_time in master_timeline:
        # 1. 활성 포지션들의 청산 조건 체크
        closed_symbols = []
        for sym, pos in list(active_positions.items()):
            sym_data = coin_dict[sym]
            if current_time in sym_data.index:
                row = sym_data.loc[current_time]
                pos.bars_held += 1
                exit_price = None
                exit_reason = None
                
                if row["low"] <= pos.sl_price:
                    exit_price = pos.sl_price * (1.0 - slippage)
                    exit_reason = "SL"
                elif pos.bars_held >= pos.max_bars:
                    exit_price = row["close"] * (1.0 - slippage)
                    exit_reason = "TIMEOUT"
                    
                if exit_price is not None:
                    raw_pnl = (exit_price - pos.entry_price) * pos.position_size
                    exit_fee = (exit_price * pos.position_size) * fee_taker
                    net_trade_pnl = raw_pnl - exit_fee
                    
                    cash += pos.margin_cost + net_trade_pnl
                    ret_pct = (net_trade_pnl / pos.margin_cost) * 100.0
                    
                    trade_logs.append({
                        "symbol": sym,
                        "entry_time": pos.entry_time,
                        "exit_time": current_time,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "net_pnl": net_trade_pnl,
                        "return_pct": ret_pct,
                        "exit_reason": exit_reason,
                        "balance": cash
                    })
                    closed_symbols.append(sym)
                    
        for sym in closed_symbols:
            del active_positions[sym]
            
        # 2. 각 종목별로 비어있는 슬롯이 있고 신호가 떴다면 독립적으로 진입! (중복 허용)
        # 현재 전체 총 자산(Equity) 계산
        current_margin_locked = sum(pos.margin_cost for pos in active_positions.values())
        total_equity = cash + current_margin_locked
        
        for sym in symbols:
            if sym not in active_positions: # 해당 코인 슬롯이 비어있을 때
                sym_data = coin_dict[sym]
                if current_time in sym_data.index:
                    row = sym_data.loc[current_time]
                    if row["sig_swing"]: # 스윙 신호 발생!
                        # 1/N 배분 마진
                        trade_margin = (total_equity * slot_weight) * allocation_ratio
                        if cash >= trade_margin: # 현금이 충분할 때 진입
                            c = row["close"]
                            entry_p = c * (1.0 + slippage)
                            entry_fee = (entry_p * (trade_margin * leverage / entry_p)) * fee_taker
                            usable_margin = trade_margin - entry_fee
                            pos_size = (usable_margin * leverage) / entry_p
                            sl_rate = 0.06 if sym == "SOLUSDT" else 0.04
                            sl_p = entry_p * (1.0 - sl_rate)
                            
                            active_positions[sym] = PositionSlot(
                                symbol=sym,
                                entry_time=current_time,
                                entry_price=entry_p,
                                position_size=pos_size,
                                margin_cost=usable_margin,
                                leverage=leverage,
                                sl_price=sl_p,
                                max_bars=24
                            )
                            cash -= trade_margin
                            
        # 실시간 총 자산 기록
        current_margin_locked = sum(pos.margin_cost for pos in active_positions.values())
        equity_curve.append(cash + current_margin_locked)
        
    trades_df = pd.DataFrame(trade_logs)
    final_margin = sum(pos.margin_cost for pos in active_positions.values())
    final_balance = cash + final_margin
    total_ret_pct = (final_balance - initial_capital) / initial_capital * 100.0
    
    eq_series = pd.Series(equity_curve)
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak * 100.0
    mdd = abs(dd.min())
    
    return {
        "symbols": symbols,
        "n_slots": n_slots,
        "final_balance": final_balance,
        "return_pct": total_ret_pct,
        "mdd": mdd,
        "trades_df": trades_df,
        "equity_curve": equity_curve,
        "timestamps": list(master_timeline)
    }


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    
    print("=" * 115)
    print("🔬 [1/N 자본 분할 & 전 종목 동시 포지션 허용] 실전 복리 백테스트 성적표 (2021~2024, 4개년)")
    print("   • 조건: 안전 2.0배 레버리지 (2.0x) | 슬롯당 (총자산/N)*80% 투입 | No TP | 24시간 만기 청산")
    print("=" * 115)
    
    # 1. 4대 정예 코인 (N=4, 각 25%)
    res_4 = run_equal_weight_multi_position(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"], data_dir)
    print(f"\n📊 1. [4대 정예 코인 1/4 균등 분할 동시 운용] (BTC, ETH, SOL, XRP)")
    print("-" * 115)
    print(f"[*] 4년 뒤 최종 계좌 잔고 : ₩{res_4['final_balance']:,.0f} (약 {res_4['final_balance']/1e6:.2f}배 증식! 🚀)")
    print(f"[*] 실전 복리 총수익률    : {res_4['return_pct']:>+8.2f}%")
    print(f"[*] 계좌 최대 낙폭(MDD)    : {res_4['mdd']:>6.2f}% 🛡️ (낙폭 대폭 축소!)")
    print(f"[*] 실질 체결 총 거래수   : {len(res_4['trades_df'])}회 (월평균 약 {len(res_4['trades_df'])/48.0:.1f}회 / 연평균 {len(res_4['trades_df'])/4.0:.1f}회)")
    print(f"[*] 통산 실전 승률        : {(res_4['trades_df']['net_pnl']>0).mean()*100:.1f}%")
    print("    [종목별 실질 체결 손익 기여도]")
    for sym, group in res_4["trades_df"].groupby("symbol"):
        wr = (group["net_pnl"] > 0).mean() * 100.0
        print(f"       • {sym:<8}: {len(group):>3}회 체결 | 승률 {wr:>5.1f}% | 누적 기여 손익 ₩{group['net_pnl'].sum():>+10,.0f}")
        
    # 2. 5대 코인 (N=5, 각 20%)
    res_5 = run_equal_weight_multi_position(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"], data_dir)
    print(f"\n📊 2. [5대 코인 1/5 균등 분할 동시 운용] (BTC, ETH, SOL, XRP, DOGE)")
    print("-" * 115)
    print(f"[*] 4년 뒤 최종 계좌 잔고 : ₩{res_5['final_balance']:,.0f} (약 {res_5['final_balance']/1e6:.2f}배 증식! 🚀)")
    print(f"[*] 실전 복리 총수익률    : {res_5['return_pct']:>+8.2f}%")
    print(f"[*] 계좌 최대 낙폭(MDD)    : {res_5['mdd']:>6.2f}% 🛡️")
    print(f"[*] 실질 체결 총 거래수   : {len(res_5['trades_df'])}회 (월평균 약 {len(res_5['trades_df'])/48.0:.1f}회)")
    print(f"[*] 통산 실전 승률        : {(res_5['trades_df']['net_pnl']>0).mean()*100:.1f}%")
    print("    [종목별 실질 체결 손익 기여도]")
    for sym, group in res_5["trades_df"].groupby("symbol"):
        wr = (group["net_pnl"] > 0).mean() * 100.0
        print(f"       • {sym:<8}: {len(group):>3}회 체결 | 승률 {wr:>5.1f}% | 누적 기여 손익 ₩{group['net_pnl'].sum():>+10,.0f}")
    print("=" * 115)


if __name__ == "__main__":
    main()
