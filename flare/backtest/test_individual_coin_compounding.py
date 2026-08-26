"""
flare.backtest.test_individual_coin_compounding

5대 메이저 코인(BTC, ETH, SOL, DOGE, XRP) 각각에 대해
개별 단독 실전 복리(Compound Growth) 백테스트 및 MDD 산출 모듈 (2021~2024, 4개년)
- 1배수 노레버리지 복리 성과
- 안전 2배수(2.0x, 80% 투입) 복리 성과
- 초기 자본금 100만 원 기준
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.backtest.test_multicoin_swing import run_symbol_swing


def run_coin_compounding_simulation(
    symbol: str,
    data_dir: Path,
    initial_capital: float = 1_000_000.0,
    leverage: float = 1.0,
    allocation_ratio: float = 1.00,
    threshold: float = -0.00010,
    sl_pct: float = 4.0
):
    funding_file = data_dir / f"{symbol.lower()}_funding_rate.csv"
    klines_file = data_dir / f"{symbol}_1h_4years_full.csv"
    
    if not funding_file.exists() or not klines_file.exists():
        return None
        
    df = pd.read_csv(klines_file)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    
    df = pd.merge_asof(df, df_fr[["fundingTime", "fundingRate"]], left_on="datetime", right_on="fundingTime", direction="backward")
    df["fundingRate"] = df["fundingRate"].ffill().fillna(0.0001)
    
    is_settle = df["datetime"].dt.hour.isin([0, 8, 16])
    sig = is_settle & (df["fundingRate"] <= threshold)
    
    n_bars = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    datetimes = df["datetime"].values
    sig_arr = sig.values
    
    cash = initial_capital
    i = 0
    trade_logs = []
    equity_curve = []
    
    while i < n_bars:
        current_eq = cash
        equity_curve.append(current_eq)
        
        if not sig_arr[i]:
            i += 1
            continue
            
        entry_idx = i
        entry_price = closes[entry_idx] * 1.0002
        trade_margin = cash * allocation_ratio
        entry_fee = (entry_price * (trade_margin * leverage / entry_price)) * 0.0005
        usable_margin = trade_margin - entry_fee
        pos_size = (usable_margin * leverage) / entry_price
        sl_price = entry_price * (1.0 - sl_pct / 100.0)
        
        exit_idx = entry_idx
        exit_price = entry_price
        exit_reason = "TIMEOUT"
        
        for step in range(1, 25): # 24시간
            curr_idx = entry_idx + step
            if curr_idx >= n_bars:
                break
                
            l = lows[curr_idx]
            c = closes[curr_idx]
            
            if l <= sl_price:
                exit_idx = curr_idx
                exit_price = sl_price * 0.9998
                exit_reason = "SL"
                break
                
            if step == 24:
                exit_idx = curr_idx
                exit_price = c * 0.9998
                exit_reason = "TIMEOUT"
                break
                
        raw_pnl = (exit_price - entry_price) * pos_size
        exit_fee = (exit_price * pos_size) * 0.0005
        net_trade_pnl = raw_pnl - exit_fee
        
        cash += net_trade_pnl
        trade_logs.append(net_trade_pnl)
        
        i = exit_idx + 1
        
    final_balance = cash
    cum_ret_pct = (final_balance - initial_capital) / initial_capital * 100.0
    
    eq_series = pd.Series(equity_curve)
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak * 100.0
    mdd = abs(dd.min())
    
    win_cnt = sum(1 for p in trade_logs if p > 0)
    win_rate = (win_cnt / len(trade_logs) * 100.0) if len(trade_logs) > 0 else 0
    
    return {
        "symbol": symbol,
        "trades": len(trade_logs),
        "win_rate": win_rate,
        "final_balance": final_balance,
        "return_pct": cum_ret_pct,
        "mdd": mdd
    }


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    
    # 5대 코인 설정
    coin_configs = [
        ("BTCUSDT (비트코인)", "BTCUSDT", -0.00010, 4.0),
        ("ETHUSDT (이더리움)", "ETHUSDT", -0.00010, 4.0),
        ("SOLUSDT (솔라나)",   "SOLUSDT", -0.00025, 6.0),
        ("XRPUSDT (리플)",     "XRPUSDT", -0.00010, 4.0),
        ("DOGEUSDT(도지코인)", "DOGEUSDT", -0.00010, 4.0)
    ]
    
    print("=" * 125)
    print("🔬 [5대 메이저 코인] 개별 단독 실전 복리(Compound) 4개년 백테스트 성적표 (2021~2024, 초기자본 100만 원)")
    print("=" * 125)
    
    # 1. 1배수 노레버리지 복리
    print("\n📊 1. [1배수 노레버리지 복리] (1.0x / 100% 투입)")
    print("-" * 125)
    h1 = "{:<20} | {:<8} | {:<8} | {:<16} | {:<16} | {:<12}"
    r1 = "{:<20} | {:>6}회 | {:>6.1f}% | {:>14} | {:>14.2f}% | {:>10.2f}%"
    print(h1.format("종목 (코인)", "4년 거래수", "승률", "최종 잔고(원)", "실제 복리수익률", "최대낙폭(MDD)"))
    print("-" * 125)
    
    for label, sym, th, sl in coin_configs:
        res = run_coin_compounding_simulation(sym, data_dir, initial_capital=1_000_000.0, leverage=1.0, allocation_ratio=1.00, threshold=th, sl_pct=sl)
        print(r1.format(label, res["trades"], res["win_rate"], f"₩{res['final_balance']:,.0f}", res["return_pct"], res["mdd"]))
        
    # 2. 안전 2배수 복리 (2.0x / 80% 투입)
    print("\n📊 2. [안전 2배수 복리] (2.0x / 80% 투입, 20% 현금 버퍼)")
    print("-" * 125)
    print(h1.format("종목 (코인)", "4년 거래수", "승률", "최종 잔고(원)", "실제 복리수익률", "최대낙폭(MDD)"))
    print("-" * 125)
    
    for label, sym, th, sl in coin_configs:
        res = run_coin_compounding_simulation(sym, data_dir, initial_capital=1_000_000.0, leverage=2.0, allocation_ratio=0.80, threshold=th, sl_pct=sl)
        print(r1.format(label, res["trades"], res["win_rate"], f"₩{res['final_balance']:,.0f}", res["return_pct"], res["mdd"]))
    print("=" * 125)


if __name__ == "__main__":
    main()
