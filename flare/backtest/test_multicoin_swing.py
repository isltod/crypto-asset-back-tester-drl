"""
flare.backtest.test_multicoin_swing

BTC, ETH, SOL, DOGE, XRP 5대 메이저 코인에 대한
Mode 2.1 스윙 롱 (FR <= -0.010%, SL -4.0%, No TP, 24h) 4개년 실전 백테스트
- 각 코인별 독립 4개년 백테스트 (거래수, 승률, 수익률, 손익비, MDD)
- 5개 코인 동시 가동 통합 포트폴리오 성과
- 수수료/슬리피지 100% 차감
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def run_symbol_swing(
    symbol: str,
    data_dir: Path,
    threshold: float = -0.00010, # FR <= -0.010%
    sl_pct: float = 4.0,
    max_horizon_hours: int = 24,
    fee_taker_pct: float = 0.05,
    slippage_pct: float = 0.02
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
    
    trades = []
    i = 0
    
    while i < n_bars:
        if not sig_arr[i]:
            i += 1
            continue
            
        entry_idx = i
        entry_time = pd.Timestamp(datetimes[entry_idx])
        entry_price = closes[entry_idx] * (1.0 + slippage_pct / 100.0)
        entry_fee = fee_taker_pct
        sl_price = entry_price * (1.0 - sl_pct / 100.0)
        
        exit_idx = entry_idx
        exit_price = entry_price
        exit_reason = "TIMEOUT"
        
        for step in range(1, max_horizon_hours + 1):
            curr_idx = entry_idx + step
            if curr_idx >= n_bars:
                break
                
            l = lows[curr_idx]
            c = closes[curr_idx]
            
            if l <= sl_price:
                exit_idx = curr_idx
                exit_price = sl_price * (1.0 - slippage_pct / 100.0)
                exit_reason = "SL"
                break
                
            if step == max_horizon_hours:
                exit_idx = curr_idx
                exit_price = c * (1.0 - slippage_pct / 100.0)
                exit_reason = "TIMEOUT"
                break
                
        raw_ret = (exit_price - entry_price) / entry_price * 100.0
        net_ret = raw_ret - (entry_fee + fee_taker_pct)
        
        trades.append({
            "symbol": symbol,
            "entry_time": entry_time,
            "exit_time": pd.Timestamp(datetimes[exit_idx]),
            "return_pct": net_ret,
            "exit_reason": exit_reason,
            "year": entry_time.year
        })
        i = exit_idx + 1
        
    trade_df = pd.DataFrame(trades)
    if len(trade_df) == 0:
        return {"symbol": symbol, "trades": 0, "win_rate": 0, "cum_ret": 0, "pf": 0, "mdd": 0, "trade_df": trade_df}
        
    rets = trade_df["return_pct"]
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    
    cum_ret = rets.sum()
    win_rate = len(wins) / len(rets) * 100.0
    pf = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else 999.0
    
    eq = (1.0 + rets / 100.0).cumprod()
    mdd = abs(((eq - eq.cummax()) / eq.cummax() * 100.0).min())
    
    return {
        "symbol": symbol,
        "trades": len(trade_df),
        "win_rate": win_rate,
        "cum_ret": cum_ret,
        "pf": pf,
        "mdd": mdd,
        "trade_df": trade_df
    }


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
    
    print("=" * 115)
    print("🔬 [멀티 코인 실전 백테스트] 5대 메이저 코인 Mode 2.1 스윙 성과 대조 (2021~2024, 4개년 풀 데이터)")
    print("   • 조건: FR <= -0.010% (-0.01% 이하) | SL -4.0% | No TP | 24시간 만기 종가 청산")
    print("=" * 115)
    
    results = []
    all_trades = []
    
    header = "{:<12} | {:<8} | {:<8} | {:<14} | {:<8} | {:<10}"
    row = "{:<12} | {:>6}회 | {:>7.1f}% | {:>13.2f}% | {:>8.2f} | {:>9.2f}%"
    print(header.format("종목 (코인)", "4년 거래수", "승률", "1배수 누적수익률", "손익비(PF)", "최대낙폭(MDD)"))
    print("-" * 115)
    
    for sym in symbols:
        res = run_symbol_swing(sym, data_dir)
        if res is not None:
            results.append(res)
            if len(res["trade_df"]) > 0:
                all_trades.append(res["trade_df"])
            print(row.format(
                sym,
                res["trades"],
                res["win_rate"],
                res["cum_ret"],
                res["pf"],
                res["mdd"]
            ))
            
    # 통합 포트폴리오
    if all_trades:
        total_trade_df = pd.concat(all_trades).sort_values("entry_time").reset_index(drop=True)
        total_rets = total_trade_df["return_pct"]
        tot_wins = total_rets[total_rets > 0]
        tot_losses = total_rets[total_rets <= 0]
        
        tot_cum_ret = total_rets.sum()
        tot_wr = len(tot_wins) / len(total_rets) * 100.0
        tot_pf = tot_wins.sum() / abs(tot_losses.sum()) if abs(tot_losses.sum()) > 0 else 999.0
        
        tot_eq = (1.0 + total_rets / 100.0).cumprod()
        tot_mdd = abs(((tot_eq - tot_eq.cummax()) / tot_eq.cummax() * 100.0).min())
        
        print("-" * 115)
        print(row.format(
            "🔥 5종목 통합",
            len(total_trade_df),
            tot_wr,
            tot_cum_ret,
            tot_pf,
            tot_mdd
        ))
        print("=" * 115)
        print(f"[*] 🌟 5개 코인 통합 시 연평균 거래 횟수: 약 {len(total_trade_df)/4.0:.1f}회 (월평균 약 {len(total_trade_df)/48.0:.1f}회 / 매주 1.5~2회 기회 발생!) 🚀")


if __name__ == "__main__":
    main()
