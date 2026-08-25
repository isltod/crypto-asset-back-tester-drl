"""
flare.backtest.test_advanced_swing

Mode 2.1 베이스라인(+36.18%)을 뛰어넘기 위한 고급 청산 알고리즘 1:1 대조 백테스트:
1) [Base]: No TP (SL -4.0% 고정 + 24h 만기)
2) [Algo A]: 이익 잠금 트레일링 스탑 (Profit Lock-in Trailing)
   - High가 +2.5% 터치 시 ➔ SL을 본전(+0.1%)으로 상향
   - High가 +4.0% 터치 시 ➔ SL을 +2.0% 이익 확정선으로 상향
3) [Algo B]: 분할 익절 (50%는 +3.0% 지정가 익절, 50%는 24h 만기)
4) [Algo C]: 적극적 트레일링 스탑 (High가 +3.0% 터치 시 ➔ 최고점 대비 -1.5% 추적 스탑)
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


def run_advanced_swing_simulation(
    df: pd.DataFrame,
    signals: pd.Series,
    algo_type: str,
    fee_maker_pct: float = 0.02,
    fee_taker_pct: float = 0.05,
    slippage_pct: float = 0.02
):
    n_bars = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    datetimes = df["datetime"].values
    sig_arr = signals.values
    
    trades = []
    i = 0
    max_bars = 288 # 24시간
    
    while i < n_bars:
        if not sig_arr[i]:
            i += 1
            continue
            
        entry_idx = i
        entry_time = pd.Timestamp(datetimes[entry_idx])
        entry_price = closes[entry_idx] * (1.0 + slippage_pct / 100.0)
        entry_fee = fee_taker_pct
        
        curr_sl = entry_price * 0.96 # 초기 SL -4.0%
        highest_p = entry_price
        
        exit_idx = entry_idx
        exit_price = entry_price
        exit_reason = "TIMEOUT"
        exit_fee = fee_taker_pct
        
        # 분할 익절용 변수
        partial_tp_done = False
        partial_tp_pnl = 0.0
        
        for step in range(1, max_bars + 1):
            curr_idx = entry_idx + step
            if curr_idx >= n_bars:
                break
                
            h = highs[curr_idx]
            l = lows[curr_idx]
            c = closes[curr_idx]
            
            highest_p = max(highest_p, h)
            gain_from_entry = (highest_p - entry_price) / entry_price * 100.0
            
            # --- 알고리즘별 동적 규칙 ---
            if algo_type == "ALGO_A": # Profit Lock-in
                if gain_from_entry >= 4.0:
                    curr_sl = max(curr_sl, entry_price * 1.020) # +2.0% 이익 잠금
                elif gain_from_entry >= 2.5:
                    curr_sl = max(curr_sl, entry_price * 1.001) # 본전 잠금
                    
            elif algo_type == "ALGO_C": # Chandelier Trailing
                if gain_from_entry >= 3.0:
                    curr_sl = max(curr_sl, highest_p * 0.985) # 최고점 대비 -1.5% 추적
                    
            elif algo_type == "ALGO_B": # Partial TP (50% at +3.0%)
                if not partial_tp_done and h >= entry_price * 1.030:
                    partial_tp_done = True
                    partial_exit_p = entry_price * 1.030
                    partial_tp_pnl = ((partial_exit_p - entry_price) / entry_price * 100.0) - (entry_fee + fee_maker_pct)
                    curr_sl = max(curr_sl, entry_price * 1.001) # 나머지 50%는 본전 잠금
            
            # 1. 손절/트레일링 체크
            if l <= curr_sl:
                exit_idx = curr_idx
                exit_price = curr_sl * (1.0 - slippage_pct / 100.0)
                exit_reason = "SL/TRAIL"
                exit_fee = fee_taker_pct
                break
                
            # 2. 만기 (24시간)
            if step == max_bars:
                exit_idx = curr_idx
                exit_price = c * (1.0 - slippage_pct / 100.0)
                exit_reason = "TIMEOUT"
                exit_fee = fee_taker_pct
                break
                
        # 최종 수익률 계산
        if algo_type == "ALGO_B":
            rem_pnl = ((exit_price - entry_price) / entry_price * 100.0) - (entry_fee + exit_fee)
            if partial_tp_done:
                net_ret_pct = 0.5 * partial_tp_pnl + 0.5 * rem_pnl
            else:
                net_ret_pct = rem_pnl
        else:
            raw_ret_pct = (exit_price - entry_price) / entry_price * 100.0
            net_ret_pct = raw_ret_pct - (entry_fee + exit_fee)
            
        trades.append({
            "entry_time": entry_time,
            "exit_time": pd.Timestamp(datetimes[exit_idx]),
            "return_pct": net_ret_pct,
            "exit_reason": exit_reason,
            "hold_bars": exit_idx - entry_idx
        })
        
        i = exit_idx + 1
        
    trade_df = pd.DataFrame(trades)
    rets = trade_df["return_pct"]
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    
    cum_ret = rets.sum()
    win_rate = len(wins) / len(rets) * 100.0 if len(rets) > 0 else 0
    pf = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else 1.0
    
    eq = (1.0 + rets / 100.0).cumprod()
    mdd = abs(((eq - eq.cummax()) / eq.cummax() * 100.0).min())
    sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(len(rets))
    
    return {
        "trades": len(trade_df),
        "win_rate": win_rate,
        "cumulative_return": cum_ret,
        "profit_factor": pf,
        "mdd": mdd,
        "sharpe": sharpe,
        "avg_ret": rets.mean()
    }


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    klines_file = data_dir / "BTCUSDT_5m_2022_2024.csv"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
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
    
    print("=" * 115)
    print("🚀 [FLARE-Swing] 베이스라인 vs 고급 청산 알고리즘 1:1 대조 백테스트 (2022~2024)")
    print("=" * 115)
    
    # 1. Base
    m_base = run_advanced_swing_simulation(eval_df, sig_swing_pure, "BASE")
    # 2. Algo A: Profit Lock-in (+2.5% ➔ 본전, +4.0% ➔ +2.0% 잠금)
    m_a = run_advanced_swing_simulation(eval_df, sig_swing_pure, "ALGO_A")
    # 3. Algo B: 50% Partial TP at +3.0%
    m_b = run_advanced_swing_simulation(eval_df, sig_swing_pure, "ALGO_B")
    # 4. Algo C: Chandelier Trailing (+3.0% 터치 후 최고점 대비 -1.5% 추적)
    m_c = run_advanced_swing_simulation(eval_df, sig_swing_pure, "ALGO_C")
    
    results = [
        {"알고리즘 전략": "1. [Base] No TP (SL -4% + 24h 만기)", **m_base},
        {"알고리즘 전략": "2. [Algo A] 이익 잠금 트레일링 (Lock-in)", **m_a},
        {"알고리즘 전략": "3. [Algo B] 50% 분할익절 (+3% TP + 만기)", **m_b},
        {"알고리즘 전략": "4. [Algo C] 고점 추적 트레일링 (Chandelier)", **m_c}
    ]
    res_df = pd.DataFrame(results)
    
    header_fmt = "{:<40} | {:<6} | {:<8} | {:<12} | {:<7} | {:<8} | {:<8} | {:<8}"
    row_fmt = "{:<40} | {:>6} | {:>7.1f}% | {:>11.2f}% | {:>7.2f} | {:>7.2f}% | {:>8.2f} | {:>+7.2f}%"
    
    print(header_fmt.format("전략 알고리즘", "거래수", "승률", "총 누적수익률", "손익비", "최대낙폭(MDD)", "샤프지수", "건당평균"))
    print("-" * 115)
    for _, r in res_df.iterrows():
        print(row_fmt.format(
            r["알고리즘 전략"],
            f"{r['trades']}회",
            r["win_rate"],
            r["cumulative_return"],
            r["profit_factor"],
            r["mdd"],
            r["sharpe"],
            r["avg_ret"]
        ))
    print("=" * 115)


if __name__ == "__main__":
    main()
