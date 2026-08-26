"""
flare.research.test_sniper_both_sides

Mode 1.1 (FLARE-Sniper, 4h) 롱 & 숏 양방향 정밀 백테스트
1) 롱 저격 (Long Sniper): 음수 펀딩비 (FR <= -0.005%, -0.010%) + 5분봉 거래량 3x & 아래꼬리 55%+
2) 숏 저격 (Short Sniper): 양수 펀딩비 (FR >= +0.030%, +0.050%, +0.100%) + 5분봉 거래량 3x & 위꼬리 55%+
- 청산 룰: SL -3.0% / No TP / 4시간 만기 종가 청산
- 실전 수수료 (Maker 0.02%, Taker 0.05%) 및 슬리피지 (0.02%) 적용
- 검증 구간: 2022~2024년 (2.5년 인샘플) & 2021년 (1년 OOS 불장)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.data.features import generate_all_features


def run_sniper_side_simulation(
    df: pd.DataFrame,
    signals: pd.Series,
    side: str = "LONG", # "LONG" or "SHORT"
    sl_pct: float = 3.0,
    max_horizon_bars: int = 48, # 4시간 (48봉)
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
    
    while i < n_bars:
        if not sig_arr[i]:
            i += 1
            continue
            
        entry_idx = i
        entry_time = pd.Timestamp(datetimes[entry_idx])
        
        if side == "LONG":
            entry_price = closes[entry_idx] * (1.0 + slippage_pct / 100.0)
            sl_price = entry_price * (1.0 - sl_pct / 100.0)
        else: # SHORT
            entry_price = closes[entry_idx] * (1.0 - slippage_pct / 100.0)
            sl_price = entry_price * (1.0 + sl_pct / 100.0)
            
        entry_fee = fee_taker_pct
        exit_idx = entry_idx
        exit_price = entry_price
        exit_reason = "TIMEOUT"
        
        for step in range(1, max_horizon_bars + 1):
            curr_idx = entry_idx + step
            if curr_idx >= n_bars:
                break
                
            h = highs[curr_idx]
            l = lows[curr_idx]
            c = closes[curr_idx]
            
            if side == "LONG":
                # 롱 손절: 저가가 sl_price 이하
                if l <= sl_price:
                    exit_idx = curr_idx
                    exit_price = sl_price * (1.0 - slippage_pct / 100.0)
                    exit_reason = "SL"
                    break
            else: # SHORT
                # 숏 손절: 고가가 sl_price 이상
                if h >= sl_price:
                    exit_idx = curr_idx
                    exit_price = sl_price * (1.0 + slippage_pct / 100.0)
                    exit_reason = "SL"
                    break
                    
            # 4시간 만기
            if step == max_horizon_bars:
                exit_idx = curr_idx
                if side == "LONG":
                    exit_price = c * (1.0 - slippage_pct / 100.0)
                else:
                    exit_price = c * (1.0 + slippage_pct / 100.0)
                exit_reason = "TIMEOUT"
                break
                
        # 수익률 계산
        if side == "LONG":
            raw_ret_pct = (exit_price - entry_price) / entry_price * 100.0
        else:
            raw_ret_pct = (entry_price - exit_price) / entry_price * 100.0
            
        net_ret_pct = raw_ret_pct - (entry_fee + fee_taker_pct)
        
        trades.append({
            "entry_time": entry_time,
            "exit_time": pd.Timestamp(datetimes[exit_idx]),
            "return_pct": net_ret_pct,
            "exit_reason": exit_reason
        })
        
        i = exit_idx + 1
        
    trade_df = pd.DataFrame(trades)
    if len(trade_df) == 0:
        return {"trades": 0, "win_rate": 0, "cum_ret": 0, "pf": 0, "mdd": 0, "sl_cnt": 0, "to_cnt": 0}
        
    rets = trade_df["return_pct"]
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    
    cum_ret = rets.sum()
    win_rate = len(wins) / len(rets) * 100.0
    pf = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else 999.0
    
    eq = (1.0 + rets / 100.0).cumprod()
    mdd = abs(((eq - eq.cummax()) / eq.cummax() * 100.0).min())
    sl_cnt = (trade_df["exit_reason"] == "SL").sum()
    to_cnt = (trade_df["exit_reason"] == "TIMEOUT").sum()
    
    return {
        "trades": len(trade_df),
        "win_rate": win_rate,
        "cum_ret": cum_ret,
        "pf": pf,
        "mdd": mdd,
        "sl_cnt": sl_cnt,
        "to_cnt": to_cnt
    }


def evaluate_dataset(klines_path: Path, label: str):
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
    df = pd.read_csv(klines_path)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    df = pd.merge_asof(df, df_fr[["fundingTime", "fundingRate"]], left_on="datetime", right_on="fundingTime", direction="backward")
    df["fundingRate"] = df["fundingRate"].ffill().fillna(0.0001)
    
    df, _ = generate_all_features(df)
    
    # 5분봉 위꼬리 스파이크 피처 생성
    total_range = (df["high"] - df["low"]).replace(0, 1e-9)
    candle_body_max = df[["open", "close"]].max(axis=1)
    upper_wick = df["high"] - candle_body_max
    df["feat_upper_wick_ratio"] = upper_wick / total_range
    is_upper_wick_spike = (df["feat_vol_ratio_24h"] >= 3.0) & (df["feat_upper_wick_ratio"] >= 0.55)
    
    eval_df = df.iloc[8640:].reset_index(drop=True)
    is_lower_wick = (eval_df["feat_is_lower_wick_spike"] == 1.0)
    is_upper_wick = is_upper_wick_spike.iloc[8640:].reset_index(drop=True)
    
    # 전략 시나리오 정의
    scenarios = [
        # [롱 저격]
        ("🟢 [롱] FR <= -0.005% & 아래꼬리 55%+", eval_df["fundingRate"] <= -0.00005, is_lower_wick, "LONG"),
        ("🟢 [롱] FR <= -0.010% & 아래꼬리 55%+", eval_df["fundingRate"] <= -0.00010, is_lower_wick, "LONG"),
        
        # [숏 저격]
        ("🔴 [숏] FR >= +0.030% & 위꼬리 55%+", eval_df["fundingRate"] >= 0.00030, is_upper_wick, "SHORT"),
        ("🔴 [숏] FR >= +0.050% & 위꼬리 55%+", eval_df["fundingRate"] >= 0.00050, is_upper_wick, "SHORT"),
        ("🔴 [숏] FR >= +0.100% & 위꼬리 55%+", eval_df["fundingRate"] >= 0.00100, is_upper_wick, "SHORT"),
    ]
    
    print(f"\n===================================================================================================")
    print(f"🔬 [{label}] Mode 1.1 스나이퍼 롱 & 숏 양방향 실전 백테스트 (SL -3%, No TP, 4h)")
    print(f"===================================================================================================")
    header = "{:<42} | {:<6} | {:<7} | {:<11} | {:<7} | {:<8} | {:<15}"
    row = "{:<42} | {:>6} | {:>6.1f}% | {:>10.2f}% | {:>7.2f} | {:>7.2f}% | SL:{:<2} TO:{:<2}"
    print(header.format("전략 시나리오 (방향 & 펀딩비 & 꼬리)", "거래수", "승률", "누적수익률", "손익비", "최대낙폭(MDD)", "청산 분포"))
    print("-" * 105)
    
    for label_str, fr_cond, wick_cond, side in scenarios:
        sig = fr_cond & wick_cond
        res = run_sniper_side_simulation(eval_df, sig, side=side, sl_pct=3.0, max_horizon_bars=48)
        print(row.format(
            label_str,
            f"{res['trades']}회",
            res["win_rate"],
            res["cum_ret"],
            res["pf"],
            res["mdd"],
            res["sl_cnt"],
            res["to_cnt"]
        ))


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    evaluate_dataset(data_dir / "BTCUSDT_5m_2022_2024.csv", "2022~2024년 약세/횡보 인샘플 (2.5년)")
    evaluate_dataset(data_dir / "BTCUSDT_5m_2021.csv", "2021년 불장 OOS (1년)")


if __name__ == "__main__":
    main()
