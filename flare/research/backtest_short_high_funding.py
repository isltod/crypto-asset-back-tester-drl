"""
flare.research.backtest_short_high_funding

양수 펀딩비 숏(Short) 실전 백테스트
- 조건 1: FR >= +0.050% (+0.0005)
- 조건 2: FR >= +0.100% (+0.0010)
- 룰: SL -4.0% / No TP / 24시간 만기 종가 청산
- 수수료/슬리피지: 진입 0.05%, 청산 0.05%, 슬리피지 0.02%
- 검증 기간: 2021~2024년 전체 4개년 (1시간봉 정밀 체결)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def run_short_simulation(
    df: pd.DataFrame,
    threshold: float,
    sl_pct: float = 4.0,
    max_horizon_hours: int = 24,
    fee_taker_pct: float = 0.05,
    slippage_pct: float = 0.02
):
    is_settle = df["datetime"].dt.hour.isin([0, 8, 16])
    sig = is_settle & (df["fundingRate"] >= threshold)
    
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
        # 숏 진입: 슬리피지로 인해 약간 더 낮은 가격에 체결
        entry_price = closes[entry_idx] * (1.0 - slippage_pct / 100.0)
        entry_fee = fee_taker_pct
        
        # 숏 손절 가격 (진입가 대비 +4.0% 상승 시 손절)
        sl_price = entry_price * (1.0 + sl_pct / 100.0)
        
        exit_idx = entry_idx
        exit_price = entry_price
        exit_reason = "TIMEOUT"
        
        for step in range(1, max_horizon_hours + 1):
            curr_idx = entry_idx + step
            if curr_idx >= n_bars:
                break
                
            h = highs[curr_idx]
            c = closes[curr_idx]
            
            # 숏 손절 체크: 고가가 sl_price 이상 도달 시
            if h >= sl_price:
                exit_idx = curr_idx
                exit_price = sl_price * (1.0 + slippage_pct / 100.0)
                exit_reason = "SL"
                break
                
            # 만기 (24시간)
            if step == max_horizon_hours:
                exit_idx = curr_idx
                exit_price = c * (1.0 + slippage_pct / 100.0)
                exit_reason = "TIMEOUT"
                break
                
        # 숏 수익률 = (entry_price - exit_price) / entry_price
        raw_ret_pct = (entry_price - exit_price) / entry_price * 100.0
        net_ret_pct = raw_ret_pct - (entry_fee + fee_taker_pct)
        
        trades.append({
            "entry_time": entry_time,
            "exit_time": pd.Timestamp(datetimes[exit_idx]),
            "return_pct": net_ret_pct,
            "exit_reason": exit_reason,
            "year": entry_time.year
        })
        
        i = exit_idx + 1 # 포지션 종료 후 다음 탐색
        
    trade_df = pd.DataFrame(trades)
    if len(trade_df) == 0:
        return {"total_trades": 0, "win_rate": 0, "cumulative_return": 0, "profit_factor": 0, "mdd": 0, "sl_count": 0, "timeout_count": 0}
        
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
    
    # 연도별 성과
    yearly_summary = {}
    for yr, group in trade_df.groupby("year"):
        yr_rets = group["return_pct"]
        yearly_summary[yr] = {
            "trades": len(group),
            "win_rate": (yr_rets > 0).mean() * 100.0,
            "return_pct": yr_rets.sum()
        }
        
    return {
        "total_trades": len(trade_df),
        "win_rate": win_rate,
        "cumulative_return": cum_ret,
        "profit_factor": pf,
        "mdd": mdd,
        "sl_count": sl_cnt,
        "timeout_count": to_cnt,
        "yearly": yearly_summary
    }


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    klines_4y = data_dir / "BTCUSDT_1h_4years_full.csv"
    
    df = pd.read_csv(klines_4y)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    df = pd.merge_asof(df, df_fr[["fundingTime", "fundingRate"]], left_on="datetime", right_on="fundingTime", direction="backward")
    df["fundingRate"] = df["fundingRate"].ffill().fillna(0.0001)
    
    targets = [
        ("1. [기관 차익거래선] FR >= +0.050% (+0.0005)", 0.0005),
        ("2. [극단 10배 폭탄] FR >= +0.100% (+0.0010)", 0.0010)
    ]
    
    print("=" * 115)
    print("🔬 [실전 백테스트] 롱 과열 시 24시간 숏(Short) 포지션 실전 성과 대조 (2021~2024, 4개년 풀 데이터)")
    print("=" * 115)
    
    for label, th in targets:
        res = run_short_simulation(df, th, sl_pct=4.0, max_horizon_hours=24)
        print(f"\n📊 {label}")
        print("-" * 115)
        print(f"    - 총 거래 횟수     : {res['total_trades']}회 (손절 SL: {res['sl_count']}회 / 만기 TO: {res['timeout_count']}회)")
        print(f"    - 4개년 통산 승률  : {res['win_rate']:.1f}%")
        print(f"    - 1배수 총 누적수익률: {res['cumulative_return']:>+8.2f}% (수수료/슬리피지 100% 실시간 차감)")
        print(f"    - 손익비 (PF)      : {res['profit_factor']:.2f}")
        print(f"    - 최대 낙폭 (MDD)  : {res['mdd']:.2f}%")
        print(f"    - 📅 연도별 성과:")
        for yr, y_stat in res["yearly"].items():
            print(f"        • {yr}년: {y_stat['trades']:>2}회 거래 | 승률 {y_stat['win_rate']:>5.1f}% | 누적수익률 {y_stat['return_pct']:>+6.2f}%")
    print("=" * 115)


if __name__ == "__main__":
    main()
