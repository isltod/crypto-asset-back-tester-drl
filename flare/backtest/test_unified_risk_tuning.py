"""
flare.backtest.test_unified_risk_tuning

FLARE 통합 단일 계좌에 대해
레버리지(Leverage)와 자금 투입 비중(Allocation Ratio)별
실제 1원 단위 복리 시계열 백테스트 및 MDD 정밀 측정 모듈
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


def run_single_simulation(
    eval_df: pd.DataFrame,
    swing_sigs: np.ndarray,
    sniper_sigs: np.ndarray,
    initial_capital: float = 1_000_000.0,
    swing_leverage: float = 2.5,
    sniper_leverage: float = 3.0,
    allocation_ratio: float = 0.80, # 계좌 잔고의 몇 %를 증거금으로 투입할 것인가 (예: 0.80 = 80% 투입, 20% 현금 보관)
    fee_taker: float = 0.0005,
    slippage: float = 0.0002
):
    highs = eval_df["high"].values
    lows = eval_df["low"].values
    closes = eval_df["close"].values
    datetimes = eval_df["datetime"].values
    n_bars = len(eval_df)
    
    cash = initial_capital
    position: Optional[Position] = None
    trade_logs = []
    equity_curve = []
    
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
                trade_logs.append(net_trade_pnl)
                position = None
                
        # 2. 신규 진입 (1계좌 1포지션)
        if position is None:
            total_equity = cash
            trade_margin = total_equity * allocation_ratio # 자금 비중 적용
            
            if swing_sigs[i]:
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
                    max_bars=288
                )
                cash -= trade_margin
                
            elif sniper_sigs[i]:
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
                    max_bars=48
                )
                cash -= trade_margin
                
        current_equity = cash if position is None else cash + position.margin_cost + ((c - position.entry_price) * position.position_size)
        equity_curve.append(current_equity)
        
    final_balance = cash if position is None else cash + position.margin_cost
    cum_ret_pct = (final_balance - initial_capital) / initial_capital * 100.0
    
    eq_series = pd.Series(equity_curve)
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak * 100.0
    mdd = abs(dd.min())
    
    win_cnt = sum(1 for pnl in trade_logs if pnl > 0)
    win_rate = (win_cnt / len(trade_logs) * 100.0) if len(trade_logs) > 0 else 0
    
    return {
        "trades": len(trade_logs),
        "final_balance": final_balance,
        "return_pct": cum_ret_pct,
        "mdd": mdd,
        "win_rate": win_rate
    }


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
    
    is_settle_bar = eval_df["datetime"].dt.minute == 0
    is_settle_hour = eval_df["datetime"].dt.hour.isin([0, 8, 16])
    sig_swing = (is_settle_bar & is_settle_hour & (eval_df["feat_funding_rsi_30d"] <= 0.05)).values
    sig_sniper = ((eval_df["feat_funding_rsi_30d"] <= 0.10) & (eval_df["feat_is_lower_wick_spike"] == 1.0)).values
    
    scenarios = [
        {"이름": "1. [공격형 올인] 스윙 3.0x / 스나 4.0x / 100% 투입", "sw_lev": 3.0, "sn_lev": 4.0, "alloc": 1.00},
        {"이름": "2. [균형형 A]   스윙 2.5x / 스나 3.0x / 100% 투입", "sw_lev": 2.5, "sn_lev": 3.0, "alloc": 1.00},
        {"이름": "3. [균형형 B]   스윙 2.5x / 스나 3.0x /  80% 투입", "sw_lev": 2.5, "sn_lev": 3.0, "alloc": 0.80},
        {"이름": "4. [안전형 A]   스윙 2.0x / 스나 2.5x / 100% 투입", "sw_lev": 2.0, "sn_lev": 2.5, "alloc": 1.00},
        {"이름": "5. [안전형 B]   스윙 2.0x / 스나 2.5x /  80% 투입", "sw_lev": 2.0, "sn_lev": 2.5, "alloc": 0.80},
        {"이름": "6. [초안전형]   스윙 1.5x / 스나 2.0x /  70% 투입", "sw_lev": 1.5, "sn_lev": 2.0, "alloc": 0.70},
        {"이름": "7. [노레버리지] 스윙 1.0x / 스나 1.0x / 100% 투입", "sw_lev": 1.0, "sn_lev": 1.0, "alloc": 1.00}
    ]
    
    results = []
    print("=" * 115)
    print("🔬 [FLARE 통합 계좌] 레버리지 & 투입비중별 실제 시계열 복리 성과 정밀 비교 (초기자본 100만원)")
    print("=" * 115)
    
    for sc in scenarios:
        res = run_single_simulation(
            eval_df, sig_swing, sig_sniper,
            initial_capital=1_000_000.0,
            swing_leverage=sc["sw_lev"],
            sniper_leverage=sc["sn_lev"],
            allocation_ratio=sc["alloc"]
        )
        results.append({
            "시나리오": sc["이름"],
            "투입비중": f"{int(sc['alloc']*100)}%",
            "최종잔고": f"₩{res['final_balance']:,.0f}",
            "복리수익률": res["return_pct"],
            "MDD": res["mdd"],
            "승률": res["win_rate"],
            "거래수": res["trades"]
        })
        
    res_df = pd.DataFrame(results)
    
    header_fmt = "{:<48} | {:<12} | {:<12} | {:<10} | {:<6}"
    row_fmt = "{:<48} | {:>12} | {:>11.2f}% | {:>9.2f}% | {:>5.1f}%"
    
    print(header_fmt.format("운용 시나리오 (레버리지 & 투입비중)", "최종 잔고(원)", "실전 복리수익률", "최대낙폭(MDD)", "승률"))
    print("-" * 115)
    for _, r in res_df.iterrows():
        print(row_fmt.format(
            r["시나리오"],
            r["최종잔고"],
            r["복리수익률"],
            r["MDD"],
            r["승률"]
        ))
    print("=" * 115)


if __name__ == "__main__":
    main()
