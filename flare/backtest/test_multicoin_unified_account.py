"""
flare.backtest.test_multicoin_unified_account

소액 자본(100만 원) 1계좌 1포지션 완전 충돌 배제 실전 복리 시계열 백테스터
- 대상: BTC, ETH, SOL, DOGE, XRP 5대 메이저 코인
- 모드: Mode 2.1 스윙 (24h) + Mode 1.1 스나이퍼 (4h)
- 1계좌 1포지션 원칙: 어떤 코인이든 이미 포지션을 보유 중이면 다른 코인/모드 신호는 100% 진입 차단 (No Multi-Exposure)
- 기간: 2021-01-01 ~ 2024-12-31 (4개년 풀 데이터)
- 실전 수수료/슬리피지 100% 실시간 차감
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import numpy as np

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@dataclass
class ActivePosition:
    symbol: str
    mode: str
    entry_time: pd.Timestamp
    entry_price: float
    position_size: float
    margin_cost: float
    leverage: float
    sl_price: float
    max_bars: int
    bars_held: int = 0


def load_coin_events(symbol: str, data_dir: Path) -> pd.DataFrame:
    funding_file = data_dir / f"{symbol.lower()}_funding_rate.csv"
    klines_file = data_dir / f"{symbol}_1h_4years_full.csv"
    
    if not funding_file.exists() or not klines_file.exists():
        return pd.DataFrame()
        
    df = pd.read_csv(klines_file)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    df_fr = pd.read_csv(funding_file)
    df_fr["fundingTime"] = pd.to_datetime(df_fr["fundingTime"], format="ISO8601", utc=True)
    df_fr = df_fr.sort_values("fundingTime").reset_index(drop=True)
    df = pd.merge_asof(df, df_fr[["fundingTime", "fundingRate"]], left_on="datetime", right_on="fundingTime", direction="backward")
    df["fundingRate"] = df["fundingRate"].ffill().fillna(0.0001)
    
    # 위꼬리/아래꼬리 계산
    total_range = (df["high"] - df["low"]).replace(0, 1e-9)
    body_min = df[["open", "close"]].min(axis=1)
    lower_wick = body_min - df["low"]
    lower_wick_ratio = lower_wick / total_range
    vol_mean_24 = df["volume"].rolling(24).mean().bfill()
    vol_ratio = df["volume"] / (vol_mean_24 + 1e-9)
    is_lower_wick_spike = (vol_ratio >= 2.0) & (lower_wick_ratio >= 0.40)
    
    # 펀딩비 임계치 (솔라나는 -0.025%, 나머지는 -0.010%)
    swing_th = -0.00025 if symbol == "SOLUSDT" else -0.00010
    sniper_th = -0.00015 if symbol == "SOLUSDT" else -0.00005
    
    is_settle = df["datetime"].dt.hour.isin([0, 8, 16])
    
    df["sig_swing"] = is_settle & (df["fundingRate"] <= swing_th)
    df["sig_sniper"] = (df["fundingRate"] <= sniper_th) & is_lower_wick_spike
    df["symbol"] = symbol
    
    return df


def run_chronological_unified_multicoin(
    initial_capital: float = 1_000_000.0, # 100만 원
    leverage: float = 2.0,                # 안전 2배 레버리지
    allocation_ratio: float = 0.80,       # 80% 투입, 20% 현금 보관
    fee_taker: float = 0.0005,
    slippage: float = 0.0002
):
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
    
    print("[*] 5대 코인 4개년 시계열 데이터 결합 중...")
    dfs = []
    for sym in symbols:
        coin_df = load_coin_events(sym, data_dir)
        if len(coin_df) > 0:
            dfs.append(coin_df)
            
    # 전체 시간대 추출 (1시간 단위 35,065개 시간축)
    master_timeline = pd.date_range("2021-01-01 00:00:00+00:00", "2024-12-31 23:00:00+00:00", freq="1h", tz="UTC")
    
    # 코인별 시간 인덱싱 매핑
    coin_dict = {}
    for coin_df in dfs:
        sym = coin_df["symbol"].iloc[0]
        coin_dict[sym] = coin_df.set_index("datetime")
        
    cash = initial_capital
    active_pos: Optional[ActivePosition] = None
    trade_logs = []
    equity_curve = []
    
    print(f"[*] 4개년 1계좌 1포지션 완전 충돌 배제 시뮬레이션 가동 ({len(master_timeline):,}개 시간축)...")
    
    for current_time in master_timeline:
        # 1. 기존 보유 포지션이 있으면 청산 조건 체크
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
                
                # 손절 체크
                if l <= active_pos.sl_price:
                    exit_price = active_pos.sl_price * (1.0 - slippage)
                    exit_reason = "SL"
                # 만기 체크
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
                        "mode": active_pos.mode,
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
                    
        # 2. 무포지션 상태일 때만 5개 코인 중 발생한 신호 탐색 (1순위: 스윙, 2순위: 스나이퍼)
        if active_pos is None:
            found_signal = False
            total_equity = cash
            trade_margin = total_equity * allocation_ratio
            
            # (1) 1순위: 스윙 신호가 뜬 코인이 있는지 5개 코인 전수 탐색
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
                            max_bars=24 # 24시간
                        )
                        cash -= trade_margin
                        found_signal = True
                        break # 한 코인 진입 즉시 다른 코인/신호 모두 차단!
                        
            # (2) 2순위: 스윙이 없고 여전히 비어있다면 스나이퍼 신호 탐색
            if not found_signal:
                for sym in symbols:
                    sym_data = coin_dict[sym]
                    if current_time in sym_data.index:
                        row = sym_data.loc[current_time]
                        if row["sig_sniper"]:
                            c = row["close"]
                            entry_p = c * (1.0 + slippage)
                            entry_fee = (entry_p * (trade_margin * leverage / entry_p)) * fee_taker
                            usable_margin = trade_margin - entry_fee
                            pos_size = (usable_margin * leverage) / entry_p
                            sl_rate = 0.04 if sym == "SOLUSDT" else 0.03
                            sl_p = entry_p * (1.0 - sl_rate)
                            
                            active_pos = ActivePosition(
                                symbol=sym,
                                mode="SNIPER",
                                entry_time=current_time,
                                entry_price=entry_p,
                                position_size=pos_size,
                                margin_cost=usable_margin,
                                leverage=leverage,
                                sl_price=sl_p,
                                max_bars=4 # 4시간
                            )
                            cash -= trade_margin
                            found_signal = True
                            break
                            
        # 자산 추적
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
    print("🏆 [소액 100만 원 단일 계좌] 5대 코인 x 듀얼 모드 실전 복리 백테스트 (4년간 포지션 겹침 100% 배제)")
    print("   • 조건: 안전 2배 레버리지 (2.0x) | 잔고의 80% 투입 (20% 현금 버퍼) | 수수료/슬리피지 실시간 차감")
    print("=" * 115)
    print(f"[*] 초기 시작 자본금   : ₩{initial_capital:,.0f} (100만 원)")
    print(f"[*] 4년 뒤 최종 계좌 잔고: ₩{final_balance:,.0f} (약 {final_balance/initial_capital:.2f}배 증식! 🚀)")
    print(f"[*] 실전 복리 총수익률 : {total_ret_pct:>+8.2f}%")
    print(f"[*] 계좌 최대 낙폭(MDD) : {mdd:>6.2f}% 🛡️")
    print(f"[*] 실질 체결 총 거래수: {len(trades_df)}회 (월평균 약 {len(trades_df)/48.0:.1f}회 / 연평균 {len(trades_df)/4.0:.1f}회)")
    print(f"[*] 통산 실전 승률     : {(trades_df['net_pnl']>0).mean()*100:.1f}% (총 {len(trades_df)}전 {(trades_df['net_pnl']>0).sum()}승)")
    print("-" * 115)
    print("📊 [종목별 실질 체결 기여도]")
    for sym, group in trades_df.groupby("symbol"):
        wr = (group["net_pnl"] > 0).mean() * 100.0
        print(f"    • {sym:<8}: {len(group):>3}회 체결 | 승률 {wr:>5.1f}% | 누적 기여 손익 ₩{group['net_pnl'].sum():>+10,.0f}")
    print("-" * 115)
    print("📊 [모드별 실질 체결 비중]")
    for m_type, group in trades_df.groupby("mode"):
        wr = (group["net_pnl"] > 0).mean() * 100.0
        print(f"    • Mode {m_type:<6}: {len(group):>3}회 체결 | 승률 {wr:>5.1f}% | 누적 기여 손익 ₩{group['net_pnl'].sum():>+10,.0f}")
    print("=" * 115)


if __name__ == "__main__":
    run_chronological_unified_multicoin()
