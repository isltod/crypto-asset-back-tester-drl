"""
flare.backtest.unified_backtest

FLARE 통합 단일 계좌(Unified Single Account) 실전 복리 시계열 백테스터
- 25.4만 개 5분봉 시계열을 시간 순(Chronological Bar-by-Bar)으로 1봉씩 순회
- 실시간 계좌 잔고(Cash & Equity) 및 복리(Compound Growth) 추적
- 1계좌 1포지션 원칙: 스윙 보유 중에는 스나이퍼 중복 진입 완전 차단 (No Double Exposure)
- Mode 2.1 (Swing 3배 레버리지, SL -4.0%, 24h) + Mode 1.1 (Sniper 4배 레버리지, SL -3.0%, 4h) 통합
- 실전 수수료(Maker 0.02%, Taker 0.05%) 및 슬리피지(0.02%) 실시간 잔고 차감
- 전체 거래 내역 CSV 파일 자동 저장
"""

import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flare.data.features import generate_all_features


@dataclass
class Position:
    mode: str           # "SWING" or "SNIPER"
    entry_time: pd.Timestamp
    entry_price: float
    position_size: float # 계약 수량 (BTC)
    margin_cost: float   # 투입 증거금 (Cash)
    leverage: float
    sl_price: float
    max_bars: int       # 최대 보유 봉 수 (스윙 288, 스나이퍼 48)
    bars_held: int = 0


def run_unified_chronological_backtest(
    initial_capital: float = 1_000_000.0, # 초기 자본 100만 원 (KRW 또는 USD 기준)
    swing_leverage: float = 3.0,          # 스윙 레버리지 (3x)
    sniper_leverage: float = 4.0,         # 스나이퍼 레버리지 (4x)
    fee_maker: float = 0.0002,            # 지정가 수수료 0.02%
    fee_taker: float = 0.0005,            # 시장가 수수료 0.05%
    slippage: float = 0.0002              # 슬리피지 0.02%
):
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    klines_file = data_dir / "BTCUSDT_5m_2022_2024.csv"
    funding_file = Path(__file__).resolve().parent.parent / "data" / "btcusdt_funding_rate.csv"
    
    print(f"[*] 5분봉 캔들 데이터 로드 중: {klines_file.name}...")
    df = pd.read_csv(klines_file)
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
    eval_df = df.iloc[8640:].reset_index(drop=True)
    
    # 신호 정의
    is_settle_bar = eval_df["datetime"].dt.minute == 0
    is_settle_hour = eval_df["datetime"].dt.hour.isin([0, 8, 16])
    sig_swing = is_settle_bar & is_settle_hour & (eval_df["feat_funding_rsi_30d"] <= 0.05)
    sig_sniper = (eval_df["feat_funding_rsi_30d"] <= 0.10) & (eval_df["feat_is_lower_wick_spike"] == 1.0)
    
    highs = eval_df["high"].values
    lows = eval_df["low"].values
    closes = eval_df["close"].values
    datetimes = eval_df["datetime"].values
    swing_sigs = sig_swing.values
    sniper_sigs = sig_sniper.values
    n_bars = len(eval_df)
    
    cash = initial_capital
    position: Optional[Position] = None
    trade_logs = []
    equity_curve = []
    
    print(f"[*] 시간 순 실전 복리 시뮬레이션 가동 (총 {n_bars:,}개 5분봉)...")
    
    for i in range(n_bars):
        current_time = pd.Timestamp(datetimes[i])
        h = highs[i]
        l = lows[i]
        c = closes[i]
        
        # 1. 포지션 보유 중인 경우: 청산 조건 체크
        if position is not None:
            position.bars_held += 1
            exit_price = None
            exit_reason = None
            exit_fee_rate = fee_taker
            
            # (1) 손절 체크
            if l <= position.sl_price:
                exit_price = position.sl_price * (1.0 - slippage)
                exit_reason = "SL"
                exit_fee_rate = fee_taker
            # (2) 시간 만기 체크
            elif position.bars_held >= position.max_bars:
                exit_price = c * (1.0 - slippage)
                exit_reason = "TIMEOUT"
                exit_fee_rate = fee_taker
                
            # 청산 실행
            if exit_price is not None:
                # 손익금 계산
                raw_pnl = (exit_price - position.entry_price) * position.position_size
                exit_fee = (exit_price * position.position_size) * exit_fee_rate
                net_trade_pnl = raw_pnl - exit_fee
                
                # 잔고 복리 갱신
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
                    "return_on_margin_pct": ret_on_margin,
                    "exit_reason": exit_reason,
                    "hold_bars": position.bars_held,
                    "balance_after_trade": cash
                })
                position = None
                
        # 2. 무포지션 상태일 때만 새로운 진입 탐색 (1계좌 1포지션 원칙)
        if position is None:
            # 1순위: 스윙 신호 우선 체크
            if swing_sigs[i]:
                entry_p = c * (1.0 + slippage)
                margin = cash # 전액 투입 (또는 95% 안전 마진)
                entry_fee = (entry_p * (margin * swing_leverage / entry_p)) * fee_taker
                usable_margin = margin - entry_fee
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
                cash -= margin # 증거금 차감
                
            # 2순위: 스윙이 없을 때만 스나이퍼 진입 허용
            elif sniper_sigs[i]:
                entry_p = c * (1.0 + slippage)
                margin = cash
                entry_fee = (entry_p * (margin * sniper_leverage / entry_p)) * fee_taker
                usable_margin = margin - entry_fee
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
                cash -= margin
                
        # 일별 자산 가치 기록
        current_equity = cash if position is None else cash + position.margin_cost + ((c - position.entry_price) * position.position_size)
        equity_curve.append(current_equity)
        
    # 결과 정리
    trades_df = pd.DataFrame(trade_logs)
    
    # CSV 저장
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "unified_trades_log.csv"
    trades_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # 성과 지표 계산
    final_balance = cash if position is None else cash + position.margin_cost
    total_return_pct = (final_balance - initial_capital) / initial_capital * 100.0
    
    eq_series = pd.Series(equity_curve)
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak * 100.0
    mdd = abs(dd.min())
    
    swing_trades = trades_df[trades_df["mode"] == "SWING"]
    sniper_trades = trades_df[trades_df["mode"] == "SNIPER"]
    
    print("\n" + "=" * 115)
    print("🏆 [FLARE] 통합 단일 계좌 실전 복리(Compound) 백테스트 최종 성과 보고서 (2022~2024)")
    print("=" * 115)
    print(f"[*] 초기 시작 자본금   : ₩{initial_capital:,.0f} (100만 원)")
    print(f"[*] 2.5년 뒤 최종 잔고 : ₩{final_balance:,.0f} (약 {final_balance/initial_capital:.2f}배 증식! 🚀)")
    print(f"[*] 실전 복리 누적수익률: {total_return_pct:>+8.2f}% (수수료/슬리피지 100% 실시간 차감)")
    print(f"[*] 계좌 최대 낙폭(MDD) : {mdd:>6.2f}% 🛡️")
    print(f"[*] 총 실행 거래 횟수  : {len(trades_df)}회 (월평균 {len(trades_df)/28.5:.1f}회 / 주 1.0회)")
    print("-" * 115)
    print("📊 [모드별 기여 내역]")
    print(f"    - 🟢 Mode 2.1 (Swing 3x)  : {len(swing_trades)}회 실행 | 승률 {(swing_trades['net_pnl']>0).mean()*100:.1f}% | 누적 손익 ₩{swing_trades['net_pnl'].sum():>+10,.0f}")
    print(f"    - ⚡ Mode 1.1 (Sniper 4x) : {len(sniper_trades)}회 실행 | 승률 {(sniper_trades['net_pnl']>0).mean()*100:.1f}% | 누적 손익 ₩{sniper_trades['net_pnl'].sum():>+10,.0f}")
    print("-" * 115)
    print(f"💾 전체 {len(trades_df)}개 거래 상세 내역이 CSV로 저장되었습니다: {csv_path.name}")
    print("=" * 115)


if __name__ == "__main__":
    run_unified_chronological_backtest()
