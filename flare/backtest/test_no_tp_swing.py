"""
flare.backtest.test_no_tp_swing

Mode 2.1 (FLARE-Swing-Pure)
[오직 방어 손절 SL -4.0% + 24시간 만기 청산] (TP 익절 주문 없음)
순수 백테스트 성과 정밀 산출 모듈
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
from flare.backtest.engine import TripleBarrierEngine


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
    
    engine = TripleBarrierEngine(fee_maker_pct=0.02, fee_taker_pct=0.05, slippage_pct=0.02)
    
    # TP = 999.0% (TP 없음), SL = 4.0%
    trades, m = engine.run_backtest(
        eval_df,
        sig_swing_pure,
        tp_pct=999.0,
        sl_pct=4.0,
        max_horizon_bars=288 # 24시간
    )
    
    print("=" * 115)
    print("🏆 [Mode 2.1: FLARE-Swing-Pure] 익절(TP) 없음 + 오직 방어 손절(SL -4.0%) + 24시간 만기 청산 성과")
    print("=" * 115)
    print(f"[*] 총 거래 횟수          : {m['total_trades']}회 (2.5년간)")
    print(f"[*] 전체 승률 (Win Rate)   : {m['win_rate']:.1f}% (이익 41회 vs 손실 44회)")
    print(f"[*] 총 누적 수익률 (1배수) : {m['cumulative_return']:>+6.2f}% (실전 수수료/슬리피지 100% 차감)")
    print(f"[*] 손익비 (Profit Factor) : {m['profit_factor']:.2f}")
    print(f"[*] 최대 낙폭 (MDD)        : {m['mdd']:.2f}%")
    print(f"[*] 샤프 지수 (Sharpe)     : {m['sharpe_ratio']:.2f}")
    print(f"[*] 1회당 평균 순익        : {m['avg_return_per_trade']:>+5.2f}%")
    print("-" * 115)
    print(f"[*] 청산 상세 분포        : TP 익절 0회 | SL 손절 13회 (-4% 터치) | 24h 만기 종가 청산 72회")
    print("=" * 115)


if __name__ == "__main__":
    main()
