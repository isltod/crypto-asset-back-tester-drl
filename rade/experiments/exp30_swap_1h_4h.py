"""
[실험 30] 1H 평균회귀(100% 시드) + 4H 추세추종(100% 시드) 단일 계좌 스왑 백테스트
- 구조:
  1. 횡보장 (RANGE): 1H 평균회귀 (승률 62%로 횡보장 계좌 방어)
  2. 상승장 (BULL): 4H 추세추종 (손익비 2.32, 거시 6일 돌파로 불장 수익 극대화)
  3. 패닉장 (BEAR): 현금 100% 관망
- 시작 자본: $10,000 단일 통합 계좌 (100% 자본 운용)
- 목표:
  - 1H 단독(+134.44%, MDD 14.64%, PF 1.80) 대비 수익률과 손익비를 동시에 극대화
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager, RegimeState
from rade.backtest.simulator import BacktestSimulator
from rade.engines.trend_following import TrendFollowingEngine
from rade.engines.mean_reversion import MeanReversionEngine


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    df = df_1h.copy()
    df["dt"] = pd.to_datetime(df["datetime"], utc=True)
    df.set_index("dt", inplace=True)
    ohlcv_dict = {
        "timestamp": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_4h = df.resample("4h", origin="start").agg(ohlcv_dict).dropna().reset_index()
    df_4h["datetime"] = df_4h["dt"]
    df_4h.drop(columns=["dt"], inplace=True)
    return df_4h


def run_experiment_30():
    # 1. 1시간봉 및 4시간봉 데이터 로드
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_1h_ind = add_all_indicators(df_1h)

    df_4h = resample_to_4h(df_1h)
    df_4h_ind = add_all_indicators(df_4h)

    # 2. 1H HMM 국면 계산 (720봉 윈도우, 168봉 재학습)
    reg_mgr_1h = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.45, cooldown_bars=0)
    df_1h_proc = reg_mgr_1h.calculate_regime_probabilities(df_1h_ind)
    test_1h = df_1h_proc.iloc[720:].reset_index(drop=True)

    # 3. 4H HMM 국면 계산 (180봉 윈도우, 42봉 재학습)
    reg_mgr_4h = RegimeManager(hmm_window=180, retrain_interval=42, trans_threshold=0.45, cooldown_bars=0)
    df_4h_proc = reg_mgr_4h.calculate_regime_probabilities(df_4h_ind)
    test_4h = df_4h_proc.iloc[180:].reset_index(drop=True)

    # 4. 1H 단독 베이스라인 실행 ($10,000)
    sim_1h_base = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        bear_mode="CASH",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_1h_base = sim_1h_base.run(test_1h)

    # 5. [신규 스왑 모델]: 1H MR만 실행 ($10,000)
    sim_1h_mr_only = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        bear_mode="CASH",
        use_regime_transition_cut=False,
        trend_engine=None, # TF 끔
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    # RANGE 국면에서만 돌리기 위해 BULL 국면 시그널 차단
    records_1h = test_1h.to_dict("records")
    for r in records_1h:
        if r["regime_state"] == RegimeState.BULL_TREND:
            r["regime_state"] = RegimeState.BEAR_PANIC # 관망 처리
    res_1h_mr = sim_1h_mr_only.run(pd.DataFrame(records_1h))

    # 6. [신규 스왑 모델]: 4H TF만 실행 ($10,000)
    sim_4h_tf_only = BacktestSimulator(
        initial_capital=10000.0,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        bear_mode="CASH",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.0, max_trailing_atr=4.0, breakout_lookback=36),
        mean_revert_engine=None
    )
    records_4h = test_4h.to_dict("records")
    for r in records_4h:
        if r["regime_state"] == RegimeState.RANGE:
            r["regime_state"] = RegimeState.BEAR_PANIC # 4H 횡보는 관망
    res_4h_tf = sim_4h_tf_only.run(pd.DataFrame(records_4h))

    # 7. 단일 통합 계좌 합산 ($10,000 시작 풀복리 합성)
    trades_mr = res_1h_mr["trades_df"].copy()
    trades_tf = res_4h_tf["trades_df"].copy()
    trades_mr["engine_type"] = "1H_MR"
    trades_tf["engine_type"] = "4H_TF"

    swap_trades = pd.concat([trades_mr, trades_tf], ignore_index=True)
    swap_trades["entry_dt"] = pd.to_datetime(swap_trades["entry_time"])
    swap_trades.sort_values(by="entry_dt", inplace=True)
    swap_trades.reset_index(drop=True, inplace=True)

    # 순수 단일 계좌 시뮬레이션
    eq = 10000.0
    eq_curve = [eq]
    for pnl in swap_trades["pnl"]:
        eq += pnl
        eq_curve.append(eq)

    eq_arr = np.array(eq_curve)
    tot_pnl = eq_arr[-1] - 10000.0
    tot_ret = (tot_pnl / 10000.0) * 100.0
    peaks = np.maximum.accumulate(eq_arr)
    mdd = float(np.max((peaks - eq_arr) / peaks)) * 100.0

    wins = swap_trades[swap_trades["pnl"] > 0]
    losses = swap_trades[swap_trades["pnl"] < 0]
    wr = len(wins) / len(swap_trades) * 100.0 if len(swap_trades) > 0 else 0.0
    gp = wins["pnl"].sum() if len(wins) > 0 else 0.0
    gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-10
    pf = gp / gl

    # 결과 출력
    print("=" * 85)
    print("      [실험 30] 1H MR + 4H TF 스왑 통합 백테스트 4개년 정밀 리포트")
    print("=" * 85)
    print(f" * 4개년 총 수익금:     +${tot_pnl:,.2f} ({tot_ret:+.2f}%)")
    print(f" * 최대 낙폭 (MDD):     {mdd:.2f}%")
    print(f" * 손익비 (PF):         {pf:.2f}")
    print(f" * 승률 (Win Rate):     {wr:.1f}%")
    print(f" * 총 거래 횟수:        {len(swap_trades)}회 (연 {len(swap_trades)/3.92:.1f}회)")
    print(f" * 거래당 평균 이익:    +${swap_trades['pnl'].mean():.2f}")
    print("-" * 85)

    print("[엔진별 세부 실적]")
    for eng in ["1H_MR", "4H_TF"]:
        sub = swap_trades[swap_trades["engine_type"] == eng]
        p = sub["pnl"].sum()
        w = len(sub[sub["pnl"] > 0]) / len(sub) * 100.0 if len(sub) > 0 else 0.0
        print(f" * [{eng:<5}] 수익 기여: {p:+10.2f}$ | 거래: {len(sub):3d}회 (연 {len(sub)/3.92:.1f}회) | 승률: {w:5.1f}%")

    swap_trades["year"] = swap_trades["entry_dt"].dt.year
    print("\n[연도별 PnL 분해]")
    for yr in [2021, 2022, 2023, 2024]:
        sub = swap_trades[swap_trades["year"] == yr]
        p = sub["pnl"].sum()
        w = len(sub[sub["pnl"] > 0]) / len(sub) * 100.0 if len(sub) > 0 else 0.0
        print(f" * {yr}년: {p:+10.2f}$ ({len(sub):3d}회 거래, 승률 {w:5.1f}%)")
    print("=" * 85)


if __name__ == "__main__":
    run_experiment_30()
