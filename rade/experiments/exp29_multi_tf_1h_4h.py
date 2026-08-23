"""
[실험 29] 1H + 4H 멀티 타임프레임 병렬 포트폴리오 백테스트
- 1. 1H 단독 RADE (조합 C) 성과
- 2. 4H 단독 RADE 성과 (4시간봉 거시 36봉=6일 돌파 & HMM)
- 3. 1H + 4H 병렬 합산 포트폴리오 (자본 50:50 분배) 성과
- 검증 목적:
  - 1H와 4H가 상호 독립적으로 거래 기회를 발생시켜 총 거래수가 연 40회 -> 연 60회 이상으로 증가하는가?
  - 15m과 달리 수수료 마모 없이 수익률과 MDD가 우수하게 유지/개선되는가?
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from rade.backtest.simulator import BacktestSimulator
from rade.engines.trend_following import TrendFollowingEngine
from rade.engines.mean_reversion import MeanReversionEngine


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """1시간봉 데이터를 4시간봉(4H)으로 정밀 OHLCV 리샘플링"""
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


def run_experiment_29():
    print("=" * 95)
    print("      [실험 29] 1H + 4H 멀티 타임프레임 병렬 포트폴리오 4개년 백테스트")
    print("=" * 95)

    # 1. 1시간봉 데이터 로드
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_1h_ind = add_all_indicators(df_1h)

    # 2. 4시간봉 데이터 생성
    df_4h = resample_to_4h(df_1h)
    df_4h_ind = add_all_indicators(df_4h)

    print(f"[Data] 1시간봉 데이터: 총 {len(df_1h_ind):,}개 캔들")
    print(f"[Data] 4시간봉 데이터: 총 {len(df_4h_ind):,}개 캔들 (정밀 리샘플링 완료)")

    # 3. 1H RADE 실행 (자본 $5,000 기준)
    print("\n[Simulation 1] 1H RADE 엔진 가동 중...")
    reg_mgr_1h = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.45, cooldown_bars=0)
    df_1h_proc = reg_mgr_1h.calculate_regime_probabilities(df_1h_ind)
    test_1h = df_1h_proc.iloc[720:].reset_index(drop=True)

    sim_1h = BacktestSimulator(
        initial_capital=5000.0,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        bear_mode="CASH",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_1h = sim_1h.run(test_1h)

    # 4. 4H RADE 실행 (자본 $5,000 기준, 4H 파라미터 최적화: 윈도우 180봉=30일, 4H 재학습 42봉=1주)
    print("[Simulation 2] 4H RADE 엔진 가동 중...")
    reg_mgr_4h = RegimeManager(hmm_window=180, retrain_interval=42, trans_threshold=0.45, cooldown_bars=0)
    df_4h_proc = reg_mgr_4h.calculate_regime_probabilities(df_4h_ind)
    test_4h = df_4h_proc.iloc[180:].reset_index(drop=True)

    sim_4h = BacktestSimulator(
        initial_capital=5000.0,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        bear_mode="CASH",
        use_regime_transition_cut=False,
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.0, max_trailing_atr=4.0, breakout_lookback=36),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=12) # 12봉 = 48h
    )
    res_4h = sim_4h.run(test_4h)

    # 5. 포트폴리오 합산 분석 (1H $5,000 + 4H $5,000 = 총 $10,000 시작)
    trades_1h = res_1h["trades_df"]
    trades_4h = res_4h["trades_df"]
    
    trades_1h["tf"] = "1H"
    trades_4h["tf"] = "4H"
    all_trades = pd.concat([trades_1h, trades_4h], ignore_index=True)
    all_trades["entry_dt"] = pd.to_datetime(all_trades["entry_time"])
    all_trades.sort_values(by="entry_dt", inplace=True)
    all_trades.reset_index(drop=True, inplace=True)

    total_pnl = all_trades["pnl"].sum()
    final_equity = 10000.0 + total_pnl
    total_ret_pct = (total_pnl / 10000.0) * 100.0

    wins = all_trades[all_trades["pnl"] > 0]
    losses = all_trades[all_trades["pnl"] < 0]
    wr = len(wins) / len(all_trades) * 100.0 if len(all_trades) > 0 else 0.0
    gp = wins["pnl"].sum() if len(wins) > 0 else 0.0
    gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-10
    pf = gp / gl

    # 자산 곡선 및 MDD 산출
    cum_eq = 10000.0 + all_trades["pnl"].cumsum()
    peaks = np.maximum.accumulate(cum_eq)
    drawdowns = (peaks - cum_eq) / peaks * 100.0
    port_mdd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # 6. 결과 출력 및 1:1 비교
    print("\n" + "=" * 95)
    print("                [ 1H 단독 vs 4H 단독 vs 1H+4H 포트폴리오 1:1 비교표 ]")
    print("=" * 95)
    print(f"{'구분':<22} | {'1H 단독 ($10k)':<20} | {'4H 단독 ($10k)':<20} | {'1H + 4H 포트폴리오 ($10k)':<25}")
    print("-" * 95)
    
    # 1H 단독 $10k 기준 환산 수치
    h1_trades = len(trades_1h)
    h4_trades = len(trades_4h)
    tot_trades = len(all_trades)

    h1_pnl_str = f"+${(res_1h['final_equity'] - 5000.0) * 2:,.2f}"
    h4_pnl_str = f"+${(res_4h['final_equity'] - 5000.0) * 2:,.2f}"
    tot_pnl_str = f"+${total_pnl:,.2f} (+{total_ret_pct:.2f}%)"
    h1_mdd_str = f"{res_1h['mdd_pct']:.2f}%"
    h4_mdd_str = f"{res_4h['mdd_pct']:.2f}%"
    port_mdd_str = f"{port_mdd:.2f}%"
    h1_pf_str = f"{res_1h['profit_factor']:.2f}"
    h4_pf_str = f"{res_4h['profit_factor']:.2f}"
    port_pf_str = f"{pf:.2f}"
    h1_wr_str = f"{res_1h['win_rate_pct']:.1f}%"
    h4_wr_str = f"{res_4h['win_rate_pct']:.1f}%"
    port_wr_str = f"{wr:.1f}%"
    h1_tr_str = f"{h1_trades}회 (연 {h1_trades/3.92:.1f}회)"
    h4_tr_str = f"{h4_trades}회 (연 {h4_trades/3.92:.1f}회)"
    port_tr_str = f"{tot_trades}회 (연 {tot_trades/3.92:.1f}회)"

    print(f"{'4개년 총 수익금':<22} | {h1_pnl_str:<20} | {h4_pnl_str:<20} | {tot_pnl_str:<25}")
    print(f"{'최대 낙폭 (MDD)':<22} | {h1_mdd_str:<20} | {h4_mdd_str:<20} | {port_mdd_str:<25}")
    print(f"{'손익비 (PF)':<22} | {h1_pf_str:<20} | {h4_pf_str:<20} | {port_pf_str:<25}")
    print(f"{'승률 (Win Rate)':<22} | {h1_wr_str:<20} | {h4_wr_str:<20} | {port_wr_str:<25}")
    print(f"{'총 거래 횟수':<22} | {h1_tr_str:<20} | {h4_tr_str:<20} | {port_tr_str:<25}")
    h1_avg_pnl = trades_1h["pnl"].mean() * 2 if len(trades_1h) > 0 else 0.0
    h4_avg_pnl = trades_4h["pnl"].mean() * 2 if len(trades_4h) > 0 else 0.0
    all_avg_pnl = all_trades["pnl"].mean() if len(all_trades) > 0 else 0.0

    print(f"{'거래당 평균 이익':<22} | {f'+${h1_avg_pnl:.2f}':<20} | {f'+${h4_avg_pnl:.2f}':<20} | {f'+${all_avg_pnl:.2f}':<25}")
    print("-" * 95)

    # 타임프레임별 기여도
    print("\n[포트폴리오 내 타임프레임별 성과 기여도]")
    for tf in ["1H", "4H"]:
        sub = all_trades[all_trades["tf"] == tf]
        sub_pnl = sub["pnl"].sum()
        sub_wr = len(sub[sub["pnl"] > 0]) / len(sub) * 100.0 if len(sub) > 0 else 0.0
        print(f" * [{tf} 엔진] 수익 기여: {sub_pnl:+9.2f}$ | 거래: {len(sub):3d}회 (연 {len(sub)/3.92:.1f}회) | 승률: {sub_wr:5.1f}%")

    # 연도별 분해
    all_trades["year"] = all_trades["entry_dt"].dt.year
    print("\n[1H + 4H 포트폴리오 연도별 합산 성과]")
    for yr in [2021, 2022, 2023, 2024]:
        sub = all_trades[all_trades["year"] == yr]
        yr_pnl = sub["pnl"].sum()
        yr_wr = len(sub[sub["pnl"] > 0]) / len(sub) * 100.0 if len(sub) > 0 else 0.0
        print(f" * {yr}년: {yr_pnl:+10.2f}$ ({len(sub):3d}회 거래, 승률 {yr_wr:5.1f}%)")

    print("=" * 95)


if __name__ == "__main__":
    run_experiment_29()
