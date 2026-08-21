"""
[실험 8] 3.5년 롱(LONG) vs 숏(SHORT) 및 엔진별 심층 분해 분석 스크립트
- 전체 롱 vs 숏 성과 비교
- 엔진별(평균회귀 vs 추세추종) × 방향별(롱 vs 숏) 4분면 매트릭스 성과
- 연도별(특히 2022년 대폭락장) 롱 vs 숏 손익 분해
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from rade.data.binance_fetcher import BinanceFuturesFetcher
from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from rade.backtest.simulator import BacktestSimulator


def run_long_short_analysis():
    print("=== [실험 8] 3.5년 롱(LONG) vs 숏(SHORT) 정밀 분해 분석 시작 ===")

    # 1. 3.5년 데이터 로드 및 백테스트
    cache_file = os.path.join("data", "BTCUSDT_1h_2021_2024.csv")
    df_raw = pd.read_csv(cache_file)
    df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)

    df_ind = add_all_indicators(df_raw)
    manager = RegimeManager(hmm_window=720, retrain_interval=168, hysteresis_upper=0.65, hysteresis_lower=0.35, cooldown_bars=3)
    df_proc = manager.calculate_regime_probabilities(df_ind)
    test_df = df_proc.dropna(subset=['regime_trend_prob']).reset_index(drop=True)

    sim = BacktestSimulator(initial_capital=10000.0, risk_per_trade_pct=0.02, leverage=3.0)
    res = sim.run(test_df)
    trades_df = res['trades_df']

    if trades_df.empty:
        print("[Error] 거래 기록이 없습니다.")
        return

    trades_df['year'] = pd.to_datetime(trades_df['exit_time']).dt.year

    # 2. [전체] 롱 vs 숏 성과 비교
    print("\n" + "=" * 75)
    print("                [ 1. 전체 롱(LONG) vs 숏(SHORT) 성과 비교 ]                ")
    print("=" * 75)
    ls_summary = []
    for side in ["LONG", "SHORT"]:
        sub = trades_df[trades_df['side'] == side]
        wins = sub[sub['pnl'] > 0]
        losses = sub[sub['pnl'] < 0]
        wr = (len(wins) / len(sub)) * 100.0 if len(sub) > 0 else 0.0
        pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else 0.0
        pnl = sub['pnl'].sum()
        gross_win = wins['pnl'].sum()
        gross_loss = abs(losses['pnl'].sum())

        ls_summary.append({
            "포지션": side,
            "거래 횟수": f"{len(sub)}회 ({len(sub)/len(trades_df)*100:.1f}%)",
            "승률": f"{wr:.1f}%",
            "누적 PnL ($)": f"${pnl:+,.2f}",
            "Profit Factor": f"{pf:.2f}",
            "총 이익 ($)": f"${gross_win:,.2f}",
            "총 손실 ($)": f"${gross_loss:,.2f}",
        })
    print(pd.DataFrame(ls_summary).to_string(index=False))
    print("=" * 75)

    # 3. [엔진 × 방향] 4분면 매트릭스 분석
    print("\n" + "=" * 75)
    print("            [ 2. 엔진별 × 포지션 방향별 (4분면 매트릭스) 성과 ]            ")
    print("=" * 75)
    matrix_rows = []
    for eng in ["MEAN_REVERSION", "TREND_FOLLOWING"]:
        for side in ["LONG", "SHORT"]:
            sub = trades_df[(trades_df['engine'] == eng) & (trades_df['side'] == side)]
            if sub.empty:
                continue
            wins = sub[sub['pnl'] > 0]
            losses = sub[sub['pnl'] < 0]
            wr = (len(wins) / len(sub)) * 100.0
            pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else 0.0
            pnl = sub['pnl'].sum()

            matrix_rows.append({
                "엔진": eng,
                "방향": side,
                "거래 횟수": f"{len(sub)}회",
                "승률": f"{wr:.1f}%",
                "Profit Factor": f"{pf:.2f}",
                "누적 PnL ($)": f"${pnl:+,.2f}",
            })
    print(pd.DataFrame(matrix_rows).to_string(index=False))
    print("=" * 75)

    # 4. [연도별] 롱 vs 숏 손익 분해 (2021 ~ 2024)
    print("\n" + "=" * 75)
    print("                  [ 3. 연도별 롱(LONG) vs 숏(SHORT) 손익 추이 ]                  ")
    print("=" * 75)
    yr_rows = []
    for yr, group in trades_df.groupby('year'):
        l_sub = group[group['side'] == "LONG"]
        s_sub = group[group['side'] == "SHORT"]

        l_pnl = l_sub['pnl'].sum() if not l_sub.empty else 0.0
        s_pnl = s_sub['pnl'].sum() if not s_sub.empty else 0.0
        l_wr = (len(l_sub[l_sub['pnl'] > 0]) / len(l_sub)) * 100.0 if not l_sub.empty else 0.0
        s_wr = (len(s_sub[s_sub['pnl'] > 0]) / len(s_sub)) * 100.0 if not s_sub.empty else 0.0

        yr_rows.append({
            "연도": f"{yr}년",
            "LONG 거래 (승률)": f"{len(l_sub)}회 ({l_wr:.1f}%)",
            "LONG PnL ($)": f"${l_pnl:+,.2f}",
            "SHORT 거래 (승률)": f"{len(s_sub)}회 ({s_wr:.1f}%)",
            "SHORT PnL ($)": f"${s_pnl:+,.2f}",
            "연간 합산 PnL ($)": f"${(l_pnl + s_pnl):+,.2f}",
        })
    print(pd.DataFrame(yr_rows).to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    run_long_short_analysis()
