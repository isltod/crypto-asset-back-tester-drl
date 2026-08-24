"""
[실험 49] STANDARD_GOLDEN vs EXTREME_CASH vs EXTREME_SHORT 자산 곡선 및 MDD 정밀 시각화 차트 생성
- 상단: 4개년 자산 곡선 (Equity Curve, $) + 주요 국면 주석
- 하단: 수중 낙폭 곡선 (Underwater Drawdown, %)
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# 한글 폰트 설정 (Windows 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager, RegimeState
from rade.backtest.simulator import BacktestSimulator
from rade.engines.trend_following import TrendFollowingEngine
from rade.engines.mean_reversion import MeanReversionEngine


def get_asymmetric_df(df_ind: pd.DataFrame, base_th: float = 0.74, bear_th: float = 0.80) -> pd.DataFrame:
    reg_raw = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.30, cooldown_bars=0)
    df_raw = reg_raw.calculate_regime_probabilities(df_ind)
    
    curr = RegimeState.RANGE
    asym_states = []
    for idx, row in df_raw.iterrows():
        p_r = row["p_range"]
        p_u = row["p_bull"]
        p_d = row["p_bear"]
        if p_d >= bear_th and p_d >= p_u and p_d >= p_r:
            curr = RegimeState.BEAR_PANIC
        elif p_u >= base_th and p_u >= p_r and p_u >= p_d:
            curr = RegimeState.BULL_TREND
        elif p_r >= base_th and p_r >= p_u and p_r >= p_d:
            curr = RegimeState.RANGE
        asym_states.append(curr)
    df_raw["regime_state"] = asym_states
    return df_raw


def run_plot():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    # 1. 국면 데이터 생성
    reg_mgr_74 = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.74, cooldown_bars=0)
    df_proc_74 = reg_mgr_74.calculate_regime_probabilities(df_ind)
    test_df_cash = df_proc_74.iloc[720:].reset_index(drop=True)

    df_asym = get_asymmetric_df(df_ind, base_th=0.74, bear_th=0.80)
    test_df_asym = df_asym.iloc[720:].reset_index(drop=True)

    # 모델 1: STANDARD_GOLDEN (2.0% x 4.0%, CASH)
    sim_m1 = BacktestSimulator(
        initial_capital=10000.0, trend_risk_pct=0.020, mr_risk_pct=0.040, leverage=3.0, bear_mode="CASH",
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_m1 = sim_m1.run(test_df_cash)

    # 모델 2: EXTREME_CASH (4.0% x 4.0%, CASH)
    sim_m2 = BacktestSimulator(
        initial_capital=10000.0, trend_risk_pct=0.040, mr_risk_pct=0.040, leverage=3.0, bear_mode="CASH",
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_m2 = sim_m2.run(test_df_cash)

    # 모델 3: EXTREME_SHORT (4.0% x 4.0% + 80% 숏)
    sim_m3 = BacktestSimulator(
        initial_capital=10000.0, trend_risk_pct=0.040, mr_risk_pct=0.040, leverage=3.0, bear_mode="SHORT",
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_m3 = sim_m3.run(test_df_asym)

    # 날짜 축 동기화
    dts = pd.to_datetime(res_m1["timestamps"])
    eq_m1 = np.array(res_m1["equity_curve"])
    eq_m2 = np.array(res_m2["equity_curve"])
    eq_m3 = np.array(res_m3["equity_curve"])

    # 벤치마크 (BTC 단순 보유)
    btc_prices = test_df_cash["close"].values
    btc_norm = (btc_prices / btc_prices[0]) * 10000.0

    # Drawdown 계산 함수
    def get_dd(eq):
        peak = np.maximum.accumulate(eq)
        return (eq - peak) / (peak + 1e-10) * 100.0

    dd_m1 = get_dd(eq_m1)
    dd_m2 = get_dd(eq_m2)
    dd_m3 = get_dd(eq_m3)

    # 플롯 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

    # 상단: 자산 곡선 (Equity Curve)
    ax1.plot(dts, eq_m3, label=f"EXTREME_SHORT (4%x4% + 80%숏) : ${res_m3['final_equity']:,.0f} (+{res_m3['total_return_pct']:.1f}%) [MDD {res_m3['mdd_pct']:.1f}%]", color="#e74c3c", linewidth=2.2)
    ax1.plot(dts, eq_m2, label=f"EXTREME_CASH  (4%x4% 관망)    : ${res_m2['final_equity']:,.0f} (+{res_m2['total_return_pct']:.1f}%) [MDD {res_m2['mdd_pct']:.1f}%]", color="#f39c12", linewidth=1.8, linestyle="--")
    ax1.plot(dts, eq_m1, label=f"STANDARD_GOLDEN (2%x4% 공식표준): ${res_m1['final_equity']:,.0f} (+{res_m1['total_return_pct']:.1f}%) [MDD {res_m1['mdd_pct']:.1f}%]", color="#2ecc71", linewidth=2.5)
    ax1.plot(dts, btc_norm, label="BTC Buy & Hold (단순 보유 벤치마크)", color="#95a5a6", linewidth=1.2, linestyle=":")

    ax1.set_title("RADE 4개년 자산 곡선 비교 : [공식 표준] vs [극한 숏 풀베팅 (+555%)]", fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel("계좌 총자산 ($)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${int(y):,}"))

    # 주요 이벤트 주석 달기
    # 2022년 하락장 숏 폭등 지점
    ax1.annotate(
        "🔥 2022년 크립토 윈터 폭락장\n• 숏으로만 +$21,195 대폭발!\n• 자산 $2.5만 ──► $4.7만 수직상승",
        xy=(pd.Timestamp("2022-08-01"), 35000),
        xytext=(pd.Timestamp("2021-10-01"), 65000),
        arrowprops=dict(facecolor='#e74c3c', shrink=0.05, width=2, headwidth=8),
        fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#e74c3c", lw=1.5)
    )

    # 2024년 불장 숏 스퀴즈 MDD 36% 발생 지점
    ax1.annotate(
        "🩸 2024년 불장 숏 스퀴즈 손절\n• 고점 $6.5만 ──► $4.1만 추락!\n• 【MDD 36.23%】 발생 지점",
        xy=(pd.Timestamp("2024-06-01"), 43000),
        xytext=(pd.Timestamp("2023-08-01"), 18000),
        arrowprops=dict(facecolor='#c0392b', shrink=0.05, width=2, headwidth=8),
        fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffcdd2", ec="#c0392b", lw=1.5)
    )

    # 하단: 낙폭 곡선 (Underwater Drawdown)
    ax2.plot(dts, dd_m3, label="EXTREME_SHORT Drawdown", color="#e74c3c", linewidth=1.5)
    ax2.plot(dts, dd_m1, label="STANDARD_GOLDEN Drawdown", color="#2ecc71", linewidth=2.0)
    ax2.fill_between(dts, dd_m3, 0, color="#e74c3c", alpha=0.25)
    ax2.fill_between(dts, dd_m1, 0, color="#2ecc71", alpha=0.35)

    ax2.axhline(-16.17, color="#27ae60", linestyle=":", label="표준 MDD (16.17%)")
    ax2.axhline(-36.23, color="#c0392b", linestyle=":", label="극한 MDD (36.23%)")

    ax2.set_title("수중 낙폭 곡선 (Underwater Drawdown, %)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("낙폭 (%)", fontsize=12)
    ax2.set_xlabel("일자", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "extreme_short_comparison_chart.png")
    plt.savefig(out_path, dpi=150)
    print(f"[Done] 차트 저장 완료: {out_path}")

    # 아티팩트 디렉토리에도 복사
    artifact_dir = r"C:\Users\wolf\.gemini\antigravity-ide\brain\e205d749-95e5-4fca-873d-b84d4d5be0b8"
    if os.path.exists(artifact_dir):
        import shutil
        artifact_path = os.path.join(artifact_dir, "extreme_short_comparison_chart.png")
        shutil.copy(out_path, artifact_path)
        print(f"[Done] 아티팩트 복사 완료: {artifact_path}")


if __name__ == "__main__":
    run_plot()
