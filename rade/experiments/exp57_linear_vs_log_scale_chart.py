"""
[실험 57] 몬스터 100x의 선형 스케일(Linear, 진짜 달러 체감) vs 로그 스케일(Log) 1:1 비교 차트
- 왼쪽: 선형 스케일 (Linear Scale) -> $11만에서 $2만으로 -80% 수직 낭떠러지 추락의 공포를 그대로 시각화
- 오른쪽: 로그 스케일 (Log Scale) -> 복리 성장률 관점
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

    # 1. 표준 모델
    reg_mgr_74 = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.74, cooldown_bars=0)
    df_proc_74 = reg_mgr_74.calculate_regime_probabilities(df_ind)
    test_df_cash = df_proc_74.iloc[720:].reset_index(drop=True)

    sim_m1 = BacktestSimulator(
        initial_capital=10000.0, trend_risk_pct=0.020, mr_risk_pct=0.040, leverage=3.0, bear_mode="CASH",
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    res_m1 = sim_m1.run(test_df_cash)

    # 2. 몬스터 모델
    df_asym = get_asymmetric_df(df_ind, base_th=0.74, bear_th=0.80)
    test_df_asym = df_asym.iloc[720:].reset_index(drop=True)

    sim_m2 = BacktestSimulator(
        initial_capital=10000.0, trend_risk_pct=0.040, mr_risk_pct=0.200, leverage=100.0, bear_mode="SHORT",
        trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
        mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
    )
    sim_m2.pos_manager.max_leverage = 100.0
    sim_m2.pos_manager.default_leverage = 100.0
    res_m2 = sim_m2.run(test_df_asym)

    dts = pd.to_datetime(res_m1["timestamps"])
    eq_m1 = np.array(res_m1["equity_curve"])
    eq_m2 = np.array(res_m2["equity_curve"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # 왼쪽: 선형 스케일 (Linear Scale - 실제 통장 잔고 체감)
    ax1.plot(dts, eq_m2, label=f"몬스터 100x: ${res_m2['final_equity']:,.0f} (+{res_m2['total_return_pct']:.0f}%)", color="#8e44ad", linewidth=2.3)
    ax1.plot(dts, eq_m1, label=f"공식 표준:  ${res_m1['final_equity']:,.0f} (+{res_m1['total_return_pct']:.0f}%)", color="#2ecc71", linewidth=2.5)
    ax1.set_title("【선형 스케일 (Linear)】: 실제 통장 잔고의 수직 낙하 체감", fontsize=14, fontweight='bold', pad=12)
    ax1.set_ylabel("계좌 총자산 ($)", fontsize=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${int(y):,}"))
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=11)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 선형 차트 주석: 수직 절벽 추락
    ax1.annotate(
        "★ MDD 80.67% 수직 절벽 추락!\n• $110,000(1억5천) ──► $21,000(2천8백) 폭락!\n• 통장 잔고 80%가 그대로 증발하는 절벽",
        xy=(pd.Timestamp("2024-04-01"), 25000),
        xytext=(pd.Timestamp("2021-08-01"), 150000),
        arrowprops=dict(facecolor='#c0392b', shrink=0.05, width=2.5, headwidth=9),
        fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#c0392b", lw=1.5)
    )

    # 오른쪽: 로그 스케일 (Log Scale - 배수/비율 관점)
    ax2.plot(dts, eq_m2, label=f"몬스터 100x [MDD {res_m2['mdd_pct']:.1f}%]", color="#8e44ad", linewidth=2.3)
    ax2.plot(dts, eq_m1, label=f"공식 표준   [MDD {res_m1['mdd_pct']:.1f}%]", color="#2ecc71", linewidth=2.5)
    ax2.set_yscale('log')
    ax2.set_title("【로그 스케일 (Log)】: 비율 압축 착시로 완만해 보임", fontsize=14, fontweight='bold', pad=12)
    ax2.set_ylabel("계좌 총자산 ($, 로그)", fontsize=12)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${int(y):,}"))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "linear_vs_log_scale_chart.png")
    plt.savefig(out_path, dpi=150)
    print(f"[Done] 차트 저장 완료: {out_path}")

    # 아티팩트 디렉토리 복사
    artifact_dir = r"C:\Users\wolf\.gemini\antigravity-ide\brain\e205d749-95e5-4fca-873d-b84d4d5be0b8"
    if os.path.exists(artifact_dir):
        import shutil
        artifact_path = os.path.join(artifact_dir, "linear_vs_log_scale_chart.png")
        shutil.copy(out_path, artifact_path)
        print(f"[Done] 아티팩트 복사 완료: {artifact_path}")


if __name__ == "__main__":
    run_plot()
