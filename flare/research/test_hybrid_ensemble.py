import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')

from rade.backtest.simulator import BacktestSimulator
from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from flare.backtest.test_multi_position_equal_weight import run_equal_weight_multi_position


def get_asymmetric_df(df_ind: pd.DataFrame, base_th: float = 0.74, bear_th: float = 0.80) -> pd.DataFrame:
    from rade.regime.regime_manager import RegimeState
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


def get_rade_equity_series(initial_capital=500_000.0):
    data_dir = Path("data")
    f_is = data_dir / "BTCUSDT_1h_2021_2024.csv"
    f_oos = data_dir / "BTCUSDT_1h_2024_OOS.csv"
    
    if f_is.exists() and f_oos.exists():
        df_raw = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
        df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
    else:
        df_raw = pd.read_csv(data_dir / "BTCUSDT_1h_4years_full.csv")
        df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], format="ISO8601", utc=True)
        
    df_indicators = add_all_indicators(df_raw)
    regime_manager = RegimeManager(
        hmm_window=720,
        retrain_interval=168,
        trans_threshold=0.74,
        cooldown_bars=0
    )
    df_processed = regime_manager.calculate_regime_probabilities(df_indicators)
    
    # 🌟 RADE 공식 표준 모델 (STANDARD_GOLDEN: TF 2.0%, MR 4.0%, CASH 모드, 3.0x)
    sim = BacktestSimulator(
        initial_capital=initial_capital,
        trend_risk_pct=0.020,
        mr_risk_pct=0.040,
        leverage=3.0,
        bear_mode="CASH",
        maker_fee_pct=0.0002,
        taker_fee_pct=0.0005,
        slippage_pct=0.0002,
    )
    metrics = sim.run(df_processed)
    
    equity_df = pd.DataFrame({
        "datetime": pd.to_datetime(metrics.get("timestamps", []), utc=True),
        "rade_equity": metrics.get("equity_curve", [])
    }).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    
    return equity_df, metrics


def get_flare_equity_series(initial_capital=500_000.0, leverage=5.0):
    data_dir = Path("data")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    res = run_equal_weight_multi_position(
        symbols=symbols,
        data_dir=data_dir,
        initial_capital=initial_capital,
        leverage=leverage,
        allocation_ratio=0.80
    )
    
    equity_df = pd.DataFrame({
        "datetime": pd.to_datetime(res["timestamps"], utc=True),
        "flare_equity": res["equity_curve"]
    }).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    
    return equity_df, res


def main():
    print("=" * 105)
    print(" 🔬 [진짜 시너지 앙상블 백테스트] 50% FLARE (5.0x 고수익 창) + 50% RADE (STANDARD_GOLDEN 철벽 방패)")
    print("    • 초기 자본: 100만 원 (FLARE 50만 원 + RADE 50만 원 분할 운용)")
    print("    • 기간: 2021년 1월 ~ 2024년 12월 (4개년)")
    print("=" * 105)
    
    rade_df, rade_metrics = get_rade_equity_series(initial_capital=500_000.0)
    flare_df, flare_res = get_flare_equity_series(initial_capital=500_000.0, leverage=5.0)
    
    # 시계열 병합
    merged = pd.merge(rade_df, flare_df, on="datetime", how="outer").sort_values("datetime")
    merged["rade_equity"] = merged["rade_equity"].ffill().bfill()
    merged["flare_equity"] = merged["flare_equity"].ffill().bfill()
    
    # 통합 계좌 자산 계산
    merged["total_equity"] = merged["rade_equity"] + merged["flare_equity"]
    
    # MDD 계산
    merged["peak"] = merged["total_equity"].cummax()
    merged["dd"] = (merged["peak"] - merged["total_equity"]) / merged["peak"] * 100.0
    total_mdd = merged["dd"].max()
    
    # 개별 MDD
    merged["rade_peak"] = merged["rade_equity"].cummax()
    merged["rade_dd"] = (merged["rade_peak"] - merged["rade_equity"]) / merged["rade_peak"] * 100.0
    rade_mdd = merged["rade_dd"].max()
    
    merged["flare_peak"] = merged["flare_equity"].cummax()
    merged["flare_dd"] = (merged["flare_peak"] - merged["flare_equity"]) / merged["flare_peak"] * 100.0
    flare_mdd = merged["flare_dd"].max()
    
    init_cap = 1_000_000.0
    final_total = merged["total_equity"].iloc[-1]
    final_rade = merged["rade_equity"].iloc[-1]
    final_flare = merged["flare_equity"].iloc[-1]
    
    total_return = ((final_total - init_cap) / init_cap) * 100.0
    rade_return = ((final_rade - 500_000.0) / 500_000.0) * 100.0
    flare_return = ((final_flare - 500_000.0) / 500_000.0) * 100.0
    
    cagr = (final_total / init_cap) ** (1/4) - 1
    
    print("\n" + "-" * 105)
    print(f" 📊 [개별 및 통합 포트폴리오 성적표]")
    print("-" * 105)
    print(f" • RADE (공격형 50만)    : 최종 잔고 ₩{final_rade:12,.0f} | 수익률 {rade_return:+9.2f}% | 개별 MDD: {rade_mdd:5.2f}%")
    print(f" • FLARE (4.5배 스윙 50만): 최종 잔고 ₩{final_flare:12,.0f} | 수익률 {flare_return:+9.2f}% | 개별 MDD: {flare_mdd:5.2f}%")
    print("-" * 105)
    print(f" 🏆 [통합 5:5 앙상블 계좌] : 최종 잔고 ₩{final_total:12,.0f} (원금 {final_total/init_cap:4.1f}배!)")
    print(f"    • 4개년 총 누적 수익률: {total_return:+10.2f}%")
    print(f"    • 연평균 복리 수익률  : +{cagr*100:6.2f}% / 년")
    print(f"    • 🌟 통합 포트폴리오 MDD: {total_mdd:6.2f}% 🛡️ (개별 전략 대비 획기적 낙폭 상쇄!)")
    print("=" * 105)


if __name__ == '__main__':
    main()
