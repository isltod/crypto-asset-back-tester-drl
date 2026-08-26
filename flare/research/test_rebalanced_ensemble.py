import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from rade.backtest.simulator import BacktestSimulator
from flare.backtest.test_multi_position_equal_weight import run_equal_weight_multi_position


def test_rebalanced_ensemble():
    sys.stdout.reconfigure(encoding='utf-8')
    data_dir = Path("data")
    
    # 1. RADE 표준 시뮬레이션
    f_is = data_dir / "BTCUSDT_1h_2021_2024.csv"
    f_oos = data_dir / "BTCUSDT_1h_2024_OOS.csv"
    df_raw = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
    df_indicators = add_all_indicators(df_raw)
    
    regime_manager = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.74, cooldown_bars=0)
    df_processed = regime_manager.calculate_regime_probabilities(df_indicators)
    
    sim = BacktestSimulator(
        initial_capital=500_000.0,
        trend_risk_pct=0.020,
        mr_risk_pct=0.040,
        leverage=3.0,
        bear_mode="CASH",
        maker_fee_pct=0.0002,
        taker_fee_pct=0.0005,
        slippage_pct=0.0002,
    )
    rade_res = sim.run(df_processed)
    
    # 2. FLARE 5x 시뮬레이션
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    flare_res = run_equal_weight_multi_position(
        symbols=symbols,
        data_dir=data_dir,
        initial_capital=500_000.0,
        leverage=5.0,
        allocation_ratio=0.80
    )
    
    df_rade = pd.DataFrame({"datetime": pd.to_datetime(rade_res["timestamps"], utc=True), "rade_eq": rade_res["equity_curve"]}).drop_duplicates("datetime")
    df_flare = pd.DataFrame({"datetime": pd.to_datetime(flare_res["timestamps"], utc=True), "flare_eq": flare_res["equity_curve"]}).drop_duplicates("datetime")
    
    merged = pd.merge(df_rade, df_flare, on="datetime", how="outer").sort_values("datetime").ffill().bfill()
    merged["total_static"] = merged["rade_eq"] + merged["flare_eq"]
    
    # Static MDD
    s_peak = merged["total_static"].cummax()
    s_dd = (s_peak - merged["total_static"]) / s_peak
    static_mdd = s_dd.max() * 100
    
    print("=" * 90)
    print(" 📊 [RADE 표준(방패) + FLARE 5x(창)] 포트폴리오 결합 분석")
    print("=" * 90)
    print(f" 1. RADE 표준 단독  : ₩50만 ──► ₩{merged['rade_eq'].iloc[-1]:,.0f} (+{rade_res['total_return_pct']:.1f}%) | 단독 MDD: {rade_res['mdd_pct']:.2f}%")
    print(f" 2. FLARE 5x 단독   : ₩50만 ──► ₩{merged['flare_eq'].iloc[-1]:,.0f} (+{flare_res['return_pct']:.1f}%) | 단독 MDD: {flare_res['mdd']:.2f}%")
    print(f" 3. 통합 5:5 계좌   : ₩100만 ──► ₩{merged['total_static'].iloc[-1]:,.0f} (+{(merged['total_static'].iloc[-1]/1000000.0 - 1)*100:.1f}%) | 통합 MDD: {static_mdd:.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    test_rebalanced_ensemble()
