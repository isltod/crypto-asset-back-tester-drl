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


def simulate_rebalanced_portfolio(rade_eq_df, flare_eq_df, initial_capital=1_000_000.0, rebalance_freq="M", target_rade_ratio=0.5):
    """
    주기적 리밸런싱 시뮬레이터
    rebalance_freq: 'M' (매월 1일), 'Q' (분기별), '6M' (반기별), None (리밸런싱 없음)
    """
    merged = pd.merge(rade_eq_df, flare_eq_df, on="datetime", how="outer").sort_values("datetime").ffill().bfill()
    merged = merged.reset_index(drop=True)
    
    # 전략별 1바 수익률(Return Series) 계산
    merged["rade_ret"] = merged["rade_equity"].pct_change().fillna(0.0)
    merged["flare_ret"] = merged["flare_equity"].pct_change().fillna(0.0)
    
    # 리밸런싱 플래그 설정
    merged["dt"] = merged["datetime"]
    if rebalance_freq == "M":
        period_series = merged["dt"].dt.to_period("M")
        merged["rebalance_event"] = (period_series != period_series.shift(1))
    elif rebalance_freq == "Q":
        period_series = merged["dt"].dt.to_period("Q")
        merged["rebalance_event"] = (period_series != period_series.shift(1))
    elif rebalance_freq == "6M":
        half_year = merged["dt"].dt.year.astype(str) + "_" + merged["dt"].dt.month.map(lambda m: "H1" if m <= 6 else "H2")
        merged["rebalance_event"] = (half_year != half_year.shift(1))
    elif rebalance_freq == "W":
        period_series = merged["dt"].dt.to_period("W")
        merged["rebalance_event"] = (period_series != period_series.shift(1))
    else:
        merged["rebalance_event"] = False
        
    merged.loc[0, "rebalance_event"] = True  # 시작점 초기화
    
    # 시계열 순회 시뮬레이션
    rade_val = initial_capital * target_rade_ratio
    flare_val = initial_capital * (1.0 - target_rade_ratio)
    
    total_curve = []
    rade_curve = []
    flare_curve = []
    
    for idx, row in merged.iterrows():
        # 1. 1바 수익률 적용
        if idx > 0:
            rade_val *= (1.0 + row["rade_ret"])
            flare_val *= (1.0 + row["flare_ret"])
            
        # 2. 리밸런싱 이벤트 발생 시 5:5 (또는 목표비율) 재배분
        if row["rebalance_event"] and idx > 0:
            tot = rade_val + flare_val
            rade_val = tot * target_rade_ratio
            flare_val = tot * (1.0 - target_rade_ratio)
            
        total_curve.append(rade_val + flare_val)
        rade_curve.append(rade_val)
        flare_curve.append(flare_val)
        
    merged["total_equity"] = total_curve
    merged["rade_alloc_eq"] = rade_curve
    merged["flare_alloc_eq"] = flare_curve
    
    # MDD 계산
    peak = merged["total_equity"].cummax()
    dd = (peak - merged["total_equity"]) / peak
    mdd_pct = dd.max() * 100.0
    
    final_eq = merged["total_equity"].iloc[-1]
    total_ret = (final_eq / initial_capital - 1.0) * 100.0
    cagr = ((final_eq / initial_capital) ** (1.0 / 4.0) - 1.0) * 100.0
    calmar = total_ret / mdd_pct if mdd_pct > 0 else 0
    
    return {
        "freq": rebalance_freq,
        "ratio": f"{int(target_rade_ratio*100)}:{int((1-target_rade_ratio)*100)}",
        "final_equity": final_eq,
        "total_return_pct": total_ret,
        "cagr": cagr,
        "mdd_pct": mdd_pct,
        "calmar": calmar,
        "equity_df": merged
    }


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    data_dir = Path("data")
    
    print("=" * 115)
    print(" 🔬 [RADE 표준 + FLARE 5x] 주기적 리밸런싱(Rebalancing) 정밀 비교 실험")
    print("    • 초기 자본: 100만 원 (4개년: 2021.01 ~ 2024.12)")
    print("=" * 115)
    
    # 1. RADE 표준
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
    
    # 2. FLARE 5x
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    flare_res = run_equal_weight_multi_position(
        symbols=symbols,
        data_dir=data_dir,
        initial_capital=500_000.0,
        leverage=5.0,
        allocation_ratio=0.80
    )
    
    df_rade = pd.DataFrame({"datetime": pd.to_datetime(rade_res["timestamps"], utc=True), "rade_equity": rade_res["equity_curve"]}).drop_duplicates("datetime")
    df_flare = pd.DataFrame({"datetime": pd.to_datetime(flare_res["timestamps"], utc=True), "flare_equity": flare_res["equity_curve"]}).drop_duplicates("datetime")
    
    # 3. 다양한 비율 x 주기 전수 그리드 스캔
    ratios = [
        ("🌟 균형형 (5:5)", 0.5),
        ("⚖️ 중립형 (6:4)", 0.6),
        ("🛡️ 안전형 (7:3)", 0.7),
        ("🏰 철벽형 (8:2)", 0.8),
    ]
    
    freqs = [
        ("리밸런싱 없음 (단순방치)", None),
        ("반기별 리밸런싱 (6개월)", "6M"),
        ("분기별 리밸런싱 (3개월)", "Q"),
        ("월별 리밸런싱 (1개월)", "M"),
    ]
    
    print(f"{'포트폴리오 배분 비율':<20} | {'리밸런싱 주기':<24} | {'최종 잔고':<16} | {'4개년 총수익':<12} | {'연평균(CAGR)':<12} | {'최대낙폭(MDD)':<14} | {'칼마비율':<8}")
    print("-" * 125)
    
    for r_name, r_ratio in ratios:
        for f_name, f_freq in freqs:
            res = simulate_rebalanced_portfolio(df_rade, df_flare, initial_capital=1_000_000.0, rebalance_freq=f_freq, target_rade_ratio=r_ratio)
            print(f"{r_name:<20} | {f_name:<24} | ₩{res['final_equity']:>12,.0f} | {res['total_return_pct']:>+10.2f}% | {res['cagr']:>+10.2f}% | {res['mdd_pct']:>11.2f}% | {res['calmar']:>6.2f}")
        print("-" * 125)
        
    print("=" * 125)


if __name__ == "__main__":
    main()
