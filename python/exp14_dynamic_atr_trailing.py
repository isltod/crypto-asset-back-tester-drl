"""
[실험 14] 동적 변동성 적응형 트레일링 스탑 (Dynamic ATR Trailing) 독립 검증
- 문제점: 고정 3.0x ATR 트레일링 스탑은 변동성이 폭발하는 랠리 및 숏스퀴즈 구간에서 노이즈에 조기 청산되어 대세 추세 수익을 놓침
- 해결책: 변동성 비율(ATR / ATR_MA50)에 따라 트레일링 버퍼를 동적으로 조절
- 검증 기간: 2021.01 ~ 2024.06 (3.5년 장기 스트레스 테스트)
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from python.utils.indicators import add_all_indicators
from python.regime.regime_manager import RegimeManager
from python.risk.position_manager import Position, PositionSide
from python.engines.mean_reversion import MeanReversionEngine
from python.engines.trend_following import TrendFollowingEngine
from python.backtest.simulator import BacktestSimulator


class DynamicATRTrendEngine(TrendFollowingEngine):
    """변동성 비율(ATR / ATR_MA50) 기반 동적 트레일링 스탑 추세추종 엔진"""

    def __init__(
        self,
        base_trailing_atr: float = 3.0,
        mode: str = "FIXED",           # "FIXED", "DYNAMIC_ALL", "DYNAMIC_CAPPED", "DYNAMIC_SHORT_ONLY"
        max_multiplier_cap: float = 4.5, # 동적 상한선 배수
        adx_threshold: float = 25.0,
        breakout_lookback: int = 36,
        sl_atr_multiplier: float = 1.5,
        min_vol_mult: float = 0.5,
        min_body_ratio: float = 0.45,
    ):
        super().__init__(
            adx_threshold=adx_threshold,
            breakout_lookback=breakout_lookback,
            sl_atr_multiplier=sl_atr_multiplier,
            trailing_atr_multiplier=base_trailing_atr,
            min_vol_mult=min_vol_mult,
            min_body_ratio=min_body_ratio,
        )
        self.base_trailing_atr = base_trailing_atr
        self.mode = mode
        self.max_multiplier_cap = max_multiplier_cap

    def _get_effective_multiplier(self, side: PositionSide, curr: Dict[str, Any]) -> float:
        atr = curr.get('atr', 1.0)
        atr_ma50 = curr.get('atr_ma50', atr)
        if atr_ma50 <= 0:
            atr_ma50 = atr

        vol_ratio = max(1.0, atr / (atr_ma50 + 1e-10))

        if self.mode == "FIXED":
            return self.base_trailing_atr

        elif self.mode == "DYNAMIC_ALL":
            # 3.0 * vol_ratio (상한 없음)
            return self.base_trailing_atr * vol_ratio

        elif self.mode == "DYNAMIC_CAPPED":
            # 3.0 * min(vol_ratio, cap_ratio) -> 최대 max_multiplier_cap
            max_ratio = self.max_multiplier_cap / self.base_trailing_atr
            return self.base_trailing_atr * min(vol_ratio, max_ratio)

        elif self.mode == "DYNAMIC_SHORT_ONLY":
            if side == PositionSide.SHORT:
                max_ratio = self.max_multiplier_cap / self.base_trailing_atr
                return self.base_trailing_atr * min(vol_ratio, max_ratio)
            else:
                return self.base_trailing_atr

        return self.base_trailing_atr

    def update_position_fast(self, pos: Position, curr: Dict[str, Any]) -> Dict[str, Any]:
        """보수적 체결 + 동적 ATR 트레일링 스탑"""
        high = curr['high']
        low = curr['low']
        open_p = curr['open']
        atr = curr['atr']

        eff_mult = self._get_effective_multiplier(pos.side, curr)

        if pos.side == PositionSide.LONG:
            # 1. 직전 확정 손절가 선검사 (보수적)
            if low <= pos.sl_price:
                exit_price = min(pos.sl_price, open_p) if open_p < pos.sl_price else pos.sl_price
                return {"action": "TRAILING_STOP", "exit_price": exit_price, "closed_ratio": 1.0, "is_maker": False}

            # 2. 다음 봉을 위한 트레일링 스탑 갱신
            if high > pos.highest_price:
                pos.highest_price = high

            trailing_sl = pos.highest_price - (atr * eff_mult)
            pos.sl_price = max(pos.sl_price, trailing_sl)

        elif pos.side == PositionSide.SHORT:
            # 1. 직전 확정 손절가 선검사 (보수적)
            if high >= pos.sl_price:
                exit_price = max(pos.sl_price, open_p) if open_p > pos.sl_price else pos.sl_price
                return {"action": "TRAILING_STOP", "exit_price": exit_price, "closed_ratio": 1.0, "is_maker": False}

            # 2. 다음 봉을 위한 트레일링 스탑 갱신
            if low < pos.lowest_price:
                pos.lowest_price = low

            trailing_sl = pos.lowest_price + (atr * eff_mult)
            pos.sl_price = min(pos.sl_price, trailing_sl)

        return {"action": "NONE", "exit_price": 0.0, "closed_ratio": 0.0, "is_maker": False}


def run_experiment_14():
    print("=== [실험 14] 동적 변동성 적응형 트레일링(Dynamic ATR) 3.5년 독립 검증 시작 ===")

    cache_file = os.path.join("data", "BTCUSDT_1h_2021_2024.csv")
    df_raw = pd.read_csv(cache_file)
    df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)

    df_ind = add_all_indicators(df_raw)
    # 50봉 ATR 이동평균 추가
    df_ind['atr_ma50'] = df_ind['atr'].rolling(window=50).mean()

    manager = RegimeManager(hmm_window=720, retrain_interval=168, hysteresis_upper=0.65, hysteresis_lower=0.35, cooldown_bars=3)
    df_proc = manager.calculate_regime_probabilities(df_ind)
    test_df = df_proc.dropna(subset=['regime_trend_prob']).reset_index(drop=True)

    # 평균회귀 엔진은 기준선으로 완벽히 고정
    standard_mr_engine = MeanReversionEngine()

    configs = [
        {
            "name": "1. 기준선 (고정 3.0x ATR)",
            "tf_engine": DynamicATRTrendEngine(base_trailing_atr=3.0, mode="FIXED"),
        },
        {
            "name": "2. 동적 ATR 무제한 (3.0x * ATR_Ratio)",
            "tf_engine": DynamicATRTrendEngine(base_trailing_atr=3.0, mode="DYNAMIC_ALL"),
        },
        {
            "name": "3. 동적 ATR 상한제한 (3.0x ~ 4.5x)",
            "tf_engine": DynamicATRTrendEngine(base_trailing_atr=3.0, mode="DYNAMIC_CAPPED", max_multiplier_cap=4.5),
        },
        {
            "name": "4. 동적 ATR 상한제한 (3.0x ~ 4.0x)",
            "tf_engine": DynamicATRTrendEngine(base_trailing_atr=3.0, mode="DYNAMIC_CAPPED", max_multiplier_cap=4.0),
        },
        {
            "name": "5. 숏 전용 동적 ATR (롱 3.0x / 숏 3.0x~4.5x)",
            "tf_engine": DynamicATRTrendEngine(base_trailing_atr=3.0, mode="DYNAMIC_SHORT_ONLY", max_multiplier_cap=4.5),
        },
    ]

    summary_rows = []
    equity_curves = {}

    for cfg in configs:
        sim = BacktestSimulator(
            initial_capital=10000.0,
            risk_per_trade_pct=0.02, # 2.0% Risk
            leverage=3.0,
            trend_engine=cfg['tf_engine'],
            mean_revert_engine=standard_mr_engine
        )
        res = sim.run(test_df)
        df_t = res['trades_df']

        if not df_t.empty:
            tf_trades = df_t[df_t['engine'] == "TREND_FOLLOWING"]
            tf_pnl = tf_trades['pnl'].sum() if not tf_trades.empty else 0.0
            tf_wr = (len(tf_trades[tf_trades['pnl'] > 0]) / len(tf_trades)) * 100.0 if not tf_trades.empty else 0.0
            tf_cnt = len(tf_trades)

            # 롱 / 숏 분해
            tf_long = tf_trades[tf_trades['side'] == "LONG"]
            tf_long_pnl = tf_long['pnl'].sum() if not tf_long.empty else 0.0
            tf_short = tf_trades[tf_trades['side'] == "SHORT"]
            tf_short_pnl = tf_short['pnl'].sum() if not tf_short.empty else 0.0

            df_t['year'] = pd.to_datetime(df_t['exit_time']).dt.year
            pnl_2022 = df_t[df_t['year'] == 2022]['pnl'].sum()
            pnl_2023 = df_t[df_t['year'] == 2023]['pnl'].sum()
        else:
            tf_pnl, tf_wr, tf_cnt, tf_long_pnl, tf_short_pnl, pnl_2022, pnl_2023 = 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0

        summary_rows.append({
            "설정": cfg['name'],
            "3.5년 총수익률": f"{res['total_return_pct']:+.2f}%",
            "MDD": f"{res['mdd_pct']:.2f}%",
            "PF": f"{res['profit_factor']:.2f}",
            "추세 승률": f"{tf_wr:.1f}%",
            "추세 롱 ($)": f"${tf_long_pnl:+,.2f}",
            "추세 숏 ($)": f"${tf_short_pnl:+,.2f}",
            "추세 총PnL ($)": f"${tf_pnl:+,.2f}",
            "2022년 ($)": f"${pnl_2022:+,.2f}",
            "2023년 ($)": f"${pnl_2023:+,.2f}",
        })
        equity_curves[cfg['name']] = res['equity_curve']

    df_sum = pd.DataFrame(summary_rows)
    print("\n" + "=" * 125)
    print("               [ 실험 14: 동적 변동성 적응형 트레일링 스탑 성과 비교표 (3.5년 장기) ]               ")
    print("=" * 125)
    print(df_sum.to_string(index=False))
    print("=" * 125)

    # 차트 저장
    plt.figure(figsize=(14, 7))
    for label, eq in equity_curves.items():
        plt.plot(eq, linewidth=1.6, label=label)

    plt.axhline(10000.0, color='gray', linestyle='--', label='Initial ($10k)')
    plt.title("RADE Experiment 14: Dynamic ATR Trailing Stop Comparison (3.5 Years)", fontsize=13, fontweight='bold')
    plt.xlabel("Hours")
    plt.ylabel("Account Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    chart_path = os.path.join("data", "exp14_dynamic_atr_trailing_plot.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Done] 동적 ATR 비교 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_experiment_14()
