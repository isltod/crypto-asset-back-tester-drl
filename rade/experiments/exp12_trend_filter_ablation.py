"""
[실험 12] 추세추종 엔진 필터 다이어트 (Ablation Study) 독립 검증
- 문제점: 5중 과잉 필터(200 EMA + 36봉 박스권 + ADX 25 + 거래량 1.5배 + 캔들 몸통 45%)로 인해 비트코인 주요 추세 탑승 기회 상실
- 목적: 필터를 단계적으로 완화/제거하여 3.5년(2021~2024) 장기 수익률 및 승률, MDD 변화 추적
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Optional, List
from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from rade.risk.position_manager import Position, PositionSide
from rade.engines.trend_following import TrendFollowingEngine
from rade.engines.mean_reversion import MeanReversionEngine
from rade.backtest.simulator import BacktestSimulator


class ModularTrendEngine(TrendFollowingEngine):
    """필터 ON/OFF 제어가 가능한 모듈형 추세추종 엔진"""

    def __init__(
        self,
        use_ema200: bool = True,
        use_box_breakout: bool = True,
        breakout_lookback: int = 36,
        use_adx: bool = True,
        adx_threshold: float = 25.0,
        use_volume: bool = True,
        min_vol_mult: float = 0.5,
        use_body_ratio: bool = True,
        min_body_ratio: float = 0.45,
        trailing_atr_multiplier: float = 3.0,
        sl_atr_multiplier: float = 1.5,
    ):
        super().__init__(
            adx_threshold=adx_threshold,
            breakout_lookback=breakout_lookback,
            sl_atr_multiplier=sl_atr_multiplier,
            trailing_atr_multiplier=trailing_atr_multiplier,
            min_vol_mult=min_vol_mult,
            min_body_ratio=min_body_ratio,
        )
        self.use_ema200 = use_ema200
        self.use_box_breakout = use_box_breakout
        self.use_adx = use_adx
        self.use_volume = use_volume
        self.use_body_ratio = use_body_ratio

    def check_entry_signal_fast(self, i: int, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        lookback_req = max(self.breakout_lookback if self.use_box_breakout else 1, 200 if self.use_ema200 else 1)
        if i < lookback_req:
            return None

        curr = records[i]
        if curr.get('is_cooldown', False) or curr.get('adx') is None:
            return None

        close = curr['close']
        open_p = curr['open']
        high = curr['high']
        low = curr['low']
        atr = curr['atr']
        adx = curr['adx']
        plus_di = curr['plus_di']
        minus_di = curr['minus_di']
        vol_change = curr.get('vol_change', 0.0)
        ema200 = curr.get('ema200', close)

        # 1. 캔들 몸통 비율 필터
        if self.use_body_ratio:
            candle_range = high - low
            body_size = abs(close - open_p)
            if candle_range > 0 and (body_size / candle_range) < self.min_body_ratio:
                return None

        # 2. 박스권 고가/저가
        if self.use_box_breakout:
            prev_slice = records[i - self.breakout_lookback : i]
            box_high = max(r['high'] for r in prev_slice)
            box_low = min(r['low'] for r in prev_slice)
        else:
            box_high = 0.0
            box_low = 99999999.0

        # 조건 체크 함수
        # 롱 조건
        long_ema_ok = (close > ema200) if self.use_ema200 else True
        long_box_ok = (close > box_high) if self.use_box_breakout else True
        long_adx_ok = (adx >= self.adx_threshold and plus_di > minus_di) if self.use_adx else (plus_di > minus_di)
        long_vol_ok = (vol_change >= self.min_vol_mult) if self.use_volume else True

        if long_ema_ok and long_box_ok and long_adx_ok and long_vol_ok:
            sl_price = close - (atr * self.sl_atr_multiplier)
            return {
                "side": PositionSide.LONG,
                "sl_price": sl_price,
                "tp1_price": None,
                "tp2_price": None,
                "engine": self.name,
            }

        # 숏 조건
        short_ema_ok = (close < ema200) if self.use_ema200 else True
        short_box_ok = (close < box_low) if self.use_box_breakout else True
        short_adx_ok = (adx >= self.adx_threshold and minus_di > plus_di) if self.use_adx else (minus_di > plus_di)
        short_vol_ok = (vol_change >= self.min_vol_mult) if self.use_volume else True

        if short_ema_ok and short_box_ok and short_adx_ok and short_vol_ok:
            sl_price = close + (atr * self.sl_atr_multiplier)
            return {
                "side": PositionSide.SHORT,
                "sl_price": sl_price,
                "tp1_price": None,
                "tp2_price": None,
                "engine": self.name,
            }

        return None


def run_experiment_12():
    print("=== [실험 12] 추세추종 필터 다이어트 (Ablation Study) 3.5년 검증 시작 ===")

    cache_file = os.path.join("data", "BTCUSDT_1h_2021_2024.csv")
    df_raw = pd.read_csv(cache_file)
    df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)

    df_ind = add_all_indicators(df_raw)
    manager = RegimeManager(hmm_window=720, retrain_interval=168, hysteresis_upper=0.65, hysteresis_lower=0.35, cooldown_bars=3)
    df_proc = manager.calculate_regime_probabilities(df_ind)
    test_df = df_proc.dropna(subset=['regime_trend_prob']).reset_index(drop=True)

    configs = [
        {
            "name": "1. 기준선 (5중 필터 전체)",
            "engine": ModularTrendEngine(use_ema200=True, use_box_breakout=True, breakout_lookback=36, use_adx=True, adx_threshold=25.0, use_volume=True, use_body_ratio=True),
        },
        {
            "name": "2. 몸통 필터 제거 (4중)",
            "engine": ModularTrendEngine(use_ema200=True, use_box_breakout=True, breakout_lookback=36, use_adx=True, adx_threshold=25.0, use_volume=True, use_body_ratio=False),
        },
        {
            "name": "3. 몸통+거래량 제거 (3중: EMA+돌파+ADX25)",
            "engine": ModularTrendEngine(use_ema200=True, use_box_breakout=True, breakout_lookback=36, use_adx=True, adx_threshold=25.0, use_volume=False, use_body_ratio=False),
        },
        {
            "name": "4. 몸통+거래량 제거 & ADX 완화(20)",
            "engine": ModularTrendEngine(use_ema200=True, use_box_breakout=True, breakout_lookback=36, use_adx=True, adx_threshold=20.0, use_volume=False, use_body_ratio=False),
        },
        {
            "name": "5. 핵심 2중 필터 (EMA+36봉 돌파, ADX제거)",
            "engine": ModularTrendEngine(use_ema200=True, use_box_breakout=True, breakout_lookback=36, use_adx=False, use_volume=False, use_body_ratio=False),
        },
        {
            "name": "6. 3중 필터 + 24봉(1일) 돌파",
            "engine": ModularTrendEngine(use_ema200=True, use_box_breakout=True, breakout_lookback=24, use_adx=True, adx_threshold=25.0, use_volume=False, use_body_ratio=False),
        },
    ]

    summary_rows = []
    equity_curves = {}

    for cfg in configs:
        sim = BacktestSimulator(
            initial_capital=10000.0,
            risk_per_trade_pct=0.02, # 2.0% Risk
            leverage=3.0,
            trend_engine=cfg['engine']
        )
        res = sim.run(test_df)
        df_t = res['trades_df']

        # 추세추종 거래만 분리
        if not df_t.empty:
            tf_trades = df_t[df_t['engine'] == "TREND_FOLLOWING"]
            tf_pnl = tf_trades['pnl'].sum() if not tf_trades.empty else 0.0
            tf_wr = (len(tf_trades[tf_trades['pnl'] > 0]) / len(tf_trades)) * 100.0 if not tf_trades.empty else 0.0
            tf_cnt = len(tf_trades)

            # 2022년 하락장 및 2023년 불장 손익
            df_t['year'] = pd.to_datetime(df_t['exit_time']).dt.year
            pnl_2022 = df_t[df_t['year'] == 2022]['pnl'].sum()
            pnl_2023 = df_t[df_t['year'] == 2023]['pnl'].sum()
        else:
            tf_pnl, tf_wr, tf_cnt, pnl_2022, pnl_2023 = 0.0, 0.0, 0, 0.0, 0.0

        summary_rows.append({
            "설정": cfg['name'],
            "3.5년 총수익률": f"{res['total_return_pct']:+.2f}%",
            "MDD": f"{res['mdd_pct']:.2f}%",
            "PF": f"{res['profit_factor']:.2f}",
            "총 거래": f"{res['total_trades']}회",
            "추세 거래": f"{tf_cnt}회",
            "추세 승률": f"{tf_wr:.1f}%",
            "추세 PnL ($)": f"${tf_pnl:+,.2f}",
            "2022년 ($)": f"${pnl_2022:+,.2f}",
            "2023년 ($)": f"${pnl_2023:+,.2f}",
        })
        equity_curves[cfg['name']] = res['equity_curve']

    df_sum = pd.DataFrame(summary_rows)
    print("\n" + "=" * 115)
    print("               [ 실험 12: 추세추종 엔진 필터 다이어트 성과 비교표 (3.5년 장기) ]               ")
    print("=" * 115)
    print(df_sum.to_string(index=False))
    print("=" * 115)

    # 차트 저장
    plt.figure(figsize=(14, 7))
    for label, eq in equity_curves.items():
        plt.plot(eq, linewidth=1.6, label=label)

    plt.axhline(10000.0, color='gray', linestyle='--', label='Initial ($10k)')
    plt.title("RADE Experiment 12: Trend Filter Ablation Study (3.5 Years)", fontsize=13, fontweight='bold')
    plt.xlabel("Hours")
    plt.ylabel("Account Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    chart_path = os.path.join("data", "exp12_trend_filter_ablation_plot.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Done] 필터 다이어트 비교 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_experiment_12()
