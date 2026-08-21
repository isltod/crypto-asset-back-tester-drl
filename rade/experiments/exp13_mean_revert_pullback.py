"""
[실험 13] 평균회귀 엔진의 눌림목(Pullback) 전환 독립 검증
- 문제점: 횡보 국면에서 대세 추세(200 EMA)와 반대 방향으로 무분별하게 진입하여 손실 누적
- 해결책: 200 EMA 상단에서는 [하단 반등 롱(눌림목)]만, 200 EMA 하단에서는 [상단 반락 숏(반등 매도)]만 허용
- 평가 기간: 2021.01 ~ 2024.06 (3.5년 장기 스트레스 테스트)
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
from rade.engines.mean_reversion import MeanReversionEngine
from rade.engines.trend_following import TrendFollowingEngine
from rade.backtest.simulator import BacktestSimulator


class TrendAlignedMeanReversionEngine(MeanReversionEngine):
    """200 EMA 대세 추세와 동조하는 눌림목/반등 매매 전용 평균회귀 엔진"""

    def __init__(
        self,
        mode: str = "BASELINE",  # "BASELINE", "PULLBACK_ONLY", "LONG_ONLY_ABOVE_EMA"
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        sl_atr_multiplier: float = 1.2,
        tp1_ratio: float = 0.8,
        max_holding_bars: int = 12,
    ):
        super().__init__(
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            sl_atr_multiplier=sl_atr_multiplier,
            tp1_ratio=tp1_ratio,
            max_holding_bars=max_holding_bars,
        )
        self.mode = mode

    def check_entry_signal_fast(self, i: int, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if i < 200:
            return None

        curr = records[i]
        prev = records[i - 1]

        if curr.get('is_cooldown', False) or curr.get('bb_lower') is None:
            return None

        # 1. 밴드 수축 필터
        bw = curr.get('bb_bandwidth', 0.0)
        bw_ma50 = curr.get('bb_bandwidth_ma50', 0.0)
        if bw_ma50 > 0 and bw > (bw_ma50 * 1.15):
            return None

        close = curr['close']
        open_p = curr['open']
        low = curr['low']
        high = curr['high']
        atr = curr['atr']
        rsi = curr['rsi']
        ema200 = curr.get('ema200', close)

        bb_middle = curr['bb_middle']
        bb_upper = curr['bb_upper']
        bb_lower = curr['bb_lower']

        prev_low = prev['low']
        prev_bb_lower = prev['bb_lower']
        prev_high = prev['high']
        prev_bb_upper = prev['bb_upper']

        # 롱 조건 (쌍바닥 반등):
        # 모드별 롱 허용 여부:
        allow_long = True
        if self.mode in ["PULLBACK_ONLY", "LONG_ONLY_ABOVE_EMA"]:
            allow_long = (close > ema200)

        if allow_long:
            if prev_low <= prev_bb_lower and low >= (prev_low * 0.998) and close > open_p and rsi <= self.rsi_oversold:
                sl_price = close - (atr * self.sl_atr_multiplier)
                return {
                    "side": PositionSide.LONG,
                    "sl_price": sl_price,
                    "tp1_price": bb_middle,
                    "tp2_price": bb_upper,
                    "engine": self.name,
                }

        # 숏 조건 (쌍봉 반락):
        # 모드별 숏 허용 여부:
        allow_short = True
        if self.mode == "PULLBACK_ONLY":
            allow_short = (close < ema200)
        elif self.mode == "LONG_ONLY_ABOVE_EMA":
            allow_short = False  # 횡보장 숏 완전 금지 (롱 전용)

        if allow_short:
            if prev_high >= prev_bb_upper and high <= (prev_high * 1.002) and close < open_p and rsi >= self.rsi_overbought:
                sl_price = close + (atr * self.sl_atr_multiplier)
                return {
                    "side": PositionSide.SHORT,
                    "sl_price": sl_price,
                    "tp1_price": bb_middle,
                    "tp2_price": bb_lower,
                    "engine": self.name,
                }

        return None


def run_experiment_13():
    print("=== [실험 13] 평균회귀 엔진 눌림목(Pullback) 전환 3.5년 독립 검증 시작 ===")

    cache_file = os.path.join("data", "BTCUSDT_1h_2021_2024.csv")
    df_raw = pd.read_csv(cache_file)
    df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)

    df_ind = add_all_indicators(df_raw)
    manager = RegimeManager(hmm_window=720, retrain_interval=168, hysteresis_upper=0.65, hysteresis_lower=0.35, cooldown_bars=3)
    df_proc = manager.calculate_regime_probabilities(df_ind)
    test_df = df_proc.dropna(subset=['regime_trend_prob']).reset_index(drop=True)

    # 추세추종 엔진은 5중 필터 기준선으로 완벽히 고정
    standard_tf_engine = TrendFollowingEngine(
        adx_threshold=25.0,
        breakout_lookback=36,
        sl_atr_multiplier=1.5,
        trailing_atr_multiplier=3.0,
        min_vol_mult=0.5,
        min_body_ratio=0.45
    )

    configs = [
        {
            "name": "1. 기준선 (무방향 양방향 횡보 매매)",
            "mr_engine": TrendAlignedMeanReversionEngine(mode="BASELINE"),
        },
        {
            "name": "2. 눌림목 전환 (EMA200 위=롱만, 아래=숏만)",
            "mr_engine": TrendAlignedMeanReversionEngine(mode="PULLBACK_ONLY"),
        },
        {
            "name": "3. 롱 온리 (EMA200 위에서 롱 눌림목만)",
            "mr_engine": TrendAlignedMeanReversionEngine(mode="LONG_ONLY_ABOVE_EMA"),
        },
        {
            "name": "4. 평균회귀 엔진 완전 제거 (추세 단독)",
            "mr_engine": None,  # 평균회귀 비활성화 효과 비교
        },
    ]

    summary_rows = []
    equity_curves = {}

    for cfg in configs:
        mr_eng = cfg['mr_engine']
        # 4번 설정의 경우 더미 엔진(시그널 미발생)으로 처리
        if mr_eng is None:
            class DummyMREngine(MeanReversionEngine):
                def check_entry_signal_fast(self, i, records):
                    return None
            mr_eng = DummyMREngine()

        sim = BacktestSimulator(
            initial_capital=10000.0,
            risk_per_trade_pct=0.02, # 2.0% Risk
            leverage=3.0,
            trend_engine=standard_tf_engine,
            mean_revert_engine=mr_eng
        )
        res = sim.run(test_df)
        df_t = res['trades_df']

        # 평균회귀 및 추세추종 거래 분해
        if not df_t.empty:
            mr_trades = df_t[df_t['engine'] == "MEAN_REVERSION"]
            mr_pnl = mr_trades['pnl'].sum() if not mr_trades.empty else 0.0
            mr_wr = (len(mr_trades[mr_trades['pnl'] > 0]) / len(mr_trades)) * 100.0 if not mr_trades.empty else 0.0
            mr_cnt = len(mr_trades)

            tf_trades = df_t[df_t['engine'] == "TREND_FOLLOWING"]
            tf_pnl = tf_trades['pnl'].sum() if not tf_trades.empty else 0.0

            df_t['year'] = pd.to_datetime(df_t['exit_time']).dt.year
            pnl_2022 = df_t[df_t['year'] == 2022]['pnl'].sum()
            pnl_2023 = df_t[df_t['year'] == 2023]['pnl'].sum()
        else:
            mr_pnl, mr_wr, mr_cnt, tf_pnl, pnl_2022, pnl_2023 = 0.0, 0.0, 0, 0.0, 0.0, 0.0

        summary_rows.append({
            "설정": cfg['name'],
            "3.5년 총수익률": f"{res['total_return_pct']:+.2f}%",
            "MDD": f"{res['mdd_pct']:.2f}%",
            "PF": f"{res['profit_factor']:.2f}",
            "총 거래": f"{res['total_trades']}회",
            "횡보 거래": f"{mr_cnt}회",
            "횡보 승률": f"{mr_wr:.1f}%",
            "횡보 PnL ($)": f"${mr_pnl:+,.2f}",
            "추세 PnL ($)": f"${tf_pnl:+,.2f}",
            "2022년 ($)": f"${pnl_2022:+,.2f}",
            "2023년 ($)": f"${pnl_2023:+,.2f}",
        })
        equity_curves[cfg['name']] = res['equity_curve']

    df_sum = pd.DataFrame(summary_rows)
    print("\n" + "=" * 120)
    print("               [ 실험 13: 평균회귀 엔진 눌림목(Pullback) 전환 성과 비교표 (3.5년 장기) ]               ")
    print("=" * 120)
    print(df_sum.to_string(index=False))
    print("=" * 120)

    # 차트 저장
    plt.figure(figsize=(14, 7))
    for label, eq in equity_curves.items():
        plt.plot(eq, linewidth=1.6, label=label)

    plt.axhline(10000.0, color='gray', linestyle='--', label='Initial ($10k)')
    plt.title("RADE Experiment 13: Mean Reversion Pullback Realignment (3.5 Years)", fontsize=13, fontweight='bold')
    plt.xlabel("Hours")
    plt.ylabel("Account Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    chart_path = os.path.join("data", "exp13_mean_revert_pullback_plot.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Done] 눌림목 전환 비교 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_experiment_13()
