"""
[실험 9] 숏(SHORT) 비대칭 트레일링 스탑 독립 검증 스크립트
- 문제점: 하락장은 급락 후 급반등(숏스퀴즈)이 잦아 대칭 3.0*ATR 트레일링 시 수익 반납
- 해결책: 롱(3.0*ATR 유지) vs 숏(1.5*ATR, 1.8*ATR, 2.0*ATR로 타이트화) 비대칭 검증
- 평가: 3.5년 전체 수익률 및 [추세추종 - 숏], 2022년 하락장 성과 변화 정밀 비교
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from python.data.binance_fetcher import BinanceFuturesFetcher
from python.utils.indicators import add_all_indicators
from python.regime.regime_manager import RegimeManager, RegimeState
from python.risk.position_manager import Position, PositionSide, PositionManager
from python.engines.mean_reversion import MeanReversionEngine


class AsymmetricTrendEngine:
    """롱/숏 비대칭 트레일링 스탑 추세추종 엔진"""

    def __init__(
        self,
        adx_threshold: float = 25.0,
        breakout_lookback: int = 36,
        sl_atr_multiplier: float = 1.5,
        long_trailing_atr: float = 3.0,   # 롱은 3.0 * ATR
        short_trailing_atr: float = 1.8,  # 숏은 1.8 * ATR
        min_vol_mult: float = 0.5,
        min_body_ratio: float = 0.45,
    ):
        self.name = "TREND_FOLLOWING"
        self.adx_threshold = adx_threshold
        self.breakout_lookback = breakout_lookback
        self.sl_atr_multiplier = sl_atr_multiplier
        self.long_trailing_atr = long_trailing_atr
        self.short_trailing_atr = short_trailing_atr
        self.min_vol_mult = min_vol_mult
        self.min_body_ratio = min_body_ratio

    def check_entry_signal_fast(self, i: int, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if i < max(self.breakout_lookback, 200):
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

        candle_range = high - low
        body_size = abs(close - open_p)
        if candle_range > 0 and (body_size / candle_range) < self.min_body_ratio:
            return None

        prev_slice = records[i - self.breakout_lookback : i]
        box_high = max(r['high'] for r in prev_slice)
        box_low = min(r['low'] for r in prev_slice)

        # 롱 조건
        if close > ema200 and close > box_high and adx >= self.adx_threshold and plus_di > minus_di and vol_change >= self.min_vol_mult:
            sl_price = close - (atr * self.sl_atr_multiplier)
            return {
                "side": PositionSide.LONG,
                "sl_price": sl_price,
                "tp1_price": None,
                "tp2_price": None,
                "engine": self.name,
            }

        # 숏 조건
        if close < ema200 and close < box_low and adx >= self.adx_threshold and minus_di > plus_di and vol_change >= self.min_vol_mult:
            sl_price = close + (atr * self.sl_atr_multiplier)
            return {
                "side": PositionSide.SHORT,
                "sl_price": sl_price,
                "tp1_price": None,
                "tp2_price": None,
                "engine": self.name,
            }

        return None

    def update_position_fast(self, pos: Position, curr: Dict[str, Any]) -> Dict[str, Any]:
        high = curr['high']
        low = curr['low']
        atr = curr['atr']

        if pos.side == PositionSide.LONG:
            if high > pos.highest_price:
                pos.highest_price = high

            # 롱 트레일링 스탑
            trailing_sl = pos.highest_price - (atr * self.long_trailing_atr)
            pos.sl_price = max(pos.sl_price, trailing_sl)

            if low <= pos.sl_price:
                return {"action": "TRAILING_STOP", "exit_price": pos.sl_price, "closed_ratio": 1.0, "is_maker": False}

        elif pos.side == PositionSide.SHORT:
            if low < pos.lowest_price:
                pos.lowest_price = low

            # 숏 비대칭 타이트 트레일링 스탑
            trailing_sl = pos.lowest_price + (atr * self.short_trailing_atr)
            pos.sl_price = min(pos.sl_price, trailing_sl)

            if high >= pos.sl_price:
                return {"action": "TRAILING_STOP", "exit_price": pos.sl_price, "closed_ratio": 1.0, "is_maker": False}

        return {"action": "NONE", "exit_price": 0.0, "closed_ratio": 0.0, "is_maker": False}


def run_asymmetric_simulation(test_df: pd.DataFrame, short_trail_atr: float) -> Dict[str, Any]:
    records = test_df.to_dict('records')
    n = len(records)
    initial_cap = 10000.0
    equity = initial_cap
    current_pos: Optional[Position] = None

    pos_mgr = PositionManager(risk_per_trade_pct=0.02, default_leverage=3.0)
    mr_engine = MeanReversionEngine()
    tf_engine = AsymmetricTrendEngine(long_trailing_atr=3.0, short_trailing_atr=short_trail_atr)

    trades_history = []
    equity_curve = [equity]
    prev_regime = None

    for i in range(n - 1):
        curr = records[i]
        nxt = records[i + 1]

        date_str = str(curr.get('datetime', i))[:10]
        pos_mgr.update_day(date_str, equity)
        curr_regime = curr.get('regime_state', RegimeState.RANGE)

        # 펀딩비
        if current_pos and (i % 8 == 0):
            equity -= (current_pos.size * curr['close'] * 0.0001)

        # 국면 전환 시 손실 컷
        if prev_regime and curr_regime != prev_regime and current_pos:
            is_losing = False
            if current_pos.side == PositionSide.LONG and curr['close'] < current_pos.entry_price:
                is_losing = True
            elif current_pos.side == PositionSide.SHORT and curr['close'] > current_pos.entry_price:
                is_losing = True

            if is_losing:
                exit_p = curr['close'] * (0.9998 if current_pos.side == PositionSide.LONG else 1.0002)
                pnl = (exit_p - current_pos.entry_price) * current_pos.size if current_pos.side == PositionSide.LONG else (current_pos.entry_price - exit_p) * current_pos.size
                fee = (current_pos.entry_price + exit_p) * current_pos.size * 0.0005
                net_pnl = pnl - fee
                equity += net_pnl
                trades_history.append({"pnl": net_pnl, "engine": current_pos.engine_name, "side": current_pos.side.value, "exit_time": curr.get('datetime', i)})
                current_pos = None

        prev_regime = curr_regime

        # 보유 포지션 업데이트
        if current_pos:
            if current_pos.engine_name == "MEAN_REVERSION":
                res = mr_engine.update_position_fast(current_pos, curr, current_bar_idx=i)
            else:
                res = tf_engine.update_position_fast(current_pos, curr)

            if res['action'] != "NONE":
                exit_p = res['exit_price']
                ratio = res['closed_ratio']
                is_maker = res.get('is_maker', False)
                closed_sz = current_pos.size * ratio

                eff_exit_p = exit_p if is_maker else (exit_p * (0.9998 if current_pos.side == PositionSide.LONG else 1.0002))
                fee_rate = 0.0002 if is_maker else 0.0005

                pnl = (eff_exit_p - current_pos.entry_price) * closed_sz if current_pos.side == PositionSide.LONG else (current_pos.entry_price - eff_exit_p) * closed_sz
                fee = (current_pos.entry_price * closed_sz * 0.0002) + (eff_exit_p * closed_sz * fee_rate)
                net_pnl = pnl - fee
                equity += net_pnl
                trades_history.append({"pnl": net_pnl, "engine": current_pos.engine_name, "side": current_pos.side.value, "exit_time": curr.get('datetime', i)})

                if ratio >= 1.0 or current_pos.size <= (closed_sz + 1e-6):
                    current_pos = None
                else:
                    current_pos.size -= closed_sz

        # 신규 진입
        if current_pos is None:
            sig = None
            w = 1.0
            if curr_regime == RegimeState.RANGE:
                sig = mr_engine.check_entry_signal_fast(i, records)
                w = curr.get('mean_revert_weight', 1.0)
            elif curr_regime == RegimeState.TREND:
                sig = tf_engine.check_entry_signal_fast(i, records)
                w = curr.get('trend_follow_weight', 1.0)

            if sig:
                eff_entry = nxt['open']
                side = sig['side']
                sz = pos_mgr.calculate_position_size(equity=equity, entry_price=eff_entry, sl_price=sig['sl_price'], side=side, weight=w)
                if sz > 0.0001:
                    current_pos = Position(
                        side=side,
                        entry_price=eff_entry,
                        size=sz,
                        sl_price=sig['sl_price'],
                        tp1_price=sig['tp1_price'],
                        tp2_price=sig['tp2_price'],
                        engine_name=sig['engine'],
                        entry_bar=i + 1,
                    )

        equity_curve.append(equity)

    eq_arr = np.array(equity_curve)
    tot_ret = ((eq_arr[-1] - initial_cap) / initial_cap) * 100.0
    peak = np.maximum.accumulate(eq_arr)
    drawdowns = ((eq_arr - peak) / peak) * 100.0
    mdd = abs(drawdowns.min()) * 100.0

    df_t = pd.DataFrame(trades_history)
    wins = df_t[df_t['pnl'] > 0] if not df_t.empty else pd.DataFrame()
    losses = df_t[df_t['pnl'] < 0] if not df_t.empty else pd.DataFrame()
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else 0.0

    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = (returns.mean() / (returns.std() + 1e-10)) * np.sqrt(8760) if returns.std() > 0 else 0.0

    return {
        "total_return_pct": tot_ret,
        "final_equity": eq_arr[-1],
        "mdd_pct": mdd,
        "profit_factor": pf,
        "sharpe_ratio": sharpe,
        "trades_df": df_t,
        "equity_curve": equity_curve,
    }


def run_experiment_9():
    print("=== [실험 9] 숏(SHORT) 비대칭 트레일링 스탑 독립 검증 시작 ===")

    cache_file = os.path.join("data", "BTCUSDT_1h_2021_2024.csv")
    df_raw = pd.read_csv(cache_file)
    df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)

    df_ind = add_all_indicators(df_raw)
    manager = RegimeManager(hmm_window=720, retrain_interval=168, hysteresis_upper=0.65, hysteresis_lower=0.35, cooldown_bars=3)
    df_proc = manager.calculate_regime_probabilities(df_ind)
    test_df = df_proc.dropna(subset=['regime_trend_prob']).reset_index(drop=True)

    configs = [
        {"name": "대칭 3.0x (Baseline)", "short_trail": 3.0},
        {"name": "숏 2.0x ATR", "short_trail": 2.0},
        {"name": "숏 1.8x ATR", "short_trail": 1.8},
        {"name": "숏 1.5x ATR", "short_trail": 1.5},
    ]

    summary_rows = []
    equity_curves = {}

    for cfg in configs:
        res = run_asymmetric_simulation(test_df, cfg['short_trail'])
        df_t = res['trades_df']
        
        # [추세추종 - 숏]만 발췌
        tf_short = df_t[(df_t['engine'] == "TREND_FOLLOWING") & (df_t['side'] == "SHORT")]
        tf_short_pnl = tf_short['pnl'].sum() if not tf_short.empty else 0.0
        tf_short_wr = (len(tf_short[tf_short['pnl'] > 0]) / len(tf_short)) * 100.0 if not tf_short.empty else 0.0

        # 2022년 하락장 손익
        df_t['year'] = pd.to_datetime(df_t['exit_time']).dt.year
        pnl_2022 = df_t[df_t['year'] == 2022]['pnl'].sum() if not df_t.empty else 0.0

        summary_rows.append({
            "설정": cfg['name'],
            "3.5년 총수익률": f"{res['total_return_pct']:+.2f}%",
            "최종 자산 ($)": f"${res['final_equity']:,.2f}",
            "MDD (%)": f"{res['mdd_pct']:.2f}%",
            "전체 PF": f"{res['profit_factor']:.2f}",
            "추세 숏 PnL ($)": f"${tf_short_pnl:+,.2f}",
            "추세 숏 승률": f"{tf_short_wr:.1f}%",
            "2022년 PnL ($)": f"${pnl_2022:+,.2f}",
        })
        equity_curves[cfg['name']] = res['equity_curve']

    df_sum = pd.DataFrame(summary_rows)
    print("\n" + "=" * 90)
    print("           [ 실험 9: 숏(SHORT) 비대칭 트레일링 스탑 성과 비교표 (3.5년 장기) ]           ")
    print("=" * 90)
    print(df_sum.to_string(index=False))
    print("=" * 90)

    # 차트 저장
    plt.figure(figsize=(12, 6))
    for label, eq in equity_curves.items():
        plt.plot(eq, linewidth=1.8, label=label)

    plt.axhline(10000.0, color='gray', linestyle='--', label='Initial ($10k)')
    plt.title("RADE Experiment 9: Asymmetric Short Trailing Stop Comparison (3.5 Years)", fontsize=12)
    plt.xlabel("Trade Progress (Trades)")
    plt.ylabel("Account Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    chart_path = os.path.join("data", "exp9_asymmetric_short_plot.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Done] 숏 비대칭 비교 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_experiment_9()
