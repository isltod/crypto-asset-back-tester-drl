"""
[실험 2] 피라미딩 (추세 불타기) 독립 검증 스크립트
- 기준점: Baseline (피라미딩 없음, 1.0% Risk)
- 비교군: 
  1) 2.0 * ATR 수익 시 0.5배 추가 진입
  2) 2.5 * ATR 수익 시 0.5배 추가 진입
  3) 3.0 * ATR 수익 시 0.5배 추가 진입
- 목적: 피라미딩이 대형 랠리 수익을 증폭시키는지 vs 되밀림 휩쏘에 취약한지 독립 검증
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
from python.engines.trend_following import TrendFollowingEngine


class PyramidingSimulator:
    """피라미딩 지원 선물 백테스터"""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_per_trade_pct: float = 0.01,
        maker_fee_pct: float = 0.0002,
        taker_fee_pct: float = 0.0005,
        slippage_pct: float = 0.0002,
        funding_fee_pct: float = 0.0001,
        pyramiding_enabled: bool = False,
        pyramid_trigger_atr_mult: float = 2.5, # 2.5 * ATR 수익 시 불타기
        pyramid_size_ratio: float = 0.5,       # 기존 사이즈의 50% 추가
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.slippage_pct = slippage_pct
        self.funding_fee_pct = funding_fee_pct
        self.pyramiding_enabled = pyramiding_enabled
        self.pyramid_trigger_atr_mult = pyramid_trigger_atr_mult
        self.pyramid_size_ratio = pyramid_size_ratio

        self.pos_manager = PositionManager(risk_per_trade_pct=risk_per_trade_pct)
        self.mean_revert_engine = MeanReversionEngine()
        self.trend_engine = TrendFollowingEngine()

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        records = df.to_dict('records')
        n = len(records)
        equity = self.initial_capital
        current_pos: Optional[Position] = None
        has_pyramided = False

        trades_history: List[Dict[str, Any]] = []
        equity_curve: List[float] = [equity]
        prev_regime = None

        for i in range(n - 1):
            curr_row = records[i]
            next_row = records[i + 1]

            date_str = str(curr_row.get('datetime', i))[:10]
            self.pos_manager.update_day(date_str, equity)
            curr_regime = curr_row.get('regime_state', RegimeState.RANGE)

            # 1. 펀딩비
            if current_pos and (i % 8 == 0):
                notional = current_pos.size * curr_row['close']
                equity -= (notional * self.funding_fee_pct)

            # 2. 국면 전환 시 손실 포지션 정리
            if prev_regime and curr_regime != prev_regime and current_pos:
                is_losing = False
                if current_pos.side == PositionSide.LONG and curr_row['close'] < current_pos.entry_price:
                    is_losing = True
                elif current_pos.side == PositionSide.SHORT and curr_row['close'] > current_pos.entry_price:
                    is_losing = True

                if is_losing:
                    exit_p = curr_row['close'] * (1.0 - self.slippage_pct if current_pos.side == PositionSide.LONG else 1.0 + self.slippage_pct)
                    pnl = (exit_p - current_pos.entry_price) * current_pos.size if current_pos.side == PositionSide.LONG else (current_pos.entry_price - exit_p) * current_pos.size
                    fee = (current_pos.entry_price * current_pos.size + exit_p * current_pos.size) * self.taker_fee_pct
                    net_pnl = pnl - fee
                    equity += net_pnl
                    trades_history.append({"pnl": net_pnl, "engine": current_pos.engine_name})
                    current_pos = None
                    has_pyramided = False

            prev_regime = curr_regime

            # 3. 포지션 업데이트
            if current_pos:
                # 피라미딩(추세 불타기) 체크
                if self.pyramiding_enabled and not has_pyramided and current_pos.engine_name == "TREND_FOLLOWING":
                    atr = curr_row['atr']
                    profit_dist = (curr_row['close'] - current_pos.entry_price) if current_pos.side == PositionSide.LONG else (current_pos.entry_price - curr_row['close'])
                    
                    if profit_dist >= (atr * self.pyramid_trigger_atr_mult):
                        # 불타기 진입 (기존의 50% 수량 추가)
                        add_size = current_pos.size * self.pyramid_size_ratio
                        add_price = curr_row['close']
                        
                        # 통합 평단가 갱신
                        total_cost = (current_pos.entry_price * current_pos.size) + (add_price * add_size)
                        new_size = current_pos.size + add_size
                        current_pos.entry_price = total_cost / new_size
                        current_pos.size = new_size
                        
                        # 수수료 차감
                        fee = add_price * add_size * self.maker_fee_pct
                        equity -= fee
                        has_pyramided = True

                # 청산 판정
                if current_pos.engine_name == "MEAN_REVERSION":
                    update_res = self.mean_revert_engine.update_position_fast(current_pos, curr_row, current_bar_idx=i)
                else:
                    update_res = self.trend_engine.update_position_fast(current_pos, curr_row)

                action = update_res['action']
                if action != "NONE":
                    exit_p = update_res['exit_price']
                    ratio = update_res['closed_ratio']
                    is_maker = update_res.get('is_maker', False)
                    closed_size = current_pos.size * ratio

                    if is_maker:
                        eff_exit_p = exit_p
                        fee_rate = self.maker_fee_pct
                    else:
                        eff_exit_p = exit_p * (1.0 - self.slippage_pct if current_pos.side == PositionSide.LONG else 1.0 + self.slippage_pct)
                        fee_rate = self.taker_fee_pct

                    pnl = (eff_exit_p - current_pos.entry_price) * closed_size if current_pos.side == PositionSide.LONG else (current_pos.entry_price - eff_exit_p) * closed_size
                    fee = (current_pos.entry_price * closed_size * self.maker_fee_pct) + (eff_exit_p * closed_size * fee_rate)
                    net_pnl = pnl - fee
                    equity += net_pnl
                    trades_history.append({"pnl": net_pnl, "engine": current_pos.engine_name})

                    if ratio >= 1.0 or current_pos.size <= (closed_size + 1e-6):
                        current_pos = None
                        has_pyramided = False
                    else:
                        current_pos.size -= closed_size

            # 4. 신규 진입
            if current_pos is None:
                signal = None
                weight = 1.0

                if curr_regime == RegimeState.RANGE:
                    signal = self.mean_revert_engine.check_entry_signal_fast(i, records)
                    weight = curr_row.get('mean_revert_weight', 1.0)
                elif curr_regime == RegimeState.TREND:
                    signal = self.trend_engine.check_entry_signal_fast(i, records)
                    weight = curr_row.get('trend_follow_weight', 1.0)

                if signal:
                    eff_entry_p = next_row['open']
                    side = signal['side']
                    pos_size = self.pos_manager.calculate_position_size(
                        equity=equity,
                        entry_price=eff_entry_p,
                        sl_price=signal['sl_price'],
                        side=side,
                        weight=weight,
                    )
                    if pos_size > 0.0001:
                        current_pos = Position(
                            side=side,
                            entry_price=eff_entry_p,
                            size=pos_size,
                            sl_price=signal['sl_price'],
                            tp1_price=signal['tp1_price'],
                            tp2_price=signal['tp2_price'],
                            engine_name=signal['engine'],
                            entry_bar=i + 1,
                        )
                        has_pyramided = False

            equity_curve.append(equity)

        eq_arr = np.array(equity_curve)
        total_ret = ((eq_arr[-1] - self.initial_capital) / self.initial_capital) * 100.0
        peak = np.maximum.accumulate(eq_arr)
        drawdowns = (eq_arr - peak) / peak
        mdd = abs(drawdowns.min()) * 100.0

        df_t = pd.DataFrame(trades_history)
        wins = df_t[df_t['pnl'] > 0] if not df_t.empty else pd.DataFrame()
        losses = df_t[df_t['pnl'] < 0] if not df_t.empty else pd.DataFrame()
        pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else 0.0

        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = (returns.mean() / (returns.std() + 1e-10)) * np.sqrt(8760) if returns.std() > 0 else 0.0

        return {
            "total_return_pct": total_ret,
            "final_equity": eq_arr[-1],
            "mdd_pct": mdd,
            "profit_factor": pf,
            "sharpe_ratio": sharpe,
            "total_trades": len(df_t),
            "equity_curve": equity_curve,
        }


def run_experiment_2():
    print("=== [실험 2] 피라미딩(추세 불타기) 독립 검증 시작 ===")

    fetcher = BinanceFuturesFetcher(data_dir="data")
    df_raw = fetcher.get_or_download_data(symbol="BTCUSDT", interval="1h", start_time_str="2023-01-01 00:00:00", end_time_str="2024-06-01 00:00:00")
    df_indicators = add_all_indicators(df_raw)
    regime_manager = RegimeManager(hmm_window=720, retrain_interval=168, hysteresis_upper=0.65, hysteresis_lower=0.35, cooldown_bars=3)
    df_processed = regime_manager.calculate_regime_probabilities(df_indicators)
    test_df = df_processed.dropna(subset=['regime_trend_prob']).reset_index(drop=True)

    configs = [
        {"name": "Baseline (불타기 없음)", "pyramid": False, "mult": 0.0},
        {"name": "Pyramid 2.0x ATR (+50% 추가)", "pyramid": True, "mult": 2.0},
        {"name": "Pyramid 2.5x ATR (+50% 추가)", "pyramid": True, "mult": 2.5},
        {"name": "Pyramid 3.0x ATR (+50% 추가)", "pyramid": True, "mult": 3.0},
    ]

    results = []
    equity_curves = {}

    for cfg in configs:
        sim = PyramidingSimulator(
            initial_capital=10000.0,
            risk_per_trade_pct=0.01,
            pyramiding_enabled=cfg['pyramid'],
            pyramid_trigger_atr_mult=cfg['mult'],
            pyramid_size_ratio=0.5,
        )
        res = sim.run(test_df)
        results.append({
            "설정": cfg['name'],
            "총 수익률": f"{res['total_return_pct']:+.2f}%",
            "최종 자산": f"${res['final_equity']:,.2f}",
            "MDD": f"{res['mdd_pct']:.2f}%",
            "Profit Factor": f"{res['profit_factor']:.2f}",
            "Sharpe Ratio": f"{res['sharpe_ratio']:.2f}",
        })
        equity_curves[cfg['name']] = res['equity_curve']

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("                [ 실험 2: 피라미딩(추세 불타기) 성과 비교표 ]                ")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print("=" * 80)

    # 차트 저장
    plt.figure(figsize=(12, 6))
    for name, eq in equity_curves.items():
        plt.plot(eq, linewidth=1.8, label=name)
    plt.axhline(10000.0, color='gray', linestyle='--', label='Initial ($10k)')
    plt.title("RADE Experiment 2: Pyramiding Comparison (Baseline vs 2.0x vs 2.5x vs 3.0x ATR)")
    plt.xlabel("Bars (Hourly)")
    plt.ylabel("Account Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join("data", "exp2_pyramiding_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n[Done] 실험 2 비교 차트 저장 완료: {plot_path}")


if __name__ == "__main__":
    run_experiment_2()
