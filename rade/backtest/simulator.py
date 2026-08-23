"""
바이낸스 선물 RADE 백테스터 시뮬레이터 [프로덕션 통합 단일 표준 엔진]
- 진입 및 1차/2차 지정가 익절: Maker 수수료 (0.02%)
- 손절, 트레일링 스탑, 타임스탑: Taker 수수료 (0.05%) + 슬리피지 (0.02%)
- 8시간 주기 펀딩비 (0.01%)
- 3-State HMM (RANGE, BULL_TREND, BEAR_PANIC) 완벽 지원
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from rade.risk.position_manager import Position, PositionSide, PositionManager
from rade.engines.mean_reversion import MeanReversionEngine
from rade.engines.trend_following import TrendFollowingEngine
from rade.regime.regime_manager import RegimeState


class BacktestSimulator:
    """RADE 시스템 선물 백테스트 프로덕션 단일 표준 시뮬레이션 엔진"""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        maker_fee_pct: float = 0.0002,      # 0.02% Maker 지정가 수수료 (지정가 익절용)
        taker_fee_pct: float = 0.0005,      # 0.05% Taker 시장가 수수료 (진입, 손절, 트레일링용)
        slippage_pct: float = 0.0002,       # 0.02% 시장가 슬리피지
        funding_fee_pct: float = 0.0001,    # 8시간당 0.01% 펀딩비
        risk_per_trade_pct: float = 0.02,   # 1회 2% 리스크
        leverage: float = 3.0,
        bear_mode: str = "CASH",            # "CASH" (100% 관망) or "SHORT" (추세 숏)
        use_regime_transition_cut: bool = False, # 국면 전환 시 손실 포지션 강제 컷 여부 (기본: False)
        trend_engine: Optional[TrendFollowingEngine] = None,
        mean_revert_engine: Optional[MeanReversionEngine] = None,
    ):
        self.initial_capital = initial_capital
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.slippage_pct = slippage_pct
        self.funding_fee_pct = funding_fee_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.leverage = leverage
        self.bear_mode = bear_mode
        self.use_regime_transition_cut = use_regime_transition_cut

        self.pos_manager = PositionManager(
            risk_per_trade_pct=risk_per_trade_pct,
            default_leverage=leverage
        )
        self.mean_revert_engine = mean_revert_engine or MeanReversionEngine()
        self.trend_engine = trend_engine or TrendFollowingEngine()

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """고속 딕셔너리 리스트 순회 백테스트 실행"""
        records = df.to_dict('records')
        n = len(records)
        if n < 2:
            return self._empty_result()

        equity = self.initial_capital
        current_pos: Optional[Position] = None

        trades_history: List[Dict[str, Any]] = []
        equity_curve: List[float] = [equity]
        timestamps: List[Any] = [records[0].get('datetime', 0)]

        prev_regime = None

        for i in range(n - 1):
            curr_row = records[i]
            next_row = records[i + 1]

            date_str = str(curr_row.get('datetime', i))[:10]
            self.pos_manager.update_day(date_str, equity)

            # 국면 상태 문자열 유연 정규화 (BULL_TREND / BULL -> BULL, RANGE -> RANGE, BEAR_PANIC / BEAR -> BEAR)
            raw_regime = curr_row.get('regime_state', curr_row.get('state_3hmm', curr_row.get('regime', 'RANGE')))
            if "BULL" in str(raw_regime):
                curr_regime = "BULL_TREND"
            elif "BEAR" in str(raw_regime):
                curr_regime = "BEAR_PANIC"
            else:
                curr_regime = "RANGE"

            # 1. 펀딩비 결제 (매 8시간 / 8봉마다)
            if current_pos and (i % 8 == 0):
                notional_val = current_pos.size * curr_row['close']
                funding_cost = notional_val * self.funding_fee_pct
                equity -= funding_cost

            # 2. 국면 전환 시 기존 손실 포지션 시장가 청산 (옵션 활성화 시)
            if self.use_regime_transition_cut and prev_regime and curr_regime != prev_regime and current_pos:
                is_losing = False
                if current_pos.side == PositionSide.LONG and curr_row['close'] < current_pos.entry_price:
                    is_losing = True
                elif current_pos.side == PositionSide.SHORT and curr_row['close'] > current_pos.entry_price:
                    is_losing = True

                if is_losing:
                    exit_price = curr_row['close'] * (1.0 - self.slippage_pct if current_pos.side == PositionSide.LONG else 1.0 + self.slippage_pct)
                    pnl = (exit_price - current_pos.entry_price) * current_pos.size if current_pos.side == PositionSide.LONG else (current_pos.entry_price - exit_price) * current_pos.size
                    fee = (current_pos.entry_price * current_pos.size * self.taker_fee_pct) + (exit_price * current_pos.size * self.taker_fee_pct)
                    net_pnl = pnl - fee
                    equity += net_pnl

                    trades_history.append({
                        "entry_time": current_pos.entry_time,
                        "exit_time": curr_row.get('datetime', i),
                        "engine": current_pos.engine_name,
                        "side": current_pos.side.value,
                        "entry_price": current_pos.entry_price,
                        "exit_price": exit_price,
                        "size": current_pos.size,
                        "pnl": net_pnl,
                        "return_pct": (net_pnl / equity) * 100 if equity > 0 else 0.0,
                        "reason": "REGIME_TRANSITION_CUT",
                    })
                    current_pos = None

            prev_regime = curr_regime

            # 3. 보유 포지션 업데이트 및 익절/손절 체크
            if current_pos:
                if current_pos.engine_name == "MEAN_REVERSION":
                    update_res = self.mean_revert_engine.update_position_fast(current_pos, curr_row, current_bar_idx=i)
                else:
                    update_res = self.trend_engine.update_position_fast(current_pos, curr_row)

                action = update_res['action']

                if action != "NONE":
                    exit_price = update_res['exit_price']
                    ratio = update_res['closed_ratio']
                    is_maker = update_res.get('is_maker', False)
                    closed_size = current_pos.size * ratio

                    # 지정가 익절은 슬리피지 없이 Maker 수수료, 손절/트레일링은 슬리피지 + Taker 수수료
                    if is_maker:
                        eff_exit_price = exit_price
                        exit_fee_rate = self.maker_fee_pct
                    else:
                        eff_exit_price = exit_price * (1.0 - self.slippage_pct if current_pos.side == PositionSide.LONG else 1.0 + self.slippage_pct)
                        exit_fee_rate = self.taker_fee_pct

                    if current_pos.side == PositionSide.LONG:
                        pnl = (eff_exit_price - current_pos.entry_price) * closed_size
                    else:
                        pnl = (current_pos.entry_price - eff_exit_price) * closed_size

                    # 진입은 실전 시장가(Taker) 수수료, 청산은 exit_fee_rate
                    fee = (current_pos.entry_price * closed_size * self.taker_fee_pct) + (eff_exit_price * closed_size * exit_fee_rate)
                    net_pnl = pnl - fee
                    equity += net_pnl

                    trades_history.append({
                        "entry_time": current_pos.entry_time,
                        "exit_time": curr_row.get('datetime', i),
                        "engine": current_pos.engine_name,
                        "side": current_pos.side.value,
                        "entry_price": current_pos.entry_price,
                        "exit_price": eff_exit_price,
                        "size": closed_size,
                        "pnl": net_pnl,
                        "return_pct": (net_pnl / equity) * 100 if equity > 0 else 0.0,
                        "reason": action,
                    })

                    if ratio >= 1.0 or current_pos.size <= (closed_size + 1e-6):
                        current_pos = None
                    else:
                        current_pos.size -= closed_size

            # 4. 신규 진입 시그널 검사
            if current_pos is None and not self.pos_manager.check_kill_switch(equity):
                signal = None

                # [국면 1: 평온 횡보] -> 평균회귀 엔진 가동
                if curr_regime == "RANGE":
                    signal = self.mean_revert_engine.check_entry_signal_fast(i, records)

                # [국면 2: 상승 추세] -> 추세추종 롱 가동
                elif curr_regime == "BULL_TREND":
                    raw_sig = self.trend_engine.check_entry_signal_fast(i, records)
                    if raw_sig and raw_sig['side'] == PositionSide.LONG:
                        signal = raw_sig

                # [국면 3: 위험/패닉 국면] -> 설정에 따라 관망(CASH) 또는 추세 숏
                elif curr_regime == "BEAR_PANIC":
                    if self.bear_mode == "SHORT":
                        raw_sig = self.trend_engine.check_entry_signal_fast(i, records)
                        if raw_sig and raw_sig['side'] == PositionSide.SHORT:
                            signal = raw_sig
                    else:
                        signal = None

                if signal:
                    raw_entry_price = next_row['open']
                    side = signal['side']
                    # 진입은 실전 시장가(Taker) + 슬리피지 모델링 적용
                    if side == PositionSide.LONG:
                        eff_entry_price = raw_entry_price * (1.0 + self.slippage_pct)
                    else:
                        eff_entry_price = raw_entry_price * (1.0 - self.slippage_pct)

                    pos_size = self.pos_manager.calculate_position_size(
                        equity=equity,
                        entry_price=eff_entry_price,
                        sl_price=signal['sl_price'],
                        side=side,
                        weight=1.0,
                    )

                    if pos_size > 0.0001:
                        current_pos = Position(
                            side=side,
                            entry_price=eff_entry_price,
                            size=pos_size,
                            sl_price=signal['sl_price'],
                            tp1_price=signal['tp1_price'],
                            tp2_price=signal['tp2_price'],
                            engine_name=signal['engine'],
                            entry_bar=i + 1,
                            entry_time=str(next_row.get('datetime', i + 1)),
                        )

            equity_curve.append(equity)
            timestamps.append(curr_row.get('datetime', i))

        metrics = self._calculate_metrics(equity_curve, trades_history)
        metrics['equity_curve'] = equity_curve
        metrics['timestamps'] = timestamps
        metrics['trades_df'] = pd.DataFrame(trades_history)
        return metrics

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.initial_capital,
            "total_return_pct": 0.0,
            "mdd_pct": 0.0,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "equity_curve": [self.initial_capital],
            "timestamps": [0],
            "trades_df": pd.DataFrame(),
        }

    def _calculate_metrics(self, equity_curve: List[float], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        eq_arr = np.array(equity_curve)
        total_return_pct = ((eq_arr[-1] - self.initial_capital) / self.initial_capital) * 100.0

        peak = np.maximum.accumulate(eq_arr)
        drawdowns = (eq_arr - peak) / (peak + 1e-10)
        mdd_pct = abs(float(drawdowns.min())) * 100.0

        if not trades:
            return {
                "initial_capital": self.initial_capital,
                "final_equity": eq_arr[-1],
                "total_return_pct": total_return_pct,
                "mdd_pct": mdd_pct,
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
            }

        df_trades = pd.DataFrame(trades)
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] < 0]

        win_rate = (len(wins) / len(df_trades)) * 100.0 if len(df_trades) > 0 else 0.0
        gross_profit = wins['pnl'].sum() if not wins.empty else 0.0
        gross_loss = abs(losses['pnl'].sum()) if not losses.empty else 1e-10
        profit_factor = gross_profit / gross_loss

        returns = pd.Series(equity_curve).pct_change().dropna()
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = (mean_ret / (std_ret + 1e-10)) * np.sqrt(8760) if std_ret > 0 else 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": eq_arr[-1],
            "total_return_pct": total_return_pct,
            "mdd_pct": mdd_pct,
            "total_trades": len(df_trades),
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
        }
