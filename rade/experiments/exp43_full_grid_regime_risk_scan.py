"""
[실험 43] 국면별 차등 리스크 전구간 (0.5% ~ 4.0%) 64개 조합 2D 매트릭스 전수 스캔
- 추세장(BULL_TREND) 리스크: [0.5%, 1.0%, 1.5%, 2.0%, 2.5%, 3.0%, 3.5%, 4.0%]
- 횡보장(RANGE) 리스크:      [0.5%, 1.0%, 1.5%, 2.0%, 2.5%, 3.0%, 3.5%, 4.0%]
- 총 8 x 8 = 64개 조합 전수 탐색 (TH=0.74, CASH 모드)
- 목적: MDD 10% 이하 초안전 구간 및 칼마 비율 극대화 최적 파레토 프론티어 도출
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager
from rade.backtest.simulator import BacktestSimulator, Position
from rade.engines.trend_following import TrendFollowingEngine
from rade.engines.mean_reversion import MeanReversionEngine
from rade.risk.position_manager import PositionSide


class DynamicRegimeSimulator(BacktestSimulator):
    """국면(BULL vs RANGE)에 따라 risk_per_trade_pct를 동적으로 차등 적용하는 시뮬레이터"""
    def __init__(self, trend_risk_pct: float, mr_risk_pct: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trend_risk_pct = trend_risk_pct
        self.mr_risk_pct = mr_risk_pct

    def run(self, df_processed: pd.DataFrame):
        records = df_processed.to_dict('records')
        n_bars = len(records)
        if n_bars < 2:
            return self._empty_result()

        equity = self.initial_capital
        current_pos = None
        trades_history = []
        equity_curve = []
        timestamps = []
        prev_regime = None

        for i in range(n_bars - 1):
            curr_row = records[i]
            next_row = records[i + 1]
            curr_regime = curr_row.get('regime_state', 'RANGE')

            # 2. 국면 전환 시 손실 중인 포지션 즉시 컷
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

                if curr_regime == "RANGE":
                    self.pos_manager.risk_per_trade_pct = self.mr_risk_pct
                    signal = self.mean_revert_engine.check_entry_signal_fast(i, records)
                elif curr_regime == "BULL_TREND":
                    self.pos_manager.risk_per_trade_pct = self.trend_risk_pct
                    raw_sig = self.trend_engine.check_entry_signal_fast(i, records)
                    if raw_sig and raw_sig['side'] == PositionSide.LONG:
                        signal = raw_sig

                if signal:
                    raw_entry_price = next_row['open']
                    side = signal['side']
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


def run_experiment_43():
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_ind = add_all_indicators(df_1h)

    reg_mgr = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.74, cooldown_bars=0)
    df_proc = reg_mgr.calculate_regime_probabilities(df_ind)
    test_df = df_proc.iloc[720:].reset_index(drop=True)

    print("=" * 105)
    print("      [실험 43] 국면별 차등 리스크 전구간 (0.5% ~ 4.0%) 64개 조합 2D 전수 스캔 (TH=0.74)")
    print("=" * 105)

    risk_levels = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
    grid_results = []

    for t_risk in risk_levels:
        for m_risk in risk_levels:
            sim = DynamicRegimeSimulator(
                trend_risk_pct=t_risk,
                mr_risk_pct=m_risk,
                initial_capital=10000.0,
                risk_per_trade_pct=0.02,
                leverage=3.0,
                bear_mode="CASH",
                use_regime_transition_cut=False,
                trend_engine=TrendFollowingEngine(trailing_atr_multiplier=4.5, max_trailing_atr=4.5),
                mean_revert_engine=MeanReversionEngine(max_holding_bars=24)
            )
            res = sim.run(test_df)
            t = res["trades_df"].copy()
            t["entry_dt"] = pd.to_datetime(t["entry_time"])
            t["year"] = t["entry_dt"].dt.year

            calmar = res["total_return_pct"] / res["mdd_pct"] if res["mdd_pct"] > 0 else 0
            y2022_pnl = t[t["year"] == 2022]["pnl"].sum() if len(t[t["year"] == 2022]) > 0 else 0.0

            grid_results.append({
                "trend_risk": round(t_risk * 100, 1),
                "mr_risk": round(m_risk * 100, 1),
                "total_return_pct": res["total_return_pct"],
                "profit_dollars": res["final_equity"] - 10000.0,
                "mdd_pct": res["mdd_pct"],
                "calmar_ratio": calmar,
                "profit_factor": res["profit_factor"],
                "win_rate_pct": res["win_rate_pct"],
                "y2022_pnl": y2022_pnl,
            })

    df_res = pd.DataFrame(grid_results)

    # 1. 2D 매트릭스: 4개년 총 수익률 (%)
    pnl_matrix = df_res.pivot(index="trend_risk", columns="mr_risk", values="total_return_pct")
    print("\n[1. 4개년 총 수익률 매트릭스 (%) - 행: 추세장(TF) / 열: 횡보장(MR)]")
    print(pnl_matrix.round(1).to_string())

    # 2. 2D 매트릭스: 최대 낙폭 MDD (%)
    mdd_matrix = df_res.pivot(index="trend_risk", columns="mr_risk", values="mdd_pct")
    print("\n[2. 최대 낙폭 MDD 매트릭스 (%) - 행: 추세장(TF) / 열: 횡보장(MR)]")
    print(mdd_matrix.round(2).to_string())

    # 3. 2D 매트릭스: 칼마 비율 (수익률 / MDD)
    calmar_matrix = df_res.pivot(index="trend_risk", columns="mr_risk", values="calmar_ratio")
    print("\n[3. 칼마 비율 매트릭스 (가성비 = 수익률 / MDD) - 행: 추세장(TF) / 열: 횡보장(MR)]")
    print(calmar_matrix.round(2).to_string())

    # 4. 2022년 하락장 손익 매트릭스 ($)
    y22_matrix = df_res.pivot(index="trend_risk", columns="mr_risk", values="y2022_pnl")
    print("\n[4. 2022년 크립토 윈터 PnL ($) - 행: 추세장(TF) / 열: 횡보장(MR)]")
    print(y22_matrix.round(1).to_string())

    # 5. MDD 10% 이하 초안전 구간 최고 수익 TOP 5
    safe_10 = df_res[df_res["mdd_pct"] <= 10.0].sort_values(by="total_return_pct", ascending=False).head(5)
    print("\n[5. MDD 10% 이하 초안전 최고 수익 TOP 5]")
    for idx, r in safe_10.iterrows():
        print(f" * TF {r['trend_risk']}% x MR {r['mr_risk']}% -> 수익: +${r['profit_dollars']:,.2f} (+{r['total_return_pct']:.1f}%) | MDD: {r['mdd_pct']:.2f}% | Calmar: {r['calmar_ratio']:.2f} | 2022년: {r['y2022_pnl']:+8.1f}$")

    # 6. MDD 15% 이하 최고 수익 TOP 5
    safe_15 = df_res[df_res["mdd_pct"] <= 15.0].sort_values(by="total_return_pct", ascending=False).head(5)
    print("\n[6. MDD 15% 이하 최고 수익 TOP 5]")
    for idx, r in safe_15.iterrows():
        print(f" * TF {r['trend_risk']}% x MR {r['mr_risk']}% -> 수익: +${r['profit_dollars']:,.2f} (+{r['total_return_pct']:.1f}%) | MDD: {r['mdd_pct']:.2f}% | Calmar: {r['calmar_ratio']:.2f} | 2022년: {r['y2022_pnl']:+8.1f}$")


if __name__ == "__main__":
    run_experiment_43()
