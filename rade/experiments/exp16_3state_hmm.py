"""
[실험 16] 3-State Gaussian HMM 기반 국면 분류기 독립 검증
- 배경: HMM 2-State는 횡보와 추세만 구분하여 하락장/패닉 위험을 분리하지 못함
- 3개 상태 정의:
  - State 0: Low-Vol Range (평온 횡보) -> 평균회귀 엔진
  - State 1: Bull Trend (상승 추세) -> 추세추종 롱
  - State 2: High-Vol Bear / Panic (고변동성 하락/패닉) -> 관망(Cash) 또는 숏
- 평가 데이터: 2021.01 ~ 2024.12 (4.0년 풀데이터: In-Sample 3.5년 + Out-of-Sample 7개월)
"""
import os
import sys

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from hmmlearn.hmm import GaussianHMM
from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager, RegimeState
from rade.risk.position_manager import Position, PositionSide, PositionManager
from rade.engines.mean_reversion import MeanReversionEngine
from rade.engines.trend_following import TrendFollowingEngine
from rade.backtest.simulator import BacktestSimulator


class HMM3StateDetector:
    """3개 은닉 상태(Range, Bull, Bear/Panic)를 학습하고 정렬하는 HMM 분류기"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.range_idx = 0
        self.bull_idx = 1
        self.bear_idx = 2

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        features = df[["return", "atr_ratio", "vol_change"]].copy()
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return features.values

    def fit(self, df: pd.DataFrame):
        X = self._prepare_features(df)
        self.model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=100,
            random_state=self.random_state,
        )
        self.model.fit(X)

        # 3개 상태 자동 정렬 (State Alignment)
        # means_[:, 0] = return, means_[:, 1] = atr_ratio
        mean_returns = self.model.means_[:, 0]
        mean_atrs = self.model.means_[:, 1]

        # 1) 평균 수익률이 가장 높은 상태 -> BULL (상승)
        bull_candidate = int(np.argmax(mean_returns))

        # 2) 나머지 2개 중 ATR(변동성)이 더 낮은 상태 -> RANGE (횡보), 더 높은 상태 -> BEAR/PANIC
        remaining = [i for i in range(3) if i != bull_candidate]
        if mean_atrs[remaining[0]] < mean_atrs[remaining[1]]:
            range_candidate = remaining[0]
            bear_candidate = remaining[1]
        else:
            range_candidate = remaining[1]
            bear_candidate = remaining[0]

        self.bull_idx = bull_candidate
        self.range_idx = range_candidate
        self.bear_idx = bear_candidate

    def get_latest_probabilities(self, df_window: pd.DataFrame) -> Tuple[float, float, float]:
        if self.model is None:
            return 0.33, 0.33, 0.33

        X = self._prepare_features(df_window)
        posteriors = self.model.predict_proba(X)
        last_p = posteriors[-1]
        return float(last_p[self.range_idx]), float(last_p[self.bull_idx]), float(last_p[self.bear_idx])


class Regime3StateManager:
    """3-State HMM 롤링 시뮬레이션 및 히스테리시스 전이 관리자"""

    def __init__(self, hmm_window: int = 720, retrain_interval: int = 168, trans_threshold: float = 0.50):
        self.hmm_window = hmm_window
        self.retrain_interval = retrain_interval
        self.trans_threshold = trans_threshold
        self.detector = HMM3StateDetector()
        self.last_trained_idx = -999

    def calculate_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        n = len(data)

        states = ["RANGE"] * n
        p_ranges = np.full(n, np.nan)
        p_bulls = np.full(n, np.nan)
        p_bears = np.full(n, np.nan)

        curr_state = "RANGE"

        for i in range(self.hmm_window, n):
            if (i - self.last_trained_idx) >= self.retrain_interval or self.last_trained_idx < 0:
                train_slice = data.iloc[i - self.hmm_window : i]
                try:
                    self.detector.fit(train_slice)
                    self.last_trained_idx = i
                except Exception:
                    pass

            recent_slice = data.iloc[max(0, i - 100) : i + 1]
            try:
                p_r, p_u, p_d = self.detector.get_latest_probabilities(recent_slice)
            except Exception:
                p_r, p_u, p_d = 0.34, 0.33, 0.33

            p_ranges[i] = p_r
            p_bulls[i] = p_u
            p_bears[i] = p_d

            # 히스테리시스 기반 3-State 상태 전이
            # 가장 지배적인 확률이 임계값(0.50)을 넘으면 전환, 아니면 현재 상태 유지
            probs = {"RANGE": p_r, "BULL": p_u, "BEAR": p_d}
            max_state = max(probs, key=probs.get)

            if probs[max_state] >= self.trans_threshold:
                curr_state = max_state

            states[i] = curr_state

        data['p_range'] = p_ranges
        data['p_bull'] = p_bulls
        data['p_bear'] = p_bears
        data['state_3hmm'] = states
        return data


class HMM3StateSimulator:
    """3-State HMM 전용 백테스터"""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_per_trade_pct: float = 0.02,
        leverage: float = 3.0,
        bear_mode: str = "CASH",  # "CASH" (관망) or "SHORT" (추세 숏)
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.leverage = leverage
        self.bear_mode = bear_mode
        self.maker_fee_pct = 0.0002
        self.taker_fee_pct = 0.0005
        self.slippage_pct = 0.0002
        self.funding_fee_pct = 0.0001

        self.pos_manager = PositionManager(risk_per_trade_pct=risk_per_trade_pct, default_leverage=leverage)
        self.mr_engine = MeanReversionEngine()
        self.tf_engine = TrendFollowingEngine()

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        records = df.to_dict('records')
        n = len(records)
        if n < 2:
            return {}

        equity = self.initial_capital
        current_pos: Optional[Position] = None
        trades_history = []
        equity_curve = [equity]
        timestamps = [records[0].get('datetime', 0)]

        for i in range(n - 1):
            curr_row = records[i]
            next_row = records[i + 1]

            date_str = str(curr_row.get('datetime', i))[:10]
            self.pos_manager.update_day(date_str, equity)

            curr_state = curr_row.get('state_3hmm', "RANGE")

            # 1. 펀딩비
            if current_pos and (i % 8 == 0):
                equity -= (current_pos.size * curr_row['close'] * self.funding_fee_pct)

            # 2. 보유 포지션 업데이트
            if current_pos:
                if current_pos.engine_name == "MEAN_REVERSION":
                    res = self.mr_engine.update_position_fast(current_pos, curr_row, current_bar_idx=i)
                else:
                    res = self.tf_engine.update_position_fast(current_pos, curr_row)

                if res['action'] != "NONE":
                    exit_price = res['exit_price']
                    ratio = res['closed_ratio']
                    is_maker = res.get('is_maker', False)
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
                        "reason": res['action'],
                    })

                    if ratio >= 1.0 or current_pos.size <= (closed_size + 1e-6):
                        current_pos = None
                    else:
                        current_pos.size -= closed_size

            # 3. 신규 진입 시그널 (3-State HMM 규칙 적용)
            if current_pos is None and not self.pos_manager.check_kill_switch(equity):
                signal = None

                # [State 0: RANGE] -> 평균회귀
                if curr_state == "RANGE":
                    signal = self.mr_engine.check_entry_signal_fast(i, records)

                # [State 1: BULL] -> 추세추종 롱
                elif curr_state == "BULL":
                    raw_sig = self.tf_engine.check_entry_signal_fast(i, records)
                    if raw_sig and raw_sig['side'] == PositionSide.LONG:
                        signal = raw_sig

                # [State 2: BEAR / PANIC] -> 설정에 따라 관망(None) 또는 숏
                elif curr_state == "BEAR":
                    if self.bear_mode == "CASH":
                        signal = None  # 패닉/하락장 현금 100% 관망
                    elif self.bear_mode == "SHORT":
                        raw_sig = self.tf_engine.check_entry_signal_fast(i, records)
                        if raw_sig and raw_sig['side'] == PositionSide.SHORT:
                            signal = raw_sig

                if signal:
                    raw_entry = next_row['open']
                    side = signal['side']
                    eff_entry = raw_entry * (1.0 + self.slippage_pct if side == PositionSide.LONG else 1.0 - self.slippage_pct)
                    pos_size = self.pos_manager.calculate_position_size(
                        equity=equity,
                        entry_price=eff_entry,
                        sl_price=signal['sl_price'],
                        side=side,
                        weight=1.0,
                    )
                    if pos_size > 0.0001:
                        current_pos = Position(
                            side=side,
                            entry_price=eff_entry,
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

        eq_arr = np.array(equity_curve)
        tot_ret = ((eq_arr[-1] - self.initial_capital) / self.initial_capital) * 100.0
        peak = np.maximum.accumulate(eq_arr)
        drawdowns = (eq_arr - peak) / (peak + 1e-10)
        mdd = abs(float(drawdowns.min())) * 100.0

        df_t = pd.DataFrame(trades_history)
        wins = df_t[df_t['pnl'] > 0] if not df_t.empty else pd.DataFrame()
        losses = df_t[df_t['pnl'] < 0] if not df_t.empty else pd.DataFrame()
        pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else 0.0
        wr = (len(wins) / len(df_t)) * 100.0 if not df_t.empty else 0.0

        return {
            "total_return_pct": tot_ret,
            "final_equity": eq_arr[-1],
            "mdd_pct": mdd,
            "profit_factor": pf,
            "win_rate_pct": wr,
            "total_trades": len(df_t),
            "trades_df": df_t,
            "equity_curve": equity_curve,
            "timestamps": timestamps,
        }


def run_experiment_16():
    print("=== [실험 16] 3-State Gaussian HMM (4.0년 In-Sample + Out-of-Sample 전체 비교) 시작 ===")

    f_is = os.path.join("data", "BTCUSDT_1h_2021_2024.csv")
    f_oos = os.path.join("data", "BTCUSDT_1h_2024_OOS.csv")

    df_is = pd.read_csv(f_is)
    df_oos = pd.read_csv(f_oos)

    df_all = pd.concat([df_is, df_oos], ignore_index=True)
    df_all.drop_duplicates(subset=['timestamp'], inplace=True)
    df_all.sort_values(by='timestamp', inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    df_all["datetime"] = pd.to_datetime(df_all["timestamp"], unit="ms", utc=True)

    df_ind = add_all_indicators(df_all)

    # 1. 모델 A: 기존 HMM 2-State
    print("\n1. 모델 A: 기존 HMM 2-State 계산 중...")
    manager_2s = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.45, cooldown_bars=3)
    df_2s = manager_2s.calculate_regime_probabilities(df_ind)
    test_df_2s = df_2s.dropna(subset=['regime_trend_prob']).reset_index(drop=True)
    sim_2s = BacktestSimulator(initial_capital=10000.0, risk_per_trade_pct=0.02, leverage=3.0)
    res_2s = sim_2s.run(test_df_2s)

    # 2. 모델 B: 신규 3-State HMM (State 2 = Bear Short)
    print("2. 모델 B: 신규 3-State HMM (하락장 숏 모드) 계산 중...")
    manager_3s = Regime3StateManager(hmm_window=720, retrain_interval=168, trans_threshold=0.45)
    df_3s = manager_3s.calculate_regimes(df_ind)
    test_df_3s = df_3s.iloc[720:].reset_index(drop=True)

    sim_3s_short = HMM3StateSimulator(initial_capital=10000.0, risk_per_trade_pct=0.02, leverage=3.0, bear_mode="SHORT")
    res_3s_short = sim_3s_short.run(test_df_3s)

    # 3. 모델 C: 신규 3-State HMM (State 2 = Cash 관망 모드)
    print("3. 모델 C: 신규 3-State HMM (하락/패닉장 관망 모드) 계산 중...")
    sim_3s_cash = HMM3StateSimulator(initial_capital=10000.0, risk_per_trade_pct=0.02, leverage=3.0, bear_mode="CASH")
    res_3s_cash = sim_3s_cash.run(test_df_3s)

    def split_metrics(res, split_date="2024-06-01 00:00:00+00:00"):
        df_t = res['trades_df']
        if df_t.empty:
            return {}
        is_trades = df_t[pd.to_datetime(df_t['exit_time']) < split_date]
        oos_trades = df_t[pd.to_datetime(df_t['exit_time']) >= split_date]
        is_pnl = is_trades['pnl'].sum() if not is_trades.empty else 0.0
        oos_pnl = oos_trades['pnl'].sum() if not oos_trades.empty else 0.0
        return {"is_pnl": is_pnl, "oos_pnl": oos_pnl, "is_cnt": len(is_trades), "oos_cnt": len(oos_trades)}

    m_2s = split_metrics(res_2s)
    m_3s_s = split_metrics(res_3s_short)
    m_3s_c = split_metrics(res_3s_cash)

    summary_rows = [
        {
            "모델": "1. 기존 HMM 2-State (기준선)",
            "4년 총수익률": f"{res_2s['total_return_pct']:+.2f}%",
            "MDD": f"{res_2s['mdd_pct']:.2f}%",
            "PF": f"{res_2s['profit_factor']:.2f}",
            "총 거래": f"{res_2s['total_trades']}회",
            "In-Sample (3.5년)": f"${m_2s.get('is_pnl', 0):+,.2f}",
            "OOS (최근 7개월)": f"${m_2s.get('oos_pnl', 0):+,.2f}",
        },
        {
            "모델": "2. 신규 HMM 3-State (하락장 숏)",
            "4년 총수익률": f"{res_3s_short['total_return_pct']:+.2f}%",
            "MDD": f"{res_3s_short['mdd_pct']:.2f}%",
            "PF": f"{res_3s_short['profit_factor']:.2f}",
            "총 거래": f"{res_3s_short['total_trades']}회",
            "In-Sample (3.5년)": f"${m_3s_s.get('is_pnl', 0):+,.2f}",
            "OOS (최근 7개월)": f"${m_3s_s.get('oos_pnl', 0):+,.2f}",
        },
        {
            "모델": "3. 신규 HMM 3-State (하락장 관망/Cash)",
            "4년 총수익률": f"{res_3s_cash['total_return_pct']:+.2f}%",
            "MDD": f"{res_3s_cash['mdd_pct']:.2f}%",
            "PF": f"{res_3s_cash['profit_factor']:.2f}",
            "총 거래": f"{res_3s_cash['total_trades']}회",
            "In-Sample (3.5년)": f"${m_3s_c.get('is_pnl', 0):+,.2f}",
            "OOS (최근 7개월)": f"${m_3s_c.get('oos_pnl', 0):+,.2f}",
        },
    ]

    df_sum = pd.DataFrame(summary_rows)
    print("\n" + "=" * 115)
    print("             [ 실험 16: 기존 HMM 2-State vs 신규 3-State HMM 4년 성과 비교표 ]             ")
    print("=" * 115)
    print(df_sum.to_string(index=False))
    print("=" * 115)

    # 차트 저장
    plt.figure(figsize=(14, 7))
    plt.plot(res_2s['timestamps'], res_2s['equity_curve'], color='gray', linestyle='--', label='2-State HMM (Baseline)')
    plt.plot(res_3s_short['timestamps'], res_3s_short['equity_curve'], color='royalblue', linewidth=1.5, label='3-State HMM (Bear Short)')
    plt.plot(res_3s_cash['timestamps'], res_3s_cash['equity_curve'], color='green', linewidth=1.8, label='3-State HMM (Bear Cash Mode)')
    plt.axhline(10000.0, color='black', linestyle=':', alpha=0.6)
    plt.axvline(pd.to_datetime("2024-06-01"), color='blue', linestyle='-.', label='Out-of-Sample Start (2024.06)')
    plt.title("RADE Experiment 16: 2-State vs 3-State HMM Comparison (4.0 Years In-Sample + OOS)", fontsize=13, fontweight='bold')
    plt.xlabel("Timeline")
    plt.ylabel("Account Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    chart_path = os.path.join("data", "exp16_3state_hmm_plot.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Done] 3-State HMM 비교 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_experiment_16()
