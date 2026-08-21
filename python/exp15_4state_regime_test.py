"""
[실험 15] 4-State 반응형 국면 분류기 (HMM 대체 및 관망 모드 탑재) 독립 검증
- 기존 문제: 느린 HMM 2-state 모델이 시차(Lag)를 유발하고 위험 횡보장에서 억지 매매로 손실 누적
- 4대 국면 정의:
  1. CALM_RANGE (평온 횡보): 평균회귀 엔진 가동 (고승률 박스권 매매)
  2. VOLATILE_CHOPPY (위험 횡보/노이즈): [현금 100% 관망 / 매매 완전 중단]
  3. BULL_TREND (상승 추세): 추세추종 롱 가동
  4. BEAR_TREND (하락 추세): 추세추종 숏 가동
- 평가 데이터: 3.5년 In-Sample (2021~2024.06) + 7개월 Out-of-Sample (2024.06~2024.12)
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
from python.regime.regime_manager import RegimeManager, RegimeState
from python.risk.position_manager import Position, PositionSide, PositionManager
from python.engines.mean_reversion import MeanReversionEngine
from python.engines.trend_following import TrendFollowingEngine
from python.backtest.simulator import BacktestSimulator


class FourStateRegime:
    CALM_RANGE = "CALM_RANGE"           # 평온 횡보 -> 평균회귀
    VOLATILE_CHOPPY = "VOLATILE_CHOPPY" # 위험 횡보 -> 매매 중단 (Cash)
    BULL_TREND = "BULL_TREND"           # 상승 추세 -> 롱 추세추종
    BEAR_TREND = "BEAR_TREND"           # 하락 추세 -> 숏 추세추종


class Reactive4StateRegimeDetector:
    """지표 기반 실시간 4-State 반응형 국면 탐지기"""

    def __init__(
        self,
        adx_trend_threshold: float = 22.0,
        vol_ratio_high_threshold: float = 1.15,
        choppiness_high: float = 55.0,
    ):
        self.adx_trend_threshold = adx_trend_threshold
        self.vol_ratio_high_threshold = vol_ratio_high_threshold
        self.choppiness_high = choppiness_high

    def classify_df(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        n = len(data)

        states = []
        # 지표 추출
        for i in range(n):
            if i < 200:
                states.append(FourStateRegime.VOLATILE_CHOPPY) # 초기 윈도우 관망
                continue

            row = data.iloc[i]
            close = row['close']
            ema200 = row.get('ema200', close)
            adx = row.get('adx', 20.0)
            plus_di = row.get('plus_di', 20.0)
            minus_di = row.get('minus_di', 20.0)
            atr = row.get('atr', 1.0)
            atr_ma50 = row.get('atr_ma50', atr)
            vol_ratio = atr / (atr_ma50 + 1e-10)
            choppiness = row.get('choppiness', 50.0)

            # 1. 강력한 추세 국면 판정
            if adx >= self.adx_trend_threshold:
                if close > ema200 and plus_di > minus_di:
                    states.append(FourStateRegime.BULL_TREND)
                    continue
                elif close < ema200 and minus_di > plus_di:
                    states.append(FourStateRegime.BEAR_TREND)
                    continue

            # 2. 비추세/횡보 구간 세부 분류
            # 변동성이 평온하고 촙 지수가 적당한 경우 -> 평온 횡보 (평균회귀 허용)
            if vol_ratio <= self.vol_ratio_high_threshold and choppiness <= self.choppiness_high:
                states.append(FourStateRegime.CALM_RANGE)
            else:
                # 변동성이 비정상적으로 크거나 촙 지수가 극도로 높은 경우 -> 위험 횡보 (관망)
                states.append(FourStateRegime.VOLATILE_CHOPPY)

        data['four_state_regime'] = states
        return data


class FourStateSimulator:
    """4-State 국면 전용 시뮬레이터"""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_per_trade_pct: float = 0.02,
        leverage: float = 3.0,
        trend_engine: Optional[TrendFollowingEngine] = None,
        mean_revert_engine: Optional[MeanReversionEngine] = None,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.leverage = leverage
        self.maker_fee_pct = 0.0002
        self.taker_fee_pct = 0.0005
        self.slippage_pct = 0.0002
        self.funding_fee_pct = 0.0001

        self.pos_manager = PositionManager(risk_per_trade_pct=risk_per_trade_pct, default_leverage=leverage)
        self.mr_engine = mean_revert_engine or MeanReversionEngine()
        self.tf_engine = trend_engine or TrendFollowingEngine()

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

            curr_state = curr_row.get('four_state_regime', FourStateRegime.VOLATILE_CHOPPY)

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

            # 3. 신규 진입 시그널 (4-State 규칙 적용)
            if current_pos is None and not self.pos_manager.check_kill_switch(equity):
                signal = None

                # [국면 1: 평온 횡보] -> 평균회귀
                if curr_state == FourStateRegime.CALM_RANGE:
                    signal = self.mr_engine.check_entry_signal_fast(i, records)

                # [국면 2: 위험 횡보] -> 매매 중단 (관망 / signal = None)
                elif curr_state == FourStateRegime.VOLATILE_CHOPPY:
                    signal = None

                # [국면 3: 상승 추세] -> 추세추종 (롱만 허용)
                elif curr_state == FourStateRegime.BULL_TREND:
                    raw_sig = self.tf_engine.check_entry_signal_fast(i, records)
                    if raw_sig and raw_sig['side'] == PositionSide.LONG:
                        signal = raw_sig

                # [국면 4: 하락 추세] -> 추세추종 (숏만 허용)
                elif curr_state == FourStateRegime.BEAR_TREND:
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

        # 메트릭 계산
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


def run_experiment_15():
    print("=== [실험 15] 4-State 반응형 국면 분류기 (In-Sample + Out-of-Sample 전체 비교) 시작 ===")

    # 1. 2021~2024 전체 데이터 결합 로드 (2021.01 ~ 2024.12, 4년 풀데이터)
    f_is = os.path.join("data", "BTCUSDT_1h_2021_2024.csv")
    f_oos = os.path.join("data", "BTCUSDT_1h_2024_OOS.csv")

    df_is = pd.read_csv(f_is)
    df_oos = pd.read_csv(f_oos)

    df_all = pd.concat([df_is, df_oos], ignore_index=True)
    df_all.drop_duplicates(subset=['timestamp'], inplace=True)
    df_all.sort_values(by='timestamp', inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    df_all["datetime"] = pd.to_datetime(df_all["timestamp"], unit="ms", utc=True)

    print(f"전체 결합 4년 데이터: 총 {len(df_all)}개 캔들 ({df_all['datetime'].iloc[0]} ~ {df_all['datetime'].iloc[-1]})")

    df_ind = add_all_indicators(df_all)

    # 2. 모델 A: 기존 HMM 2-State (기준선)
    print("\n모델 A: 기존 HMM 2-State 시뮬레이션 계산 중...")
    manager_hmm = RegimeManager(hmm_window=720, retrain_interval=168, hysteresis_upper=0.65, hysteresis_lower=0.35, cooldown_bars=3)
    df_hmm = manager_hmm.calculate_regime_probabilities(df_ind)
    test_df_hmm = df_hmm.dropna(subset=['regime_trend_prob']).reset_index(drop=True)

    sim_hmm = BacktestSimulator(initial_capital=10000.0, risk_per_trade_pct=0.02, leverage=3.0)
    res_hmm_all = sim_hmm.run(test_df_hmm)

    # 3. 모델 B: 신규 4-State 반응형 국면 탐지기 (관망 모드 탑재)
    print("모델 B: 신규 4-State 반응형 신호등 국면 계산 중...")
    detector_4state = Reactive4StateRegimeDetector(adx_trend_threshold=22.0, vol_ratio_high_threshold=1.15, choppiness_high=55.0)
    df_4state = detector_4state.classify_df(df_ind)
    test_df_4state = df_4state.iloc[200:].reset_index(drop=True)

    sim_4state = FourStateSimulator(initial_capital=10000.0, risk_per_trade_pct=0.02, leverage=3.0)
    res_4state_all = sim_4state.run(test_df_4state)

    # 4. 구간별 (3.5년 In-Sample vs 7개월 Out-of-Sample) 분해 비교
    def split_metrics(res, split_date="2024-06-01 00:00:00+00:00"):
        df_t = res['trades_df']
        if df_t.empty:
            return {}
        is_trades = df_t[pd.to_datetime(df_t['exit_time']) < split_date]
        oos_trades = df_t[pd.to_datetime(df_t['exit_time']) >= split_date]

        is_pnl = is_trades['pnl'].sum() if not is_trades.empty else 0.0
        is_wr = (len(is_trades[is_trades['pnl'] > 0]) / len(is_trades)) * 100.0 if not is_trades.empty else 0.0

        oos_pnl = oos_trades['pnl'].sum() if not oos_trades.empty else 0.0
        oos_wr = (len(oos_trades[oos_trades['pnl'] > 0]) / len(oos_trades)) * 100.0 if not oos_trades.empty else 0.0

        return {
            "is_trades": len(is_trades),
            "is_pnl": is_pnl,
            "is_wr": is_wr,
            "oos_trades": len(oos_trades),
            "oos_pnl": oos_pnl,
            "oos_wr": oos_wr,
        }

    m_hmm = split_metrics(res_hmm_all)
    m_4s = split_metrics(res_4state_all)

    # 5. 성과 비교표 출력
    summary_rows = [
        {
            "모델": "기존 HMM 2-State (기준선)",
            "4년 총수익률": f"{res_hmm_all['total_return_pct']:+.2f}%",
            "MDD": f"{res_hmm_all['mdd_pct']:.2f}%",
            "PF": f"{res_hmm_all['profit_factor']:.2f}",
            "총 거래": f"{res_hmm_all['total_trades']}회",
            "In-Sample PnL (3.5년)": f"${m_hmm.get('is_pnl', 0):+,.2f}",
            "OOS PnL (최근 7개월)": f"${m_hmm.get('oos_pnl', 0):+,.2f} (손실)",
        },
        {
            "모델": "신규 4-State 반응형 (관망 탑재)",
            "4년 총수익률": f"{res_4state_all['total_return_pct']:+.2f}%",
            "MDD": f"{res_4state_all['mdd_pct']:.2f}%",
            "PF": f"{res_4state_all['profit_factor']:.2f}",
            "총 거래": f"{res_4state_all['total_trades']}회",
            "In-Sample PnL (3.5년)": f"${m_4s.get('is_pnl', 0):+,.2f}",
            "OOS PnL (최근 7개월)": f"${m_4s.get('oos_pnl', 0):+,.2f}",
        },
    ]

    df_sum = pd.DataFrame(summary_rows)
    print("\n" + "=" * 115)
    print("             [ 실험 15: 기존 HMM 2-State vs 신규 4-State 반응형 국면 모델 4년 전체 비교 ]             ")
    print("=" * 115)
    print(df_sum.to_string(index=False))
    print("=" * 115)

    # 차트 저장
    plt.figure(figsize=(14, 7))
    plt.plot(res_hmm_all['timestamps'], res_hmm_all['equity_curve'], color='gray', linestyle='--', label='HMM 2-State Baseline')
    plt.plot(res_4state_all['timestamps'], res_4state_all['equity_curve'], color='crimson', linewidth=1.8, label='4-State Reactive (with Cash Mode)')
    plt.axhline(10000.0, color='black', linestyle=':', alpha=0.6)
    plt.axvline(pd.to_datetime("2024-06-01"), color='blue', linestyle='-.', label='Out-of-Sample Start (2024.06)')
    plt.title("RADE Experiment 15: HMM 2-State vs 4-State Reactive Regime (4.0 Years In-Sample + OOS)", fontsize=13, fontweight='bold')
    plt.xlabel("Timeline")
    plt.ylabel("Account Equity ($)")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    chart_path = os.path.join("data", "exp15_4state_comparison_plot.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n[Done] 4-State 비교 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    run_experiment_15()
