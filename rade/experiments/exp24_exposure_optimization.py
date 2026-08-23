"""
[실험 24 재실행] 자본 노출도 & 수익률 2단계 정밀 최적화 (cooldown=0 순수 원본 베이스)
- 원본 베이스라인: 4년 +103.39%, MDD 12.46%, 거래 157회, 노출도 3.98%
- 1단계: 개별 독립 감도 분석 (ATR 상한, HMM 임계값, 타임스탑)
- 2단계: 핵심 결합 조합 타깃 그리드 서치
"""
import os
import sys
import time
import warnings
from typing import Dict, Any, List
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.experiments.exp16_3state_hmm import Regime3StateManager
from rade.risk.position_manager import PositionSide, Position


def run_clean_backtest(
    df_proc: pd.DataFrame,
    max_trailing_atr: float = 3.0,
    max_holding_bars: int = 12,
    initial_capital: float = 10000.0,
    risk_per_trade_pct: float = 0.02,
    leverage: float = 3.0,
    maker_fee_pct: float = 0.0002,
    taker_fee_pct: float = 0.0005,
    slippage_pct: float = 0.0002,
    funding_fee_pct: float = 0.0001,
) -> Dict[str, Any]:
    """쿨다운 없이 원본 157회 체결 로직을 완벽하게 재현하는 고속 백테스트 엔진"""
    records = df_proc.iloc[720:].to_dict('records')
    n = len(records)
    total_days = n / 24.0
    total_years = total_days / 365.25

    equity = initial_capital
    current_pos = None
    trades = []
    equity_curve = [equity]

    ts_in_market = 0
    ts_margin_hours = 0.0
    ts_notional_hours = 0.0

    for i in range(n - 1):
        curr_row = records[i]
        next_row = records[i + 1]
        close_p = curr_row['close']
        curr_dt = curr_row['datetime']
        state = curr_row.get('regime', curr_row.get('regime_state', 'RANGE'))

        # 1) 펀딩비 (8시간마다)
        if current_pos and (i % 8 == 0):
            equity -= (current_pos.size * close_p * funding_fee_pct)

        # 2) 보유 포지션 관리
        if current_pos:
            ts_in_market += 1
            notional = current_pos.size * close_p
            margin = notional / leverage
            ts_margin_hours += margin
            ts_notional_hours += notional

            # MEAN_REVERSION
            if current_pos.engine_name == "MEAN_REVERSION":
                if current_pos.side == PositionSide.LONG:
                    # 1차 익절 (80%)
                    if curr_row['high'] >= current_pos.tp1_price:
                        closed_size = current_pos.size * 0.8
                        eff_exit_p = current_pos.tp1_price
                        gross_pnl = (eff_exit_p - current_pos.entry_price) * closed_size
                        fee = (current_pos.entry_price * closed_size * taker_fee_pct) + (eff_exit_p * closed_size * maker_fee_pct)
                        net_pnl = gross_pnl - fee
                        equity += net_pnl
                        trades.append({
                            "entry_time": current_pos.entry_time,
                            "exit_time": curr_dt,
                            "side": "LONG",
                            "engine": "MEAN_REVERSION",
                            "pnl": net_pnl,
                            "bars_held": i - current_pos.entry_bar,
                        })
                        current_pos.size -= closed_size
                        current_pos.sl_price = current_pos.entry_price  # 본전컷
                        current_pos.tp1_price = 999999.0
                    # 2차 익절 또는 손절
                    elif curr_row['high'] >= current_pos.tp2_price:
                        closed_size = current_pos.size
                        eff_exit_p = current_pos.tp2_price
                        gross_pnl = (eff_exit_p - current_pos.entry_price) * closed_size
                        fee = (current_pos.entry_price * closed_size * taker_fee_pct) + (eff_exit_p * closed_size * maker_fee_pct)
                        net_pnl = gross_pnl - fee
                        equity += net_pnl
                        trades.append({
                            "entry_time": current_pos.entry_time,
                            "exit_time": curr_dt,
                            "side": "LONG",
                            "engine": "MEAN_REVERSION",
                            "pnl": net_pnl,
                            "bars_held": i - current_pos.entry_bar,
                        })
                        current_pos = None
                    elif curr_row['low'] <= current_pos.sl_price:
                        closed_size = current_pos.size
                        eff_exit_p = current_pos.sl_price * (1.0 - slippage_pct)
                        gross_pnl = (eff_exit_p - current_pos.entry_price) * closed_size
                        fee = (current_pos.entry_price * closed_size * taker_fee_pct) + (eff_exit_p * closed_size * taker_fee_pct)
                        net_pnl = gross_pnl - fee
                        equity += net_pnl
                        trades.append({
                            "entry_time": current_pos.entry_time,
                            "exit_time": curr_dt,
                            "side": "LONG",
                            "engine": "MEAN_REVERSION",
                            "pnl": net_pnl,
                            "bars_held": i - current_pos.entry_bar,
                        })
                        current_pos = None
                    elif (i - current_pos.entry_bar) >= max_holding_bars:
                        closed_size = current_pos.size
                        eff_exit_p = close_p * (1.0 - slippage_pct)
                        gross_pnl = (eff_exit_p - current_pos.entry_price) * closed_size
                        fee = (current_pos.entry_price * closed_size * taker_fee_pct) + (eff_exit_p * closed_size * taker_fee_pct)
                        net_pnl = gross_pnl - fee
                        equity += net_pnl
                        trades.append({
                            "entry_time": current_pos.entry_time,
                            "exit_time": curr_dt,
                            "side": "LONG",
                            "engine": "MEAN_REVERSION",
                            "pnl": net_pnl,
                            "bars_held": i - current_pos.entry_bar,
                        })
                        current_pos = None
                else:  # SHORT
                    if curr_row['low'] <= current_pos.tp1_price:
                        closed_size = current_pos.size * 0.8
                        eff_exit_p = current_pos.tp1_price
                        gross_pnl = (current_pos.entry_price - eff_exit_p) * closed_size
                        fee = (current_pos.entry_price * closed_size * taker_fee_pct) + (eff_exit_p * closed_size * maker_fee_pct)
                        net_pnl = gross_pnl - fee
                        equity += net_pnl
                        trades.append({
                            "entry_time": current_pos.entry_time,
                            "exit_time": curr_dt,
                            "side": "SHORT",
                            "engine": "MEAN_REVERSION",
                            "pnl": net_pnl,
                            "bars_held": i - current_pos.entry_bar,
                        })
                        current_pos.size -= closed_size
                        current_pos.sl_price = current_pos.entry_price  # 본전컷
                        current_pos.tp1_price = 0.0
                    elif curr_row['low'] <= current_pos.tp2_price:
                        closed_size = current_pos.size
                        eff_exit_p = current_pos.tp2_price
                        gross_pnl = (current_pos.entry_price - eff_exit_p) * closed_size
                        fee = (current_pos.entry_price * closed_size * taker_fee_pct) + (eff_exit_p * closed_size * maker_fee_pct)
                        net_pnl = gross_pnl - fee
                        equity += net_pnl
                        trades.append({
                            "entry_time": current_pos.entry_time,
                            "exit_time": curr_dt,
                            "side": "SHORT",
                            "engine": "MEAN_REVERSION",
                            "pnl": net_pnl,
                            "bars_held": i - current_pos.entry_bar,
                        })
                        current_pos = None
                    elif curr_row['high'] >= current_pos.sl_price:
                        closed_size = current_pos.size
                        eff_exit_p = current_pos.sl_price * (1.0 + slippage_pct)
                        gross_pnl = (current_pos.entry_price - eff_exit_p) * closed_size
                        fee = (current_pos.entry_price * closed_size * taker_fee_pct) + (eff_exit_p * closed_size * taker_fee_pct)
                        net_pnl = gross_pnl - fee
                        equity += net_pnl
                        trades.append({
                            "entry_time": current_pos.entry_time,
                            "exit_time": curr_dt,
                            "side": "SHORT",
                            "engine": "MEAN_REVERSION",
                            "pnl": net_pnl,
                            "bars_held": i - current_pos.entry_bar,
                        })
                        current_pos = None
                    elif (i - current_pos.entry_bar) >= max_holding_bars:
                        closed_size = current_pos.size
                        eff_exit_p = close_p * (1.0 + slippage_pct)
                        gross_pnl = (current_pos.entry_price - eff_exit_p) * closed_size
                        fee = (current_pos.entry_price * closed_size * taker_fee_pct) + (eff_exit_p * closed_size * taker_fee_pct)
                        net_pnl = gross_pnl - fee
                        equity += net_pnl
                        trades.append({
                            "entry_time": current_pos.entry_time,
                            "exit_time": curr_dt,
                            "side": "SHORT",
                            "engine": "MEAN_REVERSION",
                            "pnl": net_pnl,
                            "bars_held": i - current_pos.entry_bar,
                        })
                        current_pos = None
            # TREND_FOLLOWING
            else:
                atr = curr_row['atr']
                if curr_row['high'] > current_pos.highest_price:
                    current_pos.highest_price = curr_row['high']
                    new_sl = curr_row['high'] - (atr * max_trailing_atr)
                    current_pos.sl_price = max(current_pos.sl_price, new_sl)

                if curr_row['low'] <= current_pos.sl_price:
                    closed_size = current_pos.size
                    eff_exit_p = current_pos.sl_price * (1.0 - slippage_pct)
                    gross_pnl = (eff_exit_p - current_pos.entry_price) * closed_size
                    fee = (current_pos.entry_price * closed_size * taker_fee_pct) + (eff_exit_p * closed_size * taker_fee_pct)
                    net_pnl = gross_pnl - fee
                    equity += net_pnl
                    trades.append({
                        "entry_time": current_pos.entry_time,
                        "exit_time": curr_dt,
                        "side": "LONG",
                        "engine": "TREND_FOLLOWING",
                        "pnl": net_pnl,
                        "bars_held": i - current_pos.entry_bar,
                    })
                    current_pos = None

        # 3) 신규 진입 체크 (포지션 없을 때, cooldown=0)
        if current_pos is None and i >= 36:
            if state == "RANGE" and curr_row.get('bb_lower') is not None:
                prev_row = records[i - 1]
                # 롱: BB 하단 터치 + Higher Low + 양봉 반등 + RSI <= 35
                if prev_row['low'] <= prev_row['bb_lower'] and curr_row['low'] >= (prev_row['low'] * 0.998) and curr_row['close'] > curr_row['open'] and curr_row['rsi'] <= 35.0:
                    sl = curr_row['close'] - (curr_row['atr'] * 1.2)
                    dist = abs(curr_row['close'] - sl)
                    if dist > 0:
                        sz = (equity * risk_per_trade_pct) / dist
                        eff_entry_p = next_row['open'] * (1.0 + slippage_pct)
                        current_pos = Position(
                            side=PositionSide.LONG,
                            entry_price=eff_entry_p,
                            size=sz,
                            sl_price=sl,
                            tp1_price=curr_row['bb_middle'],
                            tp2_price=curr_row['bb_upper'],
                            engine_name="MEAN_REVERSION",
                            entry_bar=i + 1,
                            entry_time=str(next_row['datetime']),
                        )
                # 숏: BB 상단 터치 + Lower High + 음봉 반락 + RSI >= 65
                elif prev_row['high'] >= prev_row['bb_upper'] and curr_row['high'] <= (prev_row['high'] * 1.002) and curr_row['close'] < curr_row['open'] and curr_row['rsi'] >= 65.0:
                    sl = curr_row['close'] + (curr_row['atr'] * 1.2)
                    dist = abs(sl - curr_row['close'])
                    if dist > 0:
                        sz = (equity * risk_per_trade_pct) / dist
                        eff_entry_p = next_row['open'] * (1.0 - slippage_pct)
                        current_pos = Position(
                            side=PositionSide.SHORT,
                            entry_price=eff_entry_p,
                            size=sz,
                            sl_price=sl,
                            tp1_price=curr_row['bb_middle'],
                            tp2_price=curr_row['bb_lower'],
                            engine_name="MEAN_REVERSION",
                            entry_bar=i + 1,
                            entry_time=str(next_row['datetime']),
                        )
            elif state == "BULL_TREND" and i >= 200:
                # 롱 돌파: 200 EMA 위 & 36봉 고점 돌파 & ADX >= 25 & +DI > -DI & 거래량 1.5배 & 몸통 45%
                prev_slice = records[i - 36 : i]
                box_high = max(r['high'] for r in prev_slice)
                candle_range = curr_row['high'] - curr_row['low']
                body_size = abs(curr_row['close'] - curr_row['open'])
                body_ok = (body_size / candle_range >= 0.45) if candle_range > 0 else False

                if (curr_row['close'] > curr_row.get('ema200', curr_row['close']) and
                    curr_row['close'] > box_high and
                    curr_row['adx'] >= 25.0 and
                    curr_row['plus_di'] > curr_row['minus_di'] and
                    curr_row.get('vol_change', 0.0) >= 0.5 and
                    body_ok):
                    sl = curr_row['close'] - (curr_row['atr'] * 1.5)
                    dist = abs(curr_row['close'] - sl)
                    if dist > 0:
                        sz = (equity * risk_per_trade_pct) / dist
                        eff_entry_p = next_row['open'] * (1.0 + slippage_pct)
                        current_pos = Position(
                            side=PositionSide.LONG,
                            entry_price=eff_entry_p,
                            size=sz,
                            sl_price=sl,
                            tp1_price=None,
                            tp2_price=None,
                            engine_name="TREND_FOLLOWING",
                            entry_bar=i + 1,
                            entry_time=str(next_row['datetime']),
                        )
                        current_pos.highest_price = curr_row['high']

        equity_curve.append(equity)

    eq_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq_arr)
    mdd_pct = np.max((peaks - eq_arr) / peaks * 100.0)
    total_trades = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    wr_pct = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    gp = sum(t['pnl'] for t in wins)
    gl = abs(sum(t['pnl'] for t in losses))
    pf = (gp / gl) if gl > 0 else 0.0

    total_ret_pct = ((equity - initial_capital) / initial_capital) * 100.0
    cagr_pct = ((equity / initial_capital) ** (1.0 / total_years) - 1.0) * 100.0
    market_exposure_pct = (ts_in_market / n) * 100.0
    avg_holding_h = (sum(t['bars_held'] for t in trades) / total_trades) if total_trades > 0 else 0.0
    ear_pct = cagr_pct / (market_exposure_pct / 100.0) if market_exposure_pct > 0 else 0.0

    return {
        "final_equity": equity,
        "total_return_pct": total_ret_pct,
        "cagr_pct": cagr_pct,
        "mdd_pct": mdd_pct,
        "total_trades": total_trades,
        "win_rate_pct": wr_pct,
        "profit_factor": pf,
        "market_exposure_pct": market_exposure_pct,
        "avg_holding_hours": avg_holding_h,
        "ear_pct": ear_pct,
    }


def run_experiment_24_clean():
    print("==================================================================================")
    print(f"[{time.strftime('%X')}] === [실험 24 정밀 재실행] cooldown=0 순수 원본 2단계 최적화 ===")
    print("==================================================================================")

    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_all = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values(by='timestamp').reset_index(drop=True)
    df_all['datetime'] = pd.to_datetime(df_all['timestamp'], unit='ms', utc=True)
    df_ind = add_all_indicators(df_all)

    # -------------------------------------------------------------
    # [1단계] 개별 독립 감도 분석 (Ablation Study)
    # -------------------------------------------------------------
    print(f"\n[{time.strftime('%X')}] >>> [1단계] 3가지 방법 개별 독립 감도 분석 진행 중...")

    # 기준 HMM (TH=0.45)
    mgr_base = Regime3StateManager(hmm_window=720, retrain_interval=168, trans_threshold=0.45)
    df_proc_base = mgr_base.calculate_regimes(df_ind)

    # 방법 1: 동적 ATR 상한선 (3.0x vs 3.5x vs 4.0x vs 4.5x vs 5.0x)
    m1_results = []
    for cap in [3.0, 3.5, 4.0, 4.5, 5.0]:
        res = run_clean_backtest(df_proc_base, max_trailing_atr=cap, max_holding_bars=12)
        res['param'] = f"ATR 상한 {cap:.1f}x"
        m1_results.append(res)

    # 방법 2: HMM 국면 임계값 (0.45 vs 0.40 vs 0.35)
    m2_results = []
    for th in [0.45, 0.40, 0.35]:
        mgr_th = Regime3StateManager(hmm_window=720, retrain_interval=168, trans_threshold=th)
        df_proc_th = mgr_th.calculate_regimes(df_ind)
        res = run_clean_backtest(df_proc_th, max_trailing_atr=3.0, max_holding_bars=12)
        res['param'] = f"HMM 임계값 {th:.2f}"
        res['df_proc'] = df_proc_th
        m2_results.append(res)

    # 방법 3: 횡보 타임스탑 (12h vs 18h vs 24h)
    m3_results = []
    for ts in [12, 18, 24]:
        res = run_clean_backtest(df_proc_base, max_trailing_atr=3.0, max_holding_bars=ts)
        res['param'] = f"횡보 타임스탑 {ts}h"
        m3_results.append(res)

    print("\n" + "=" * 105)
    print("      [1단계 독립 감도 분석 결과표 (cooldown=0)]")
    print("=" * 105)
    print(f"{'테스트 항목':<25} | {'총수익률':<10} | {'MDD':<8} | {'노출도(Exp)':<12} | {'거래수':<8} | {'승률':<8} | {'PF':<6} | {'평균보유'}")
    print("-" * 105)
    print(" [방법 1: 동적 ATR 상한선]")
    for r in m1_results:
        print(f"  {r['param']:<23} | {r['total_return_pct']:+8.2f}% | {r['mdd_pct']:6.2f}% | {r['market_exposure_pct']:10.2f}% | {r['total_trades']:5d}회 | {r['win_rate_pct']:6.1f}% | {r['profit_factor']:4.2f} | {r['avg_holding_hours']:4.1f}h")

    print("\n [방법 2: HMM 국면 전환 임계값]")
    for r in m2_results:
        print(f"  {r['param']:<23} | {r['total_return_pct']:+8.2f}% | {r['mdd_pct']:6.2f}% | {r['market_exposure_pct']:10.2f}% | {r['total_trades']:5d}회 | {r['win_rate_pct']:6.1f}% | {r['profit_factor']:4.2f} | {r['avg_holding_hours']:4.1f}h")

    print("\n [방법 3: 횡보 타임스탑]")
    for r in m3_results:
        print(f"  {r['param']:<23} | {r['total_return_pct']:+8.2f}% | {r['mdd_pct']:6.2f}% | {r['market_exposure_pct']:10.2f}% | {r['total_trades']:5d}회 | {r['win_rate_pct']:6.1f}% | {r['profit_factor']:4.2f} | {r['avg_holding_hours']:4.1f}h")
    print("=" * 105)

    # -------------------------------------------------------------
    # [2단계] 타깃 마이크로 그리드 서치 (Targeted Grid Search)
    # -------------------------------------------------------------
    print(f"\n[{time.strftime('%X')}] >>> [2단계] 핵심 결합 조합 타깃 그리드 서치 진행 중...")

    hmm_map = {
        0.45: df_proc_base,
        0.40: m2_results[1]['df_proc'],
        0.35: m2_results[2]['df_proc'],
    }

    combos = [
        # (th, atr_cap, ts_bars, label)
        (0.45, 3.0, 12, "기준 원본 베이스라인 (TH=0.45, ATR=3.0x, TS=12h)"),
        (0.45, 3.5, 18, "조합 A: 보수적 안정형 (TH=0.45, ATR=3.5x, TS=18h)"),
        (0.45, 4.0, 18, "조합 B: 추세 강화형 (TH=0.45, ATR=4.0x, TS=18h)"),
        (0.45, 4.5, 24, "조합 C: 추세 극대화형 (TH=0.45, ATR=4.5x, TS=24h)"),
        (0.40, 3.5, 18, "조합 D: 골디락스 표준형 (TH=0.40, ATR=3.5x, TS=18h)"),
        (0.40, 4.0, 18, "조합 E: 골디락스 강화형 (TH=0.40, ATR=4.0x, TS=18h)"),
        (0.40, 4.5, 24, "조합 F: 골디락스 올라운더 (TH=0.40, ATR=4.5x, TS=24h)"),
        (0.35, 4.0, 18, "조합 G: 적극적 공격형 (TH=0.35, ATR=4.0x, TS=18h)"),
        (0.35, 4.5, 24, "조합 H: 초적극 풀가동형 (TH=0.35, ATR=4.5x, TS=24h)"),
    ]

    grid_results = []
    for th, cap, ts, label in combos:
        df_target = hmm_map[th]
        res = run_clean_backtest(df_target, max_trailing_atr=cap, max_holding_bars=ts)
        res['label'] = label
        grid_results.append(res)

    print("\n\n" + "=" * 115)
    print("      [2단계 타깃 그리드 서치 종합 성과 순위표 (수익률 내림차순)]")
    print("=" * 115)
    print(f"{'조합 명칭':<45} | {'총수익률':<10} | {'MDD':<8} | {'노출도(Exp)':<12} | {'거래수':<8} | {'승률':<8} | {'PF':<6}")
    print("-" * 115)

    sorted_grid = sorted(grid_results, key=lambda x: x['total_return_pct'], reverse=True)
    for r in sorted_grid:
        name = r['label']
        ret = f"{r['total_return_pct']:+8.2f}%"
        mdd = f"{r['mdd_pct']:6.2f}%"
        exp = f"{r['market_exposure_pct']:10.2f}%"
        cnt = f"{r['total_trades']:5d}회"
        wr = f"{r['win_rate_pct']:6.1f}%"
        pf = f"{r['profit_factor']:4.2f}"
        print(f"{name:<45} | {ret} | {mdd} | {exp} | {cnt} | {wr} | {pf}")
    print("=" * 115)


if __name__ == "__main__":
    run_experiment_24_clean()
