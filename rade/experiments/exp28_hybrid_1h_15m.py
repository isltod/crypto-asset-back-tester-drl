"""
[실험 28-2] 1H 국면 + 15m 추세 롱 전용 하이브리드 정밀 검증
- 배경: 15m 평균회귀(MR)는 잔파도 노이즈로 휩소 손실(-$3,241)을 냄.
- 개선:
  - 1H RANGE 국면: 검증된 1H 볼린저 평균회귀 유지
  - 1H BULL 국면: 15m 추세 롱(눌림목 & 6h 돌파)으로 기회 확장
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators, compute_atr, compute_rsi, compute_bollinger_bands
from rade.regime.regime_manager import RegimeManager, RegimeState
from rade.risk.position_manager import Position, PositionSide, PositionManager


def run_experiment_28_refined():
    print("=" * 95)
    print("      [실험 28-2] 1H 평균회귀(유지) + 15m 추세 롱 하이브리드 정밀 백테스트")
    print("=" * 95)

    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_1h = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_1h["datetime"] = pd.to_datetime(df_1h["timestamp"], unit="ms", utc=True)
    df_1h_ind = add_all_indicators(df_1h)

    reg_mgr = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.45, cooldown_bars=0)
    df_1h_proc = reg_mgr.calculate_regime_probabilities(df_1h_ind)

    # 15m 데이터 로드
    df_15m = pd.read_csv("data/BTCUSDT_15m.csv")
    df_15m["datetime"] = pd.to_datetime(df_15m["timestamp"], unit="ms", utc=True)
    df_15m["ema20"] = df_15m["close"].ewm(span=20, adjust=False).mean()
    df_15m["ema50"] = df_15m["close"].ewm(span=50, adjust=False).mean()
    df_15m["atr_15m"] = compute_atr(df_15m["high"], df_15m["low"], df_15m["close"], period=14)
    vol_ma20 = df_15m["volume"].rolling(window=20).mean()
    df_15m["vol_change"] = (df_15m["volume"] - vol_ma20) / (vol_ma20 + 1e-10)

    # 1H 지표 병합
    df_1h_sub = df_1h_proc[["datetime", "regime_state", "atr", "ema200", "bb_lower", "bb_middle", "rsi"]].copy()
    df_1h_sub.rename(columns={"atr": "atr_1h", "ema200": "ema200_1h", "regime_state": "regime_1h", "bb_lower": "bb_lower_1h", "bb_middle": "bb_mid_1h", "rsi": "rsi_1h"}, inplace=True)

    df_hybrid = pd.merge_asof(df_15m.sort_values("datetime"), df_1h_sub.sort_values("datetime"), on="datetime", direction="backward")
    df_hybrid["regime_1h"] = df_hybrid["regime_1h"].fillna(RegimeState.RANGE)
    df_hybrid["atr_1h"] = df_hybrid["atr_1h"].bfill()
    df_hybrid["ema200_1h"] = df_hybrid["ema200_1h"].bfill()

    test_slice = df_hybrid.iloc[2880:].reset_index(drop=True)
    records = test_slice.to_dict("records")
    n = len(records)

    equity = 10000.0
    current_pos = None
    trades_history = []
    equity_curve = [equity]
    pos_mgr = PositionManager(risk_per_trade_pct=0.02, default_leverage=3.0)

    for i in range(n - 1):
        curr = records[i]
        nxt = records[i + 1]
        date_str = str(curr.get("datetime", i))[:10]
        pos_mgr.update_day(date_str, equity)
        regime = curr.get("regime_1h", RegimeState.RANGE)

        # 펀딩비 (매 8시간 = 32봉)
        if current_pos and (i % 32 == 0):
            equity -= (current_pos.size * curr["close"] * 0.0001)

        # 보유 포지션 관리
        if current_pos:
            closed = False
            action = "NONE"
            exit_price = 0.0
            ratio = 1.0
            is_maker = False
            high = curr["high"]
            low = curr["low"]
            open_p = curr["open"]
            close = curr["close"]

            if current_pos.engine_name == "HYBRID_TREND":
                if low <= current_pos.sl_price:
                    action = "TRAILING_STOP"
                    exit_price = min(current_pos.sl_price, open_p) if open_p < current_pos.sl_price else current_pos.sl_price
                    closed = True
                else:
                    if high > current_pos.highest_price:
                        current_pos.highest_price = high
                    trail_dist = curr["atr_1h"] * 4.5
                    new_sl = current_pos.highest_price - trail_dist
                    if new_sl > current_pos.sl_price:
                        current_pos.sl_price = new_sl

            elif current_pos.engine_name == "1H_MR":
                if low <= current_pos.sl_price:
                    action = "STOP_LOSS"
                    exit_price = min(current_pos.sl_price, open_p) if open_p < current_pos.sl_price else current_pos.sl_price
                    closed = True
                elif not current_pos.is_half_closed and high >= curr["bb_mid_1h"]:
                    action = "TP1_MAKER"
                    exit_price = curr["bb_mid_1h"]
                    ratio = 0.80
                    is_maker = True
                    current_pos.is_half_closed = True
                    current_pos.sl_price = current_pos.entry_price
                elif (i - current_pos.entry_bar) >= 96: # 24h
                    action = "TIME_STOP"
                    exit_price = close
                    closed = True

            if action != "NONE":
                closed_size = current_pos.size * ratio
                eff_exit_p = exit_price if is_maker else (exit_price * (1.0 - 0.0002))
                fee_rate = 0.0002 if is_maker else 0.0005
                pnl = (eff_exit_p - current_pos.entry_price) * closed_size
                fee = (current_pos.entry_price * closed_size * 0.0005) + (eff_exit_p * closed_size * fee_rate)
                net_pnl = pnl - fee
                equity += net_pnl

                trades_history.append({
                    "entry_time": current_pos.entry_time,
                    "exit_time": curr["datetime"],
                    "engine": current_pos.engine_name,
                    "side": current_pos.side.value,
                    "entry_price": current_pos.entry_price,
                    "exit_price": eff_exit_p,
                    "size": closed_size,
                    "pnl": net_pnl,
                    "return_pct": (net_pnl / equity) * 100 if equity > 0 else 0.0,
                    "reason": action,
                })
                if closed or ratio >= 1.0 or current_pos.size <= (closed_size + 1e-6):
                    current_pos = None
                else:
                    current_pos.size -= closed_size

        # 신규 진입
        if current_pos is None and not pos_mgr.check_kill_switch(equity):
            signal = None
            close = curr["close"]
            open_p = curr["open"]

            # 1H BULL 국면 -> 15m 정밀 눌림목 진입
            if regime == RegimeState.BULL_TREND:
                is_bull_ctx = close > curr["ema200_1h"]
                is_pullback = (curr["low"] <= curr["ema50"] * 1.002) and (close > curr["ema20"]) and (close > open_p)
                prev_24 = records[max(0, i - 24) : i]
                box_high_15m = max(r["high"] for r in prev_24)
                is_breakout = (close > box_high_15m) and (curr["vol_change"] >= 0.5)

                if is_bull_ctx and (is_pullback or is_breakout):
                    sl_p = close - (curr["atr_1h"] * 1.5)
                    signal = {"side": PositionSide.LONG, "sl_price": sl_p, "engine": "HYBRID_TREND"}

            # 1H RANGE 국면 -> 1H 볼린저 하단 반등 진입 (정시 캔들에서만 체크)
            elif regime == RegimeState.RANGE:
                # 1시간 정시 캔들(00분)일 때만 1H BB 반등 체크
                dt_str = str(curr.get("datetime", ""))
                is_hour_bar = (":00:00" in dt_str)
                if is_hour_bar:
                    if curr["low"] < curr["bb_lower_1h"] and close > open_p and curr["rsi_1h"] < 35.0:
                        sl_p = min(curr["low"], close - (curr["atr_1h"] * 1.5))
                        signal = {"side": PositionSide.LONG, "sl_price": sl_p, "engine": "1H_MR"}

            if signal:
                raw_entry = nxt["open"]
                eff_entry = raw_entry * 1.0002
                pos_size = pos_mgr.calculate_position_size(equity=equity, entry_price=eff_entry, sl_price=signal["sl_price"], side=signal["side"], weight=1.0)
                if pos_size > 0.0001:
                    current_pos = Position(
                        side=signal["side"],
                        entry_price=eff_entry,
                        size=pos_size,
                        sl_price=signal["sl_price"],
                        engine_name=signal["engine"],
                        entry_bar=i + 1,
                        entry_time=str(nxt.get("datetime", i + 1))
                    )

        equity_curve.append(equity)

    eq_arr = np.array(equity_curve)
    tot_ret = ((eq_arr[-1] - 10000.0) / 10000.0) * 100.0
    peaks = np.maximum.accumulate(eq_arr)
    mdd = float(np.max((peaks - eq_arr) / peaks)) * 100.0
    trades_df = pd.DataFrame(trades_history)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    wr = len(wins) / len(trades_df) * 100.0 if len(trades_df) > 0 else 0.0
    gp = wins["pnl"].sum() if len(wins) > 0 else 0.0
    gl = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-10
    pf = gp / gl

    print("\n" + "=" * 95)
    print("                [ 개선된 1H MR + 15m 추세 롱 하이브리드 성과 비교 ]")
    print("=" * 95)
    print(f" * 4개년 총수익률:     {tot_ret:+.2f}% (최종 자산: ${eq_arr[-1]:,.2f})")
    print(f" * 최대 낙폭 (MDD):     {mdd:.2f}%")
    print(f" * 손익비 (PF):         {pf:.2f}")
    print(f" * 승률 (Win Rate):     {wr:.1f}%")
    print(f" * 총 거래 횟수:        {len(trades_df)}회 (연 {len(trades_df)/3.92:.1f}회)")
    print("-" * 95)

    for eng in ["HYBRID_TREND", "1H_MR"]:
        sub = trades_df[trades_df["engine"] == eng]
        if len(sub) > 0:
            pnl = sub["pnl"].sum()
            e_wr = len(sub[sub["pnl"] > 0]) / len(sub) * 100.0
            print(f"  [{eng:<12}] 수익: {pnl:+10.2f}$ | 거래: {len(sub):3d}회 | 승률: {e_wr:5.1f}%")
    print("=" * 95)


if __name__ == "__main__":
    run_experiment_28_refined()
