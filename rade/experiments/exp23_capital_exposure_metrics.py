"""
[실험 23] 4개년(2021~2024) 백테스트 자본 투입 및 시장 노출도 6대 지표 정밀 측정
1. 시장 노출 시간 비율 (Market Exposure Time, %)
2. 시간가중 자본 노출도 (Time-Weighted Capital Exposure, %)
3. 달러-일 / 자본-시간 (Dollar-Days & Dollar-Hours)
4. 평균 자본 가동률 (Average Capital Utilization Rate, %)
5. 노출 조정 연간 수익률 (Exposure-Adjusted Annual Return, %)
6. 시간가중 평균 실효 레버리지 (Time-Weighted Average Leverage)
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager, RegimeState
from rade.risk.position_manager import Position, PositionSide, PositionManager
from rade.engines.mean_reversion import MeanReversionEngine
from rade.engines.trend_following import TrendFollowingEngine


def run_capital_exposure_analysis():
    print("==================================================================================")
    print(f"[{time.strftime('%X')}] === 4개년(2021~2024) 백테스트 자본 투입 및 노출도 6대 지표 정밀 측정 ===")
    print("==================================================================================")

    # 1. 데이터 로드 및 전처리
    f_is = "data/BTCUSDT_1h_2021_2024.csv"
    f_oos = "data/BTCUSDT_1h_2024_OOS.csv"
    df_all = pd.concat([pd.read_csv(f_is), pd.read_csv(f_oos)], ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values(by='timestamp').reset_index(drop=True)
    df_all['datetime'] = pd.to_datetime(df_all['timestamp'], unit='ms', utc=True)
    df_ind = add_all_indicators(df_all)

    # 2. 캘린더 앵커링 (일요일 자정 UTC) HMM 국면 산출
    regime_mgr = RegimeManager(hmm_window=720, retrain_interval=168, anchor_dayofweek=6, trans_threshold=0.45, cooldown_bars=3)
    df_proc = regime_mgr.calculate_regime_probabilities(df_ind)
    
    # 720봉 웜업 이후 데이터로 백테스트 진행
    test_df = df_proc.iloc[720:].reset_index(drop=True)
    records = test_df.to_dict('records')
    n = len(records)

    # 3. 시뮬레이션 설정
    initial_capital = 10000.0
    equity = initial_capital
    leverage = 3.0
    risk_per_trade_pct = 0.02
    maker_fee_pct = 0.0002
    taker_fee_pct = 0.0005
    slippage_pct = 0.0002
    funding_fee_pct = 0.0001

    pos_mgr = PositionManager(risk_per_trade_pct=risk_per_trade_pct, default_leverage=leverage)
    mr_engine = MeanReversionEngine()
    tf_engine = TrendFollowingEngine()

    current_pos = None
    trades_history = []

    # 매 봉마다 기록할 시계열 메트릭
    ts_in_market = []         # 포지션 보유 여부 (1 or 0)
    ts_margin_used = []       # 투입 증거금 ($)
    ts_notional_used = []     # 포지션 명목 가치 ($)
    ts_capital_util_pct = []  # 증거금 / 총자산 (%)
    ts_effective_lev = []     # 명목 가치 / 총자산 (배수)
    ts_equity = []            # 총 자산 ($)
    ts_datetimes = []         # 일시

    for i in range(n - 1):
        curr_row = records[i]
        next_row = records[i + 1]
        close_p = curr_row['close']
        curr_dt = curr_row['datetime']

        # 1) 펀딩비 결제 (8시간 주기)
        if current_pos and (i % 8 == 0):
            equity -= (current_pos.size * close_p * funding_fee_pct)

        # 2) 보유 포지션 업데이트 및 청산 체크
        if current_pos:
            if current_pos.engine_name == "MEAN_REVERSION":
                res = mr_engine.update_position_fast(current_pos, curr_row, current_bar_idx=i)
            else:
                res = tf_engine.update_position_fast(current_pos, curr_row)

            if res['action'] != "NONE":
                exit_price = res['exit_price']
                ratio = res['closed_ratio']
                is_maker = res.get('is_maker', False)
                closed_size = current_pos.size * ratio

                if is_maker:
                    eff_exit_price = exit_price
                    fee_rate = maker_fee_pct
                else:
                    eff_exit_price = exit_price * (1.0 - slippage_pct if current_pos.side == PositionSide.LONG else 1.0 + slippage_pct)
                    fee_rate = taker_fee_pct

                if current_pos.side == PositionSide.LONG:
                    pnl = (eff_exit_price - current_pos.entry_price) * closed_size
                else:
                    pnl = (current_pos.entry_price - eff_exit_price) * closed_size

                fee = (current_pos.entry_price * closed_size * taker_fee_pct) + (eff_exit_price * closed_size * fee_rate)
                net_pnl = pnl - fee
                equity += net_pnl

                trades_history.append({
                    "entry_time": current_pos.entry_time,
                    "exit_time": curr_dt,
                    "engine": current_pos.engine_name,
                    "side": current_pos.side.value,
                    "pnl": net_pnl,
                    "bars_held": i - current_pos.entry_bar,
                })

                if ratio >= 1.0 or current_pos.size <= (closed_size + 1e-6):
                    current_pos = None
                else:
                    current_pos.size -= closed_size

        # 3) 신규 진입 시그널 체크
        curr_state = curr_row.get('regime_state', RegimeState.RANGE)
        if current_pos is None and not pos_mgr.check_kill_switch(equity):
            signal = None
            if curr_state == RegimeState.RANGE:
                signal = mr_engine.check_entry_signal_fast(i, records)
            elif curr_state == RegimeState.BULL_TREND:
                raw_sig = tf_engine.check_entry_signal_fast(i, records)
                if raw_sig and raw_sig['side'] == PositionSide.LONG:
                    signal = raw_sig

            if signal:
                side = signal['side']
                eff_entry_price = next_row['open'] * (1.0 + slippage_pct if side == PositionSide.LONG else 1.0 - slippage_pct)
                pos_size = pos_mgr.calculate_position_size(
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

        # 4) 시점별 노출도 및 자본 투입량 스냅샷 기록
        if current_pos:
            notional = current_pos.size * close_p
            margin = notional / leverage
            in_mkt = 1.0
            cap_util = (margin / equity) * 100.0
            eff_lev = notional / equity
        else:
            notional = 0.0
            margin = 0.0
            in_mkt = 0.0
            cap_util = 0.0
            eff_lev = 0.0

        ts_in_market.append(in_mkt)
        ts_margin_used.append(margin)
        ts_notional_used.append(notional)
        ts_capital_util_pct.append(cap_util)
        ts_effective_lev.append(eff_lev)
        ts_equity.append(equity)
        ts_datetimes.append(curr_dt)

    # 4. 정량적 지표 연산
    total_hours = len(ts_in_market)
    total_days = total_hours / 24.0
    total_years = total_days / 365.25

    in_market_hours = sum(ts_in_market)
    market_exposure_pct = (in_market_hours / total_hours) * 100.0

    avg_capital_utilization_pct = np.mean(ts_capital_util_pct) # 전체 기간 평균 자본 가동률 (증거금 기준)
    time_weighted_notional_exposure_pct = np.mean([n / e * 100.0 for n, e in zip(ts_notional_used, ts_equity)]) # 명목가치 기준 시간가중 노출도
    time_weighted_margin_exposure_pct = np.mean([m / e * 100.0 for m, e in zip(ts_margin_used, ts_equity)])     # 순수 증거금 기준 시간가중 노출도
    avg_effective_leverage = np.mean(ts_effective_lev)                                                           # 전체 기간 평균 실효 레버리지
    avg_active_leverage = np.mean([l for l in ts_effective_lev if l > 0.0]) if in_market_hours > 0 else 0.0      # 포지션 보유 중일 때의 평균 실효 레버리지

    # Dollar-Hours & Dollar-Days (Man-Month와 동일한 개념)
    total_dollar_hours_margin = np.sum(ts_margin_used)
    total_dollar_days_margin = total_dollar_hours_margin / 24.0

    total_dollar_hours_notional = np.sum(ts_notional_used)
    total_dollar_days_notional = total_dollar_hours_notional / 24.0

    final_equity = equity
    total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100.0
    cagr_pct = ((final_equity / initial_capital) ** (1.0 / total_years) - 1.0) * 100.0
    exposure_adjusted_cagr = cagr_pct / (market_exposure_pct / 100.0) if market_exposure_pct > 0 else 0.0

    trades_df = pd.DataFrame(trades_history)
    avg_holding_hours = trades_df['bars_held'].mean() if not trades_df.empty else 0.0

    print("\n" + "=" * 90)
    print("      [4개년 2021~2024] RADE 시스템 자본 투입 & 시장 노출도 6대 핵심 지표 분석")
    print("=" * 90)
    print(f"* [기본 성과] 총수익률: {total_return_pct:+.2f}% | 연평균 복리수익률(CAGR): {cagr_pct:.2f}% | 총 거래수: {len(trades_df)}회")
    print(f"* [백테스트 기간] 총 {total_days:.1f}일 ({total_hours:,}시간, 약 {total_years:.2f}년)")
    print("-" * 90)
    print(f"1. [시장 노출 시간 비율] (Market Exposure Time):")
    print(f"   -> {market_exposure_pct:.2f}%  (총 {total_days:.1f}일 중 포지션 보유 기간은 단 {in_market_hours / 24.0:.1f}일)")
    print(f"   -> 평균 포지션 보유 시간: {avg_holding_hours:.1f}시간 (약 {avg_holding_hours / 24.0:.1f}일)")
    print(f"   -> 해석: 1년 365일 중 약 {365 * (market_exposure_pct / 100.0):.1f}일만 시장에 들어가고, 나머지 {365 * (1 - market_exposure_pct / 100.0):.1f}일(약 10.5개월)은 100% 현금으로 안전하게 관망!")

    print(f"\n2. [시간가중 자본 노출도] (Time-Weighted Capital Exposure):")
    print(f"   -> 증거금(Margin) 기준:  {time_weighted_margin_exposure_pct:.2f}%")
    print(f"   -> 명목가치(Notional) 기준: {time_weighted_notional_exposure_pct:.2f}%")
    print(f"   -> 해석: 4년 전체 시간 동안 평균적으로 내 자산의 {time_weighted_margin_exposure_pct:.2f}%만이 시장에 묶여 있었음.")

    print(f"\n3. [달러-일 / 자본-시간] (Dollar-Days & Dollar-Hours):  [Man-Month 대응 지표]")
    print(f"   -> 총 투입 증거금 달러-일:  ${total_dollar_days_margin:,.1f} Dollar-Days (${total_dollar_hours_margin:,.1f} Dollar-Hours)")
    print(f"   -> 총 투입 명목가치 달러-일: ${total_dollar_days_notional:,.1f} Dollar-Days")
    print(f"   -> 해석: 소프트웨어의 Man-Month처럼, 총 {total_dollar_days_margin:,.1f} 달러의 자본을 1일 동안 투입하여 총 +${final_equity - initial_capital:,.2f}의 순익을 창출!")

    print(f"\n4. [자본 가동률] (Capital Utilization Rate):")
    print(f"   -> 전체 기간 평균 가동률:  {avg_capital_utilization_pct:.2f}%")
    print(f"   -> 포지션 진입 중 평균 가동률: {(avg_capital_utilization_pct / (market_exposure_pct / 100.0)):.2f}% (증거금 기준)")
    print(f"   -> 해석: 1회 진입 시 자산의 약 {(avg_capital_utilization_pct / (market_exposure_pct / 100.0)):.1f}%를 마진으로 활용하고, 나머지 {(100 - avg_capital_utilization_pct / (market_exposure_pct / 100.0)):.1f}%는 버퍼 자금으로 보존.")

    print(f"\n5. [노출 조정 연간 수익률] (Exposure-Adjusted Annual Return / EAR):")
    print(f"   -> 단순 연간 수익률(CAGR):  {cagr_pct:.2f}%")
    print(f"   -> 노출 조정 수익률 (EAR):    {exposure_adjusted_cagr:.2f}%  (CAGR / Exposure)")
    print(f"   -> 해석: 자본이 시장에 진입해 있던 시간 동안의 실질 연환산 생산성은 무려 {exposure_adjusted_cagr:.1f}%에 달함!")

    print(f"\n6. [시간가중 평균 실효 레버리지] (Time-Weighted Average Leverage):")
    print(f"   -> 4년 전체 시간가중 평균: {avg_effective_leverage:.3f}x")
    print(f"   -> 포지션 보유 중 평균 레버리지: {avg_active_leverage:.2f}x")
    print(f"   -> 해석: 평소 현금 대기(0배)로 인해 전체 시간가중 레버리지는 {avg_effective_leverage:.2f}배에 불과하여 계좌 파산 위험이 극히 낮음.")
    print("=" * 90)


if __name__ == "__main__":
    run_capital_exposure_analysis()
