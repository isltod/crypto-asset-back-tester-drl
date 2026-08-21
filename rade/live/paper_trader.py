"""
RADE 페이퍼 트레이딩 실시간 엔진 (Paper Trader Engine)
- 매 정시(00분 05초) Cron 또는 독립 데몬으로 실행
- 바이낸스 선물 1시간봉 실시간 수집 -> 3-State HMM 국면 판정 -> 가상 체결 및 포지션 관리
- 3중 영속화: state.json, trades_history.csv, hourly_snapshots.csv
- 텔레그램 실시간 알림 발송
"""
import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rade.data_collector.binance_fetcher import BinanceFuturesFetcher
from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager, RegimeState
from rade.risk.position_manager import PositionSide, PositionManager
from rade.engines.mean_reversion import MeanReversionEngine
from rade.engines.trend_following import TrendFollowingEngine
from rade.live.notifier import TelegramNotifier


# 로깅 설정
LIVE_DATA_DIR = os.path.join("data", "live")
os.makedirs(LIVE_DATA_DIR, exist_ok=True)
LOG_FILE = os.path.join(LIVE_DATA_DIR, "paper_trader.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PaperTrader")


class PaperTrader:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        initial_capital: float = 10000.0,
        risk_per_trade_pct: float = 0.02,
        leverage: float = 3.0,
        maker_fee_pct: float = 0.0002,
        taker_fee_pct: float = 0.0005,
        slippage_pct: float = 0.0002,
        funding_fee_pct: float = 0.0001,
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.leverage = leverage
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.slippage_pct = slippage_pct
        self.funding_fee_pct = funding_fee_pct

        self.state_file = os.path.join(LIVE_DATA_DIR, "state.json")
        self.trades_file = os.path.join(LIVE_DATA_DIR, "trades_history.csv")
        self.snapshots_file = os.path.join(LIVE_DATA_DIR, "hourly_snapshots.csv")

        self.fetcher = BinanceFuturesFetcher(data_dir=LIVE_DATA_DIR)
        self.notifier = TelegramNotifier()
        self.pos_manager = PositionManager(risk_per_trade_pct=risk_per_trade_pct, default_leverage=leverage)
        self.mr_engine = MeanReversionEngine()
        self.tf_engine = TrendFollowingEngine()
        self.regime_manager = RegimeManager(hmm_window=720, retrain_interval=168, trans_threshold=0.45, cooldown_bars=3)

        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """로컬 state.json 파일에서 현재 계좌 및 포지션 상태 로드"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"state.json 로드 실패: {e}")

        # 초기 기본 상태
        return {
            "initial_capital": self.initial_capital,
            "equity": self.initial_capital,
            "position": None,  # 보유 포지션 객체 딕셔너리
            "current_regime": RegimeState.RANGE,
            "last_trade_id": 0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _save_state(self):
        """현재 계좌 및 포지션 상태를 state.json에 저장"""
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def _append_trade_history(self, trade_record: Dict[str, Any]):
        """완료된 거래 내역을 trades_history.csv에 영구 기록"""
        df_new = pd.DataFrame([trade_record])
        if not os.path.exists(self.trades_file):
            df_new.to_csv(self.trades_file, index=False, encoding="utf-8-sig")
        else:
            df_new.to_csv(self.trades_file, mode="a", header=False, index=False, encoding="utf-8-sig")

    def _append_hourly_snapshot(self, snapshot: Dict[str, Any]):
        """매 시간별 계좌 및 국면 스냅샷을 hourly_snapshots.csv에 기록"""
        df_new = pd.DataFrame([snapshot])
        if not os.path.exists(self.snapshots_file):
            df_new.to_csv(self.snapshots_file, index=False, encoding="utf-8-sig")
        else:
            df_new.to_csv(self.snapshots_file, mode="a", header=False, index=False, encoding="utf-8-sig")

    def execute_cycle(self):
        """1시간 단위 단일 실행 사이클 (Fetch -> Regime -> Update Position -> Signal Check -> Save)"""
        logger.info(f"=== [RADE Paper Trading Cycle Start: {self.symbol}] ===")

        # 1. 최근 800개 캔들 실시간 다운로드
        df_raw = self.fetcher.fetch_klines(
            symbol=self.symbol,
            interval="1h",
            limit=800
        )
        if df_raw.empty or len(df_raw) < 730:
            logger.error("최근 캔들 데이터 부족으로 사이클 중단.")
            return

        df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
        df_raw.sort_values(by="timestamp", inplace=True)
        df_raw.reset_index(drop=True, inplace=True)

        # 2. 지표 및 3-State HMM 국면 산출
        df_ind = add_all_indicators(df_raw)
        df_proc = self.regime_manager.calculate_regime_probabilities(df_ind)
        records = df_proc.to_dict('records')

        curr_bar = records[-1]  # 방금 마감된 최신 캔들
        curr_time = str(curr_bar.get('datetime', ''))
        close_p = curr_bar['close']
        curr_regime = curr_bar.get('regime_state', RegimeState.RANGE)
        p_range = curr_bar.get('p_range', 0.0)
        p_bull = curr_bar.get('p_bull', 0.0)
        p_bear = curr_bar.get('p_bear', 0.0)

        logger.info(f"[{curr_time}] 종가: ${close_p:,.2f} | 국면: {curr_regime} (Range:{p_range:.1%}, Bull:{p_bull:.1%}, Bear:{p_bear:.1%})")

        # 3. 펀딩비 결제 (매 8시간 주기: 00:00, 08:00, 16:00 UTC)
        pos = self.state.get("position")
        current_hour = pd.to_datetime(curr_bar['timestamp'], unit='ms', utc=True).hour
        if pos and (current_hour % 8 == 0):
            notional = pos['size'] * close_p
            funding_cost = notional * self.funding_fee_pct
            self.state['equity'] -= funding_cost
            pos['accum_funding_fee'] = pos.get('accum_funding_fee', 0.0) + funding_cost
            logger.info(f"[펀딩비 결제] -${funding_cost:.2f} 차감 (현재 자본: ${self.state['equity']:,.2f})")

        # 4. 보유 포지션 업데이트 및 익절/손절 체크
        if pos:
            pos_side = PositionSide.LONG if pos['side'] == "LONG" else PositionSide.SHORT
            # 가상 Position 객체 어댑터
            class PosAdapter:
                def __init__(self, d):
                    self.side = pos_side
                    self.entry_price = d['entry_price']
                    self.size = d['size']
                    self.sl_price = d['sl_price']
                    self.tp1_price = d.get('tp1_price')
                    self.tp2_price = d.get('tp2_price')
                    self.highest_price = d.get('highest_price', d['entry_price'])
                    self.lowest_price = d.get('lowest_price', d['entry_price'])
                    self.is_half_closed = d.get('is_half_closed', False)
                    self.entry_bar = d.get('entry_bar', 0)
                    self.engine_name = d.get('engine', 'TREND_FOLLOWING')

            pos_obj = PosAdapter(pos)

            if pos_obj.engine_name == "MEAN_REVERSION":
                update_res = self.mr_engine.update_position_fast(pos_obj, curr_bar, current_bar_idx=len(records)-1)
            else:
                update_res = self.tf_engine.update_position_fast(pos_obj, curr_bar)

            # 포지션 상태 동기화
            pos['sl_price'] = pos_obj.sl_price
            pos['highest_price'] = pos_obj.highest_price
            pos['lowest_price'] = pos_obj.lowest_price
            pos['is_half_closed'] = pos_obj.is_half_closed

            action = update_res['action']
            if action != "NONE":
                exit_price = update_res['exit_price']
                ratio = update_res['closed_ratio']
                is_maker = update_res.get('is_maker', False)
                closed_size = pos['size'] * ratio

                if is_maker:
                    eff_exit_p = exit_price
                    fee_rate = self.maker_fee_pct
                else:
                    eff_exit_p = exit_price * (1.0 - self.slippage_pct if pos_side == PositionSide.LONG else 1.0 + self.slippage_pct)
                    fee_rate = self.taker_fee_pct

                if pos_side == PositionSide.LONG:
                    gross_pnl = (eff_exit_p - pos['entry_price']) * closed_size
                else:
                    gross_pnl = (pos['entry_price'] - eff_exit_p) * closed_size

                entry_fee = pos['entry_price'] * closed_size * self.taker_fee_pct
                exit_fee = eff_exit_p * closed_size * fee_rate
                funding_fee = pos.get('accum_funding_fee', 0.0) * ratio
                net_pnl = gross_pnl - entry_fee - exit_fee - funding_fee

                self.state['equity'] += net_pnl
                ret_pct = (net_pnl / self.state['equity']) * 100.0

                self.state['last_trade_id'] += 1
                trade_record = {
                    "trade_id": f"LIVE-{self.state['last_trade_id']:04d}",
                    "engine": pos['engine'],
                    "regime_at_entry": pos.get('regime_at_entry', 'UNKNOWN'),
                    "side": pos['side'],
                    "leverage": self.leverage,
                    "entry_time": pos['entry_time'],
                    "exit_time": curr_time,
                    "entry_price": pos['entry_price'],
                    "exit_price": eff_exit_p,
                    "size": closed_size,
                    "gross_pnl": gross_pnl,
                    "entry_fee": entry_fee,
                    "exit_fee": exit_fee,
                    "funding_fee": funding_fee,
                    "net_pnl": net_pnl,
                    "return_pct": ret_pct,
                    "equity_after": self.state['equity'],
                    "exit_reason": action,
                }
                self._append_trade_history(trade_record)

                msg = (
                    f"🎯 *[RADE 페이퍼 포지션 청산]*\n"
                    f"• *사유*: `{action}`\n"
                    f"• *포지션*: `{pos['side']}` {closed_size:.4f} BTC ({pos['engine']})\n"
                    f"• *진입가*: ${pos['entry_price']:,.2f} ➔ *청산가*: ${eff_exit_p:,.2f}\n"
                    f"• *순손익(Net)*: *${net_pnl:+,.2f} ({ret_pct:+.2f}%)*\n"
                    f"• *수수료/펀딩비*: -${(entry_fee + exit_fee + funding_fee):,.2f}\n"
                    f"• *현재 총자산*: *${self.state['equity']:,.2f}*"
                )
                self.notifier.send_message(msg)

                if ratio >= 1.0 or pos['size'] <= (closed_size + 1e-6):
                    self.state['position'] = None
                    pos = None
                else:
                    pos['size'] -= closed_size

        # 5. 신규 진입 시그널 검사 (포지션이 없을 때)
        if self.state.get("position") is None and not self.pos_manager.check_kill_switch(self.state['equity']):
            signal = None
            last_idx = len(records) - 1

            if curr_regime == RegimeState.RANGE:
                signal = self.mr_engine.check_entry_signal_fast(last_idx, records)
            elif curr_regime == RegimeState.BULL_TREND:
                raw_sig = self.tf_engine.check_entry_signal_fast(last_idx, records)
                if raw_sig and raw_sig['side'] == PositionSide.LONG:
                    signal = raw_sig
            elif curr_regime == RegimeState.BEAR_PANIC:
                signal = None  # 현금 100% 관망

            if signal:
                side = signal['side']
                side_str = "LONG" if side == PositionSide.LONG else "SHORT"
                # 현재 봉 마감가에 실시간 슬리피지 가산하여 다음 봉 시가 체결 모델링
                eff_entry_price = close_p * (1.0 + self.slippage_pct if side == PositionSide.LONG else 1.0 - self.slippage_pct)

                pos_size = self.pos_manager.calculate_position_size(
                    equity=self.state['equity'],
                    entry_price=eff_entry_price,
                    sl_price=signal['sl_price'],
                    side=side,
                    weight=1.0,
                )

                if pos_size > 0.0001:
                    new_pos = {
                        "side": side_str,
                        "engine": signal['engine'],
                        "regime_at_entry": curr_regime,
                        "entry_time": curr_time,
                        "entry_price": eff_entry_price,
                        "size": pos_size,
                        "sl_price": signal['sl_price'],
                        "tp1_price": signal.get('tp1_price'),
                        "tp2_price": signal.get('tp2_price'),
                        "highest_price": eff_entry_price,
                        "lowest_price": eff_entry_price,
                        "is_half_closed": False,
                        "accum_funding_fee": 0.0,
                        "entry_bar": len(records) - 1,
                    }
                    self.state['position'] = new_pos

                    msg = (
                        f"🚀 *[RADE 페이퍼 신규 진입]*\n"
                        f"• *엔진*: `{signal['engine']}`\n"
                        f"• *국면*: `{curr_regime}` (Bull:{p_bull:.1%}, Bear:{p_bear:.1%})\n"
                        f"• *포지션*: *{side_str}* {pos_size:.4f} BTC ({self.leverage}x)\n"
                        f"• *진입가*: ${eff_entry_price:,.2f} | *손절가(SL)*: ${signal['sl_price']:,.2f}\n"
                        f"• *계좌 자본*: ${self.state['equity']:,.2f}"
                    )
                    self.notifier.send_message(msg)

        # 6. 매 시간별 계좌 스냅샷 기록
        unrealized_pnl = 0.0
        pos = self.state.get("position")
        if pos:
            if pos['side'] == "LONG":
                unrealized_pnl = (close_p - pos['entry_price']) * pos['size']
            else:
                unrealized_pnl = (pos['entry_price'] - close_p) * pos['size']

        snapshot = {
            "timestamp": curr_bar['timestamp'],
            "datetime": curr_time,
            "btc_close": close_p,
            "regime_state": curr_regime,
            "p_range": p_range,
            "p_bull": p_bull,
            "p_bear": p_bear,
            "has_position": bool(pos),
            "pos_side": pos['side'] if pos else "NONE",
            "pos_size": pos['size'] if pos else 0.0,
            "unrealized_pnl": unrealized_pnl,
            "account_equity": self.state['equity'] + unrealized_pnl,
        }
        self._append_hourly_snapshot(snapshot)

        # 7. 상태 파일 저장
        self.state['current_regime'] = curr_regime
        self._save_state()

        logger.info(f"사이클 완료. 평가 자산: ${(self.state['equity'] + unrealized_pnl):,.2f}")


if __name__ == "__main__":
    trader = PaperTrader(symbol="BTCUSDT", initial_capital=10000.0)
    trader.execute_cycle()
