"""
flare.live.flare_paper_trader
- 4대 메이저 코인(BTC, ETH, SOL, XRP) 펀딩비 왜곡 스윙 실시간 페이퍼 트레이더
- 바이낸스 선물 1시간봉 및 최신 펀딩비 실시간 수집 -> 진입/손절/만기 가상 체결
- data/live/flare/ (또는 지정된 instance_id) 디렉토리에 독립 영속화
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import requests

# 프로젝트 루트 디렉토리 절대 경로 고정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.live.notifier import TelegramNotifier


logger = logging.getLogger("FlarePaperTrader")


class FlarePaperTrader:
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        initial_capital: float = 10000.0,
        leverage: float = 5.0,
        instance_id: str = "flare_5x",
        allocation_ratio: float = 0.80,
        fee_taker: float = 0.0005,
        fee_maker: float = 0.0002,
        slippage: float = 0.0002,
        suppress_start_notify: bool = False,
    ):
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.instance_id = instance_id
        self.allocation_ratio = allocation_ratio
        self.fee_taker = fee_taker
        self.fee_maker = fee_maker
        self.slippage = slippage
        self.suppress_start_notify = suppress_start_notify

        self.instance_dir = os.path.join(PROJECT_ROOT, "data", "live", self.instance_id)
        os.makedirs(self.instance_dir, exist_ok=True)

        self.state_file = os.path.join(self.instance_dir, "state.json")
        self.trades_file = os.path.join(self.instance_dir, "trades_history.csv")
        self.snapshots_file = os.path.join(self.instance_dir, "hourly_snapshots.csv")

        self.notifier = TelegramNotifier()
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"state.json 로드 실패: {e}")

        return {
            "initial_capital": self.initial_capital,
            "cash": self.initial_capital,
            "equity": self.initial_capital,
            "active_positions": {},  # {symbol: position_dict}
            "last_trade_id": 0,
            "is_initialized": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _save_state(self):
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def notify_start(self):
        """시스템 최초 가동 시작 알림 발송"""
        if self.suppress_start_notify:
            self.state["is_initialized"] = True
            self._save_state()
            return

        kst_now = datetime.now(timezone.utc).astimezone(timezone(pd.Timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
        sym_str = ", ".join([s.replace("USDT", "") for s in self.symbols])
        msg = (
            f"🟢 *[⚡ FLARE 멀티코인 5x 스윙 시스템 가동 시작]*\n"
            f"• *시작 시각*: `{kst_now}`\n"
            f"• *초기 자본*: `${self.state['equity']:,.2f}`\n"
            f"• *거래 대상*: `{sym_str} (선물 4대 메이저)`\n"
            f"• *전략 엔진*: `음수 펀딩비 숏스퀴즈 반등 사냥`\n"
            f"• *레버리지*: `{self.leverage:.1f}x` | *슬롯 배분*: `1/N 동시 중복 진입`\n"
            f"• *스케줄러*: `매 정시(00:05) 자동 펀딩비/손절/24h만기 감시 가동`"
        )
        self.notifier.send_message(msg)
        self.state["is_initialized"] = True
        self._save_state()

    def _append_trade_history(self, trade_record: Dict[str, Any]):
        df_new = pd.DataFrame([trade_record])
        if not os.path.exists(self.trades_file):
            df_new.to_csv(self.trades_file, index=False, encoding="utf-8-sig")
        else:
            df_new.to_csv(self.trades_file, mode="a", header=False, index=False, encoding="utf-8-sig")

    def _append_hourly_snapshot(self, snapshot: Dict[str, Any]):
        df_new = pd.DataFrame([snapshot])
        if not os.path.exists(self.snapshots_file):
            df_new.to_csv(self.snapshots_file, index=False, encoding="utf-8-sig")
        else:
            df_new.to_csv(self.snapshots_file, mode="a", header=False, index=False, encoding="utf-8-sig")

    def fetch_latest_klines(self, symbol: str, limit: int = 50) -> pd.DataFrame:
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": symbol, "interval": "1h", "limit": limit}
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df

    def fetch_latest_funding_rate(self, symbol: str) -> float:
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        params = {"symbol": symbol}
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        return float(data.get("lastFundingRate", 0.0001))

    def _send_daily_report(self, curr_time_kst: str):
        """매일 오전 9시 KST FLARE 정기 브리핑 발송"""
        if self.suppress_start_notify:
            return  # 앙상블 내부 서브 인스턴스는 단독 브리핑 억제

        active_positions = self.state.get("active_positions", {})
        total_equity = self.state.get("equity", self.initial_capital)
        init_cap = self.state.get("initial_capital", self.initial_capital)
        total_ret_pct = ((total_equity - init_cap) / init_cap) * 100.0
        trade_cnt = self.state.get("last_trade_id", 0)

        if active_positions:
            pos_strs = [f"`{sym}`({pos['bars_held']}h 보유)" for sym, pos in active_positions.items()]
            pos_info = ", ".join(pos_strs)
        else:
            pos_info = "보유 포지션 없음 (100% 현금 대기)"

        msg = (
            f"📊 *[⚡ FLARE 5x 일일 정기 브리핑 (오후 3시)]*\n"
            f"• *기준 일시*: `{curr_time_kst}`\n"
            f"• *총 평가 자본*: *${total_equity:,.2f} ({total_ret_pct:+.2f}%)*\n"
            f"• *활성 포지션*: {pos_info}\n"
            f"• *누적 완료 거래*: {trade_cnt}회"
        )
        self.notifier.send_message(msg)

    def execute_cycle(self, force_start_notify: bool = False, force_daily_report: bool = False):
        if force_start_notify or not self.state.get("is_initialized", False):
            self.notify_start()

        utc_now = datetime.now(timezone.utc)
        now_kst = utc_now.astimezone(timezone(pd.Timedelta(hours=9)))
        curr_time_kst = now_kst.strftime("%Y-%m-%d %H:%M:%S KST")
        today_kst_str = now_kst.strftime("%Y-%m-%d")

        logger.info(f"[{self.instance_id}] FLARE 사이클 시작: {curr_time_kst}")

        active_positions = self.state.get("active_positions", {})
        cash = self.state.get("cash", self.initial_capital)

        # 1. 활성 포지션들의 손절/만기/익절 검사
        closed_symbols = []
        for sym, pos in list(active_positions.items()):
            try:
                klines = self.fetch_latest_klines(sym, limit=5)
                last_bar = klines.iloc[-2] # 방금 마감된 직전 1시간봉
                curr_bar = klines.iloc[-1]
                pos["bars_held"] = pos.get("bars_held", 0) + 1

                exit_price = None
                exit_reason = None
                is_maker = False

                # A. 손절 체크 (저가가 손절가 터치)
                if last_bar["low"] <= pos["sl_price"]:
                    exit_price = pos["sl_price"] * (1.0 - self.slippage)
                    exit_reason = "SL (손절)"
                    is_maker = False
                # B. 24시간 만기 종가 청산
                elif pos["bars_held"] >= pos.get("max_bars", 24):
                    exit_price = last_bar["close"] * (1.0 - self.slippage)
                    exit_reason = "TIMEOUT (24h 만기)"
                    is_maker = False

                if exit_price is not None:
                    closed_size = pos["position_size"]
                    raw_pnl = (exit_price - pos["entry_price"]) * closed_size
                    fee_rate = self.fee_maker if is_maker else self.fee_taker
                    exit_fee = (exit_price * closed_size) * fee_rate
                    net_trade_pnl = raw_pnl - exit_fee
                    cash += pos["margin_cost"] + net_trade_pnl

                    ret_pct = (net_trade_pnl / pos["margin_cost"]) * 100.0
                    self.state["last_trade_id"] += 1

                    trade_record = {
                        "trade_id": f"FLARE-{self.state['last_trade_id']:04d}",
                        "symbol": sym,
                        "side": "LONG",
                        "leverage": self.leverage,
                        "entry_time": pos["entry_time"],
                        "exit_time": curr_time_kst,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "size": closed_size,
                        "net_pnl": net_trade_pnl,
                        "return_pct": ret_pct,
                        "exit_reason": exit_reason,
                        "cash_after": cash,
                    }
                    self._append_trade_history(trade_record)

                    msg = (
                        f"🎯 *[⚡ FLARE 5x 포지션 청산]*\n"
                        f"• *코인*: `{sym}` (LONG {self.leverage:.1f}x)\n"
                        f"• *사유*: `{exit_reason}` (보유 {pos['bars_held']}시간)\n"
                        f"• *진입가*: ${pos['entry_price']:,.4f} ➔ *청산가*: ${exit_price:,.4f}\n"
                        f"• *수익률*: *{ret_pct:+.2f}% (${net_trade_pnl:+,.2f})*\n"
                        f"• *현재 현금*: *${cash:,.2f}*"
                    )
                    self.notifier.send_message(msg)
                    closed_symbols.append(sym)

            except Exception as e:
                logger.error(f"{sym} 포지션 체크 중 오류: {e}")

        for sym in closed_symbols:
            del active_positions[sym]

        # 2. 신규 진입 신호 검사 (비어있는 슬롯 탐색)
        n_slots = len(self.symbols)
        slot_weight = 1.0 / n_slots

        # 현재 평가액 계산
        current_margin_locked = sum(pos["margin_cost"] for pos in active_positions.values())
        total_equity = cash + current_margin_locked

        is_settle_hour = utc_now.hour in [0, 8, 16] # 펀딩비 정산 시간

        for sym in self.symbols:
            if sym not in active_positions:
                try:
                    fr = self.fetch_latest_funding_rate(sym)
                    swing_th = -0.00025 if sym == "SOLUSDT" else -0.00010

                    # 펀딩비 왜곡 발생 시 진입
                    if is_settle_hour and fr <= swing_th:
                        trade_margin = (total_equity * slot_weight) * self.allocation_ratio
                        if cash >= trade_margin:
                            klines = self.fetch_latest_klines(sym, limit=5)
                            c = klines.iloc[-1]["close"]
                            eff_entry_price = c * (1.0 + self.slippage)

                            pos_val = trade_margin * self.leverage
                            pos_size = pos_val / eff_entry_price

                            sl_pct = 0.06 if sym == "SOLUSDT" else 0.04
                            sl_price = eff_entry_price * (1.0 - sl_pct)

                            entry_fee = pos_val * self.fee_taker
                            cash -= (trade_margin + entry_fee)

                            new_pos = {
                                "symbol": sym,
                                "entry_time": curr_time_kst,
                                "entry_price": eff_entry_price,
                                "position_size": pos_size,
                                "margin_cost": trade_margin,
                                "leverage": self.leverage,
                                "sl_price": sl_price,
                                "max_bars": 24,
                                "bars_held": 0,
                                "funding_rate_at_entry": fr,
                            }
                            active_positions[sym] = new_pos

                            msg = (
                                f"🚀 *[⚡ FLARE 5x 신규 진입]*\n"
                                f"• *코인*: `{sym}` (LONG {self.leverage:.1f}x)\n"
                                f"• *진입 펀딩비*: `{fr*100:.4f}%` (음수 왜곡 포착!)\n"
                                f"• *진입가*: ${eff_entry_price:,.4f} | *손절가(SL)*: ${sl_price:,.4f}\n"
                                f"• *투입 마진*: ${trade_margin:,.2f} (포지션 크기: ${pos_val:,.2f})"
                            )
                            self.notifier.send_message(msg)

                except Exception as e:
                    logger.error(f"{sym} 신호 검사 중 오류: {e}")

        # 3. 상태 저장 및 스냅샷
        current_margin_locked = sum(pos["margin_cost"] for pos in active_positions.values())
        self.state["cash"] = cash
        self.state["active_positions"] = active_positions
        self.state["equity"] = cash + current_margin_locked
        self._save_state()

        snapshot = {
            "timestamp": curr_time_kst,
            "cash": cash,
            "locked_margin": current_margin_locked,
            "total_equity": self.state["equity"],
            "active_count": len(active_positions),
            "active_symbols": ",".join(active_positions.keys()),
        }
        self._append_hourly_snapshot(snapshot)

        # 4. 매일 오후 15:00 KST 일일 정기 브리핑
        if (now_kst.hour == 15 and self.state.get("last_daily_report_date") != today_kst_str) or force_daily_report:
            self._send_daily_report(curr_time_kst)
            self.state["last_daily_report_date"] = today_kst_str
            self._save_state()

        logger.info(f"[{self.instance_id}] FLARE 사이클 완료. 총 평가액: ${self.state['equity']:,.2f}")


if __name__ == "__main__":
    trader = FlarePaperTrader(instance_id="flare_5x", leverage=5.0)
    trader.execute_cycle()
