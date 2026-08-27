"""
flare.live.ensemble_paper_trader
- 8:2 앙상블 포트폴리오 (RADE 표준 80% + FLARE 5x 20%) 실시간 통합 페이퍼 트레이더
- 매 시간 RADE와 FLARE 독립 사이클 실행 후 통합 잔고 추적
- 분기별(3개월) 자동 리밸런싱 집행 (이익 확정 락인 & 변동성 수확)
- data/live/ensemble_82/ 에 통합 영속화 및 텔레그램 브리핑
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.live.paper_trader import PaperTrader
from flare.live.flare_paper_trader import FlarePaperTrader
from rade.live.notifier import TelegramNotifier

logger = logging.getLogger("EnsemblePaperTrader")


class EnsemblePaperTrader:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        rade_ratio: float = 0.80,
        flare_ratio: float = 0.20,
        instance_id: str = "ensemble_82",
    ):
        self.initial_capital = initial_capital
        self.rade_ratio = rade_ratio
        self.flare_ratio = flare_ratio
        self.instance_id = instance_id

        self.instance_dir = os.path.join(PROJECT_ROOT, "data", "live", self.instance_id)
        os.makedirs(self.instance_dir, exist_ok=True)

        self.state_file = os.path.join(self.instance_dir, "state.json")
        self.snapshots_file = os.path.join(self.instance_dir, "hourly_snapshots.csv")
        self.rebalance_file = os.path.join(self.instance_dir, "rebalance_history.csv")

        self.notifier = TelegramNotifier()

        # 하위 독립 서브 트레이더 초기화 (격리된 서브 디렉토리 사용 및 내부 시작 알림 억제)
        self.rade_trader = PaperTrader(
            symbol="BTCUSDT",
            initial_capital=self.initial_capital * self.rade_ratio,
            preset_name="STANDARD_GOLDEN",
            instance_id=f"{self.instance_id}_rade",
            suppress_start_notify=True
        )
        self.flare_trader = FlarePaperTrader(
            initial_capital=self.initial_capital * self.flare_ratio,
            leverage=5.0,
            instance_id=f"{self.instance_id}_flare",
            suppress_start_notify=True
        )

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
            "total_equity": self.initial_capital,
            "last_rebalance_quarter": None,
            "is_initialized": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _save_state(self):
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def notify_start(self):
        """앙상블 포트폴리오 최초 가동 시작 알림 발송"""
        kst_now = datetime.now(timezone.utc).astimezone(timezone(pd.Timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
        rade_init = self.initial_capital * self.rade_ratio
        flare_init = self.initial_capital * self.flare_ratio
        msg = (
            f"🟢 *[🏰 8:2 앙상블 리밸런싱 포트폴리오 가동 시작]*\n"
            f"• *시작 시각*: `{kst_now}`\n"
            f"• *총 자본*: `${self.state['total_equity']:,.2f}`\n"
            f"• *RADE 공식 표준 ({int(self.rade_ratio*100)}%)*: `${rade_init:,.2f}` (BTC HMM 안심 방패 🛡️)\n"
            f"• *FLARE 5x 스윙 ({int(self.flare_ratio*100)}%)*: `${flare_init:,.2f}` (4대 코인 펀딩비 창 ⚡)\n"
            f"• *리밸런싱 주기*: `분기별(3개월) 자동 이익 락인 & 8:2 재배분`\n"
            f"• *스케줄러*: `매 정시(00:05) 앙상블 통합 사이클 가동`"
        )
        self.notifier.send_message(msg)
        self.state["is_initialized"] = True
        self._save_state()

    def _append_snapshot(self, snapshot: Dict[str, Any]):
        df_new = pd.DataFrame([snapshot])
        if not os.path.exists(self.snapshots_file):
            df_new.to_csv(self.snapshots_file, index=False, encoding="utf-8-sig")
        else:
            df_new.to_csv(self.snapshots_file, mode="a", header=False, index=False, encoding="utf-8-sig")

    def _append_rebalance_log(self, log_record: Dict[str, Any]):
        df_new = pd.DataFrame([log_record])
        if not os.path.exists(self.rebalance_file):
            df_new.to_csv(self.rebalance_file, index=False, encoding="utf-8-sig")
        else:
            df_new.to_csv(self.rebalance_file, mode="a", header=False, index=False, encoding="utf-8-sig")

    def check_and_execute_rebalance(self, current_dt: datetime):
        """분기(Quarter: 1, 4, 7, 10월) 첫째 주에 자동 리밸런싱 집행"""
        quarter_str = f"{current_dt.year}-Q{(current_dt.month - 1) // 3 + 1}"
        last_rebalance = self.state.get("last_rebalance_quarter")

        if last_rebalance is None:
            self.state["last_rebalance_quarter"] = quarter_str
            self._save_state()
            return

        # 분기가 변경되었고 포지션이 비어있거나 정리 가능한 시점
        if quarter_str != last_rebalance:
            rade_equity = self.rade_trader.state["equity"]
            flare_equity = self.flare_trader.state["equity"]
            total_equity = rade_equity + flare_equity

            target_rade = total_equity * self.rade_ratio
            target_flare = total_equity * self.flare_ratio

            # 자산 재배분
            self.rade_trader.state["equity"] = target_rade
            self.rade_trader._save_state()

            self.flare_trader.state["cash"] = target_flare
            self.flare_trader.state["equity"] = target_flare
            self.flare_trader._save_state()

            self.state["last_rebalance_quarter"] = quarter_str
            self._save_state()

            log = {
                "timestamp": current_dt.strftime("%Y-%m-%d %H:%M:%S KST"),
                "quarter": quarter_str,
                "total_equity_before": total_equity,
                "rade_equity_before": rade_equity,
                "flare_equity_before": flare_equity,
                "rade_target": target_rade,
                "flare_target": target_flare,
            }
            self._append_rebalance_log(log)

            msg = (
                f"⚖️ *[🏰 8:2 앙상블 분기 리밸런싱 집행]*\n"
                f"• *분기*: `{quarter_str}`\n"
                f"• *총 평가 자산*: *${total_equity:,.2f}*\n"
                f"• *RADE (80%)*: ${rade_equity:,.2f} ──► *${target_rade:,.2f}*\n"
                f"• *FLARE (20%)*: ${flare_equity:,.2f} ──► *${target_flare:,.2f}*\n"
                f"• *효과*: 초과 수익 확정 락인 및 8:2 황금비율 재조정 완료! ✨"
            )
            self.notifier.send_message(msg)

    def _send_daily_report(self, curr_time_kst: str):
        """매일 오전 9시 KST 8:2 앙상블 통합 정기 브리핑 발송"""
        rade_equity = self.rade_trader.state["equity"]
        flare_equity = self.flare_trader.state["equity"]
        total_equity = rade_equity + flare_equity
        init_cap = self.state.get("initial_capital", self.initial_capital)
        total_ret_pct = ((total_equity - init_cap) / init_cap) * 100.0

        pos_rade = self.rade_trader.state.get("position")
        rade_pos_str = f"{pos_rade['side']} {pos_rade['size']:.4f} BTC" if pos_rade else "현금 대기"

        flare_positions = self.flare_trader.state.get("active_positions", {})
        if flare_positions:
            flare_pos_str = ", ".join([f"{s}({p['bars_held']}h)" for s, p in flare_positions.items()])
        else:
            flare_pos_str = "현금 대기"

        msg = (
            f"📊 *[🏰 8:2 앙상블 일일 정기 브리핑 (오전 9시)]*\n"
            f"• *기준 일시*: `{curr_time_kst}`\n"
            f"• *총 평가 자본*: *${total_equity:,.2f} ({total_ret_pct:+.2f}%)*\n"
            f"• *RADE 표준 (80%)*: ${rade_equity:,.2f} (비중: {rade_equity/total_equity*100:.1f}% | {rade_pos_str})\n"
            f"• *FLARE 5x (20%)*: ${flare_equity:,.2f} (비중: {flare_equity/total_equity*100:.1f}% | {flare_pos_str})\n"
            f"• *다음 리밸런싱*: 분기 변경 시 자동 집행 (8:2 재배분)"
        )
        self.notifier.send_message(msg)

    def execute_cycle(self, force_start_notify: bool = False, force_daily_report: bool = False):
        if force_start_notify or not self.state.get("is_initialized", False):
            self.notify_start()

        utc_now = datetime.now(timezone.utc)
        now_kst = utc_now.astimezone(timezone(pd.Timedelta(hours=9)))
        curr_time_kst = now_kst.strftime("%Y-%m-%d %H:%M:%S KST")
        today_kst_str = now_kst.strftime("%Y-%m-%d")

        # 1. RADE 표준 및 FLARE 5x 하위 사이클 순차 실행
        self.rade_trader.execute_cycle()
        self.flare_trader.execute_cycle()

        # 2. 분기 리밸런싱 검사
        self.check_and_execute_rebalance(now_kst)

        # 3. 통합 평가액 산출 및 스냅샷 기록
        rade_equity = self.rade_trader.state["equity"]
        flare_equity = self.flare_trader.state["equity"]
        total_equity = rade_equity + flare_equity
        self.state["total_equity"] = total_equity
        self._save_state()

        snapshot = {
            "timestamp": curr_time_kst,
            "total_equity": total_equity,
            "rade_equity": rade_equity,
            "flare_equity": flare_equity,
            "rade_weight": (rade_equity / total_equity) * 100.0,
            "flare_weight": (flare_equity / total_equity) * 100.0,
        }
        self._append_snapshot(snapshot)

        # 4. 매일 오전 09:00 KST 일일 정기 브리핑
        if (now_kst.hour == 9 and self.state.get("last_daily_report_date") != today_kst_str) or force_daily_report:
            self._send_daily_report(curr_time_kst)
            self.state["last_daily_report_date"] = today_kst_str
            self._save_state()

        logger.info(f"[8:2 앙상블] 통합 사이클 완료. 총 자산: ${total_equity:,.2f} (RADE: ${rade_equity:,.2f} / FLARE: ${flare_equity:,.2f})")


if __name__ == "__main__":
    ensemble = EnsemblePaperTrader(instance_id="ensemble_82")
    ensemble.execute_cycle()
