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

# 프로젝트 루트 디렉토리 절대 경로 고정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from rade.data_collector.binance_fetcher import BinanceFuturesFetcher
from rade.utils.indicators import add_all_indicators
from rade.regime.regime_manager import RegimeManager, RegimeState
from rade.risk.position_manager import PositionSide, PositionManager
from rade.engines.mean_reversion import MeanReversionEngine
from rade.engines.trend_following import TrendFollowingEngine
from rade.live.notifier import TelegramNotifier


# 절대 경로 기준 데이터 디렉토리 설정
LIVE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "live")
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


from rade.config.presets import get_preset, StrategyConfig


class PaperTrader:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        initial_capital: float = 10000.0,
        preset_name: str = "STANDARD_GOLDEN",
        instance_id: Optional[str] = None,
        maker_fee_pct: float = 0.0002,
        taker_fee_pct: float = 0.0005,
        slippage_pct: float = 0.0002,
        funding_fee_pct: float = 0.0001,
        suppress_start_notify: bool = False,
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.preset_name = preset_name
        self.preset_config: StrategyConfig = get_preset(preset_name)
        self.instance_id = instance_id or preset_name.lower()
        self.suppress_start_notify = suppress_start_notify
        
        # 인스턴스 전용 격리 데이터 디렉토리
        self.instance_dir = os.path.join(PROJECT_ROOT, "data", "live", self.instance_id)
        os.makedirs(self.instance_dir, exist_ok=True)
        
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.slippage_pct = slippage_pct
        self.funding_fee_pct = funding_fee_pct

        self.state_file = os.path.join(self.instance_dir, "state.json")
        self.trades_file = os.path.join(self.instance_dir, "trades_history.csv")
        self.snapshots_file = os.path.join(self.instance_dir, "hourly_snapshots.csv")
        self.model_file = os.path.join(self.instance_dir, "hmm_model.pkl")

        self.fetcher = BinanceFuturesFetcher(data_dir=os.path.join(PROJECT_ROOT, "data", "live", "shared_cache"))
        self.notifier = TelegramNotifier()
        
        # 포지션 매니저에 프리셋 레버리지 및 리스크 주입
        self.pos_manager = PositionManager(
            risk_per_trade_pct=self.preset_config.trend_risk_pct,
            default_leverage=self.preset_config.leverage,
            max_leverage=self.preset_config.leverage
        )
        self.mr_engine = MeanReversionEngine(max_holding_bars=self.preset_config.mean_revert_max_holding)
        self.tf_engine = TrendFollowingEngine(
            trailing_atr_multiplier=self.preset_config.trailing_atr_multiplier,
            max_trailing_atr=self.preset_config.max_trailing_atr
        )
        self.regime_manager = RegimeManager(
            hmm_window=self.preset_config.hmm_window,
            retrain_interval=self.preset_config.retrain_interval,
            anchor_dayofweek=6,
            trans_threshold=self.preset_config.hmm_base_threshold,
            cooldown_bars=0
        )

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
            "current_regime": None,  # 최초 시작 시 None으로 설정하여 첫 국면 감지
            "last_trade_id": 0,
            "is_initialized": False,
            "last_daily_report_date": "",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _save_state(self):
        """현재 계좌 및 포지션 상태를 state.json에 저장"""
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
        msg = (
            f"🟢 *[🌟 {self.preset_config.name} 가동 시작]*\n"
            f"• *시작 시각*: `{kst_now}`\n"
            f"• *초기 자본*: `${self.state['equity']:,.2f}`\n"
            f"• *거래 대상*: `{self.symbol} (선물 1시간봉)`\n"
            f"• *레버리지*: `{self.preset_config.leverage:.1f}x` | *하락장*: `{self.preset_config.bear_mode}`\n"
            f"• *1회 리스크*: `추세 {self.preset_config.trend_risk_pct:.1%} / 횡보 {self.preset_config.mr_risk_pct:.1%}`\n"
            f"• *스케줄러*: `매 정시(00:05) 자동 분석 및 포지션 관리 가동`"
        )
        self.notifier.send_message(msg)
        self.state["is_initialized"] = True
        self._save_state()

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

    def _send_daily_report(self, curr_time_kst: str, close_p: float, curr_regime: str, p_bull: float, p_bear: float, unrealized_pnl: float):
        """매일 오전 9시 KST 정기 계좌 브리핑 발송"""
        if self.suppress_start_notify:
            return  # 앙상블 내부 서브 인스턴스는 단독 브리핑 억제

        pos = self.state.get("position")
        total_equity = self.state['equity'] + unrealized_pnl
        init_cap = self.state.get("initial_capital", self.initial_capital)
        total_ret_pct = ((total_equity - init_cap) / init_cap) * 100.0

        if pos:
            pos_info = f"`{pos['side']}` {pos['size']:.4f} BTC (진입: ${pos['entry_price']:,.2f} | 평가손익: ${unrealized_pnl:+,.2f})"
        else:
            pos_info = "보유 포지션 없음 (현금 대기)"

        # 거래 통계
        trade_cnt = self.state.get("last_trade_id", 0)

        msg = (
            f"📊 *[{self.preset_config.name} 일일 정기 브리핑 (오후 3시)]*\n"
            f"• *기준 일시*: `{curr_time_kst}`\n"
            f"• *비트코인 종가*: `${close_p:,.2f}`\n"
            f"• *현재 국면*: `{curr_regime}` (Bull:{p_bull:.1%}, Bear:{p_bear:.1%})\n"
            f"• *총 평가 자본*: *${total_equity:,.2f} ({total_ret_pct:+.2f}%)*\n"
            f"• *보유 포지션*: {pos_info}\n"
            f"• *누적 완료 거래*: {trade_cnt}회"
        )
        self.notifier.send_message(msg)

    def execute_cycle(self, force_start_notify: bool = False, force_daily_report: bool = False):
        """1시간 단위 단일 실행 사이클 (Fetch -> Regime -> Update Position -> Signal Check -> Save)"""
        logger.info(f"=== [RADE Paper Trading Cycle Start: {self.symbol}] ===")

        # 0. 최초 실행 시작 알림 체크
        if force_start_notify or not self.state.get("is_initialized", False):
            self.notify_start()

        # 1. 최근 800개 캔들 실시간 초고속 다운로드 (단일 요청)
        df_raw = self.fetcher.fetch_recent_klines(
            symbol=self.symbol,
            interval="1h",
            limit=800
        )
        if df_raw.empty or len(df_raw) < 730:
            logger.error(f"최근 캔들 데이터 부족 ({len(df_raw)}개)으로 사이클 중단.")
            return

        df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
        df_raw.sort_values(by="timestamp", inplace=True)
        df_raw.reset_index(drop=True, inplace=True)

        # 2. 지표 및 3-State HMM 국면 산출 (주 1회 일요일 09:00 KST 재학습 + 매시간 최신 사후확률 실시간 추론)
        df_ind = add_all_indicators(df_raw)
        regime_info, retrained = self.regime_manager.update_live_regime(df_ind, model_path=self.model_file)
        records = df_ind.to_dict('records')

        curr_bar = records[-1]  # 방금 마감된 최신 캔들
        utc_dt = pd.to_datetime(curr_bar['timestamp'], unit='ms', utc=True)
        
        # 실제 현재 실행 시각 (현실 시간 KST)
        now_kst = datetime.now(timezone.utc).astimezone(timezone(pd.Timedelta(hours=9)))
        curr_time_kst = now_kst.strftime('%Y-%m-%d %H:%M:%S KST')
        today_kst_str = now_kst.strftime('%Y-%m-%d')

        close_p = curr_bar['close']
        curr_regime = regime_info['regime_state']
        p_range = regime_info['p_range']
        p_bull = regime_info['p_bull']
        p_bear = regime_info['p_bear']

        logger.info(f"[{curr_time_kst}] 종가: ${close_p:,.2f} | 국면: {curr_regime} (Range:{p_range:.1%}, Bull:{p_bull:.1%}, Bear:{p_bear:.1%}) | HMM재학습: {retrained}")

        # 2-0. 주간 HMM 정기 재학습 알림 (일요일 오전 9시 KST)
        if retrained and regime_info.get("is_anchor_time", False):
            msg_retrain = (
                f"🧠 *[RADE HMM 주간 정기 재학습 완료]*\n"
                f"• *기준 시각*: `{curr_time_kst}` (매주 일요일 자정 UTC 앵커)\n"
                f"• *학습 표본*: 최근 30일(720시간) 캔들 데이터\n"
                f"• *갱신 국면*: `{curr_regime}` (Range:{p_range:.1%}, Bull:{p_bull:.1%}, Bear:{p_bear:.1%})\n"
                f"• *모델 상태*: `data/live/hmm_model.pkl` 정상 갱신 완료"
            )
            self.notifier.send_message(msg_retrain)

        # 2-1. 국면 전환(Regime Shift) 감지 및 실시간 알림
        prev_regime = self.state.get("current_regime")
        if prev_regime is not None and prev_regime != curr_regime:
            action_desc = {
                RegimeState.BULL_TREND: "🚀 [상승 추세] 추세추종 롱 엔진 가동 (동적 4.0x ATR 트레일링)",
                RegimeState.RANGE: "⚖️ [평온 횡보] 평균회귀 80:20 분할익절 엔진 가동",
                RegimeState.BEAR_PANIC: "🛡️ [위험/패닉] 현금 100% 안전 관망 (Cash Mode / No Trade)",
            }.get(curr_regime, "시장 관망")

            msg = (
                f"🔄 *[RADE 시장 국면 전환 감지]*\n"
                f"• *시각*: `{curr_time_kst}`\n"
                f"• *국면 변화*: `{prev_regime}` ➔ *`{curr_regime}`*\n"
                f"• *확률 분포*: `Range: {p_range:.1%}` | `Bull: {p_bull:.1%}` | `Bear: {p_bear:.1%}`\n"
                f"• *비트코인 종가*: `${close_p:,.2f}`\n"
                f"• *시스템 대응*: {action_desc}"
            )
            self.notifier.send_message(msg)

        # 3. 펀딩비 결제 (매 8시간 주기: 00:00, 08:00, 16:00 UTC = 09:00, 17:00, 01:00 KST)
        pos = self.state.get("position")
        current_hour_utc = utc_dt.hour
        if pos and (current_hour_utc % 8 == 0):
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
                update_res = self.mr_engine.update_position_fast(pos_obj, curr_bar, current_bar_idx=len(df_ind)-1)
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

                trade_record = {
                    "trade_id": f"LIVE-{self.state['last_trade_id']:04d}",
                    "preset": self.preset_name,
                    "engine": pos['engine'],
                    "regime_at_entry": pos.get('regime_at_entry', 'UNKNOWN'),
                    "side": pos['side'],
                    "leverage": self.preset_config.leverage,
                    "entry_time": pos['entry_time'],
                    "exit_time": curr_time_kst,
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
                    f"🎯 *[{self.preset_config.name} 페이퍼 청산]*\n"
                    f"• *시각*: `{curr_time_kst}`\n"
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
                self.pos_manager.risk_per_trade_pct = self.preset_config.mr_risk_pct  # 횡보장 프리셋 리스크
                signal = self.mr_engine.check_entry_signal_fast(last_idx, records)
            elif curr_regime == RegimeState.BULL_TREND:
                self.pos_manager.risk_per_trade_pct = self.preset_config.trend_risk_pct  # 추세장 프리셋 리스크
                raw_sig = self.tf_engine.check_entry_signal_fast(last_idx, records)
                if raw_sig and raw_sig['side'] == PositionSide.LONG:
                    signal = raw_sig
            elif curr_regime == RegimeState.BEAR_PANIC:
                if self.preset_config.bear_mode == "SHORT" and p_bear >= self.preset_config.hmm_bear_threshold:
                    self.pos_manager.risk_per_trade_pct = self.preset_config.trend_risk_pct  # 하락장 숏 리스크
                    raw_sig = self.tf_engine.check_entry_signal_fast(last_idx, records)
                    if raw_sig and raw_sig['side'] == PositionSide.SHORT:
                        signal = raw_sig
                else:
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
                        "entry_time": curr_time_kst,
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
                        f"🚀 *[{self.preset_config.name} 페이퍼 신규 진입]*\n"
                        f"• *시각*: `{curr_time_kst}`\n"
                        f"• *엔진*: `{signal['engine']}`\n"
                        f"• *국면*: `{curr_regime}` (Bull:{p_bull:.1%}, Bear:{p_bear:.1%})\n"
                        f"• *포지션*: *{side_str}* {pos_size:.4f} BTC ({self.preset_config.leverage:.1f}x)\n"
                        f"• *진입가*: ${eff_entry_price:,.2f} | *손절가(SL)*: ${signal['sl_price']:,.2f}\n"
                        f"• *투입 마진*: ${(pos_size * eff_entry_price / self.preset_config.leverage):,.2f} (1회 리스크: {self.pos_manager.risk_per_trade_pct:.1%})"
                    )
                    self.notifier.send_message(msg)

        # 6. 매 시간별 계좌 상태 스냅샷 저장
        pos = self.state.get("position")
        unrealized_pnl = 0.0
        if pos:
            if pos['side'] == "LONG":
                unrealized_pnl = (close_p - pos['entry_price']) * pos['size']
            else:
                unrealized_pnl = (pos['entry_price'] - close_p) * pos['size']

        snapshot = {
            "timestamp": curr_time_kst,
            "btc_price": close_p,
            "regime": curr_regime,
            "p_range": p_range,
            "p_bull": p_bull,
            "p_bear": p_bear,
            "has_position": bool(pos),
            "pos_side": pos['side'] if pos else "NONE",
            "pos_size": pos['size'] if pos else 0.0,
            "unrealized_pnl": unrealized_pnl,
            "account_equity": self.state['equity'] + unrealized_pnl,
        }
        # 7. 매일 오후 15:00 KST 일일 정기 브리핑 알림
        if (now_kst.hour == 15 and self.state.get("last_daily_report_date") != today_kst_str) or force_daily_report:
            self._send_daily_report(curr_time_kst, close_p, curr_regime, p_bull, p_bear, unrealized_pnl)
            self.state["last_daily_report_date"] = today_kst_str

        # 8. 상태 파일 저장
        self.state['current_regime'] = curr_regime
        self._save_state()

        logger.info(f"[{self.preset_name}] 사이클 완료. 평가 자산: ${(self.state['equity'] + unrealized_pnl):,.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RADE Paper Trader")
    parser.add_argument("--notify-start", action="store_true", help="시스템 가동 시작 알림을 텔레그램으로 강제 발송")
    parser.add_argument("--daily-report", action="store_true", help="일일 정기 브리핑을 텔레그램으로 즉시 발송")
    args = parser.parse_args()

    trader = PaperTrader(symbol="BTCUSDT", initial_capital=10000.0)

    if args.daily_report:
        # 즉시 일일 브리핑 테스트
        pos = trader.state.get("position")
        close_p = 78000.0  # 기본값
        unrealized = 0.0
        trader._send_daily_report("수동 요청", close_p, trader.state.get("current_regime", "BULL_TREND"), 0.95, 0.05, unrealized)
    else:
        trader.execute_cycle(force_start_notify=args.notify_start)

