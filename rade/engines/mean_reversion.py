"""
엔진 1: 평균회귀 엔진 (Mean Reversion Engine - 횡보장 전용) [고승률 방어판]
- 볼린저 밴드 수축 필터 (Bandwidth Squeeze Filter): 밴드 팽창/돌파 시 진입 차단
- 쌍바닥(Higher Low) / 쌍봉(Lower High) 반등 확인
- 1.2 * ATR 타이트 손절 (손실 최소화)
- 80:20 분할익절 (중심선 도달 시 80% 고승률 수익 확정 + 본전컷)
- 타임스탑 (12봉 동안 회귀 실패 시 탈출)
"""
from typing import Optional, Dict, Any, List
from rade.risk.position_manager import Position, PositionSide


class MeanReversionEngine:
    """횡보 국면 밴드 수축 및 쌍바닥/쌍봉 기반 고승률 평균회귀 매매 엔진"""

    def __init__(
        self,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        sl_atr_multiplier: float = 1.2,
        tp1_ratio: float = 0.8,       # 1차 중심선 익절 80% (승률 및 수익 확정 극대화)
        max_holding_bars: int = 24,   # 24봉 (24시간) 타임스탑 (최적 조합 C)
    ):
        self.name = "MEAN_REVERSION"
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp1_ratio = tp1_ratio
        self.max_holding_bars = max_holding_bars

    def check_entry_signal_fast(self, i: int, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """고속 진입 시그널 검사"""
        if i < 2:
            return None

        curr = records[i]
        prev = records[i - 1]

        if curr.get('is_cooldown', False) or curr.get('bb_lower') is None:
            return None

        # 1. 밴드 수축 필터 (밴드가 비정상적으로 팽창 중이면 횡보가 아닌 추세 폭발이므로 진입 금지)
        bw = curr.get('bb_bandwidth', 0.0)
        bw_ma50 = curr.get('bb_bandwidth_ma50', 0.0)
        if bw_ma50 > 0 and bw > (bw_ma50 * 1.15):
            return None

        close = curr['close']
        open_p = curr['open']
        low = curr['low']
        high = curr['high']
        atr = curr['atr']
        rsi = curr['rsi']

        bb_middle = curr['bb_middle']
        bb_upper = curr['bb_upper']
        bb_lower = curr['bb_lower']

        prev_low = prev['low']
        prev_bb_lower = prev['bb_lower']
        prev_high = prev['high']
        prev_bb_upper = prev['bb_upper']

        # 롱 조건 (쌍바닥 반등):
        # 1) 직전 봉 저가가 BB 하단 터치
        # 2) 현재 봉 저점이 직전 저점 지지 (Higher Low: low >= prev_low * 0.998)
        # 3) 현재 양봉 반등 (close > open)
        # 4) RSI 과매도권 (< 35)
        if prev_low <= prev_bb_lower and low >= (prev_low * 0.998) and close > open_p and rsi <= self.rsi_oversold:
            sl_price = close - (atr * self.sl_atr_multiplier)
            return {
                "side": PositionSide.LONG,
                "sl_price": sl_price,
                "tp1_price": bb_middle,
                "tp2_price": bb_upper,
                "engine": self.name,
            }

        # 숏 조건 (쌍봉 반락):
        # 1) 직전 봉 고가가 BB 상단 터치
        # 2) 현재 봉 고점이 직전 고점 저항 (Lower High: high <= prev_high * 1.002)
        # 3) 현재 음봉 반락 (close < open)
        # 4) RSI 과매수권 (> 65)
        if prev_high >= prev_bb_upper and high <= (prev_high * 1.002) and close < open_p and rsi >= self.rsi_overbought:
            sl_price = close + (atr * self.sl_atr_multiplier)
            return {
                "side": PositionSide.SHORT,
                "sl_price": sl_price,
                "tp1_price": bb_middle,
                "tp2_price": bb_lower,
                "engine": self.name,
            }

        return None

    def update_position_fast(self, pos: Position, curr: Dict[str, Any], current_bar_idx: int = 0) -> Dict[str, Any]:
        """포지션 상태 업데이트 (보수적 손절 우선 + 80:20 분할익절 및 타임스탑)"""
        high = curr['high']
        low = curr['low']
        open_p = curr['open']

        if pos.side == PositionSide.LONG:
            # 1. 손절 체크 (최우선 순위, 갭다운 대응)
            if low <= pos.sl_price:
                exit_price = min(pos.sl_price, open_p) if open_p < pos.sl_price else pos.sl_price
                return {"action": "STOP_LOSS", "exit_price": exit_price, "closed_ratio": 1.0, "is_maker": False}

            # 2. 타임스탑 체크 (진입 후 12봉 경과 시 시장가 정리)
            holding_bars = current_bar_idx - pos.entry_bar if pos.entry_bar > 0 else 0
            if holding_bars >= self.max_holding_bars:
                return {"action": "TIME_STOP", "exit_price": curr['close'], "closed_ratio": 1.0, "is_maker": False}

            # 3. 2차 전량 익절 (BB 상단)
            if pos.tp2_price and high >= pos.tp2_price:
                return {"action": "FULL_TP", "exit_price": pos.tp2_price, "closed_ratio": 1.0, "is_maker": True}

            # 4. 1차 80% 분할 익절 (BB 중심선 도달 시 80% 익절 + 본전컷 이동)
            if not pos.is_half_closed and pos.tp1_price and high >= pos.tp1_price:
                pos.is_half_closed = True
                pos.sl_price = pos.entry_price  # Breakeven
                return {"action": "HALF_TP", "exit_price": pos.tp1_price, "closed_ratio": self.tp1_ratio, "is_maker": True}

        elif pos.side == PositionSide.SHORT:
            # 1. 손절 체크 (최우선 순위, 갭업 대응)
            if high >= pos.sl_price:
                exit_price = max(pos.sl_price, open_p) if open_p > pos.sl_price else pos.sl_price
                return {"action": "STOP_LOSS", "exit_price": exit_price, "closed_ratio": 1.0, "is_maker": False}

            # 2. 타임스탑 체크 (진입 후 12봉 경과 시 시장가 정리)
            holding_bars = current_bar_idx - pos.entry_bar if pos.entry_bar > 0 else 0
            if holding_bars >= self.max_holding_bars:
                return {"action": "TIME_STOP", "exit_price": curr['close'], "closed_ratio": 1.0, "is_maker": False}

            # 3. 2차 전량 익절 (BB 하단)
            if pos.tp2_price and low <= pos.tp2_price:
                return {"action": "FULL_TP", "exit_price": pos.tp2_price, "closed_ratio": 1.0, "is_maker": True}

            # 4. 1차 80% 분할 익절
            if not pos.is_half_closed and pos.tp1_price and low <= pos.tp1_price:
                pos.is_half_closed = True
                pos.sl_price = pos.entry_price  # Breakeven
                return {"action": "HALF_TP", "exit_price": pos.tp1_price, "closed_ratio": self.tp1_ratio, "is_maker": True}

        return {"action": "NONE", "exit_price": 0.0, "closed_ratio": 0.0, "is_maker": False}
