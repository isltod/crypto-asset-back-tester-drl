"""
엔진 1: 평균회귀 엔진 (Mean Reversion Engine - 횡보장 전용) [정석 50:50 분할익절 버전]
- 반등 확인 캔들(Reversal Confirmation Bar)
- 1.5 * ATR 동적 손절
- 1차 익절(50%): BB 중심선 도달 시 50% 익절 + 손절가를 본전(Breakeven)으로 상향
- 2차 익절(50%): 반대편 BB 도달 시 전량 청산
"""
from typing import Optional, Dict, Any, List
from python.risk.position_manager import Position, PositionSide


class MeanReversionEngine:
    """횡보 국면 50:50 분할익절 평균회귀 매매 엔진"""

    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        sl_atr_multiplier: float = 1.5,
        min_vol_mult: float = 0.2,
    ):
        self.name = "MEAN_REVERSION"
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.sl_atr_multiplier = sl_atr_multiplier
        self.min_vol_mult = min_vol_mult

    def check_entry_signal_fast(self, i: int, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """고속 진입 시그널 검사"""
        if i < 2:
            return None

        curr = records[i]
        prev = records[i - 1]

        if curr.get('is_cooldown', False) or curr.get('bb_lower') is None:
            return None

        close = curr['close']
        open_p = curr['open']
        atr = curr['atr']
        rsi = curr['rsi']
        vol_change = curr.get('vol_change', 0.0)

        bb_middle = curr['bb_middle']
        bb_upper = curr['bb_upper']
        bb_lower = curr['bb_lower']

        prev_low = prev['low']
        prev_bb_lower = prev['bb_lower']
        prev_high = prev['high']
        prev_bb_upper = prev['bb_upper']

        # 롱 조건: 직전 봉 BB 하단 터치 + 현재 양봉 반등 + RSI 과매도 + 거래량 증가
        if prev_low <= prev_bb_lower and close > open_p and rsi <= self.rsi_oversold and vol_change >= self.min_vol_mult:
            sl_price = close - (atr * self.sl_atr_multiplier)
            return {
                "side": PositionSide.LONG,
                "sl_price": sl_price,
                "tp1_price": bb_middle,
                "tp2_price": bb_upper,
                "engine": self.name,
            }

        # 숏 조건: 직전 봉 BB 상단 터치 + 현재 음봉 반락 + RSI 과매수 + 거래량 증가
        if prev_high >= prev_bb_upper and close < open_p and rsi >= self.rsi_overbought and vol_change >= self.min_vol_mult:
            sl_price = close + (atr * self.sl_atr_multiplier)
            return {
                "side": PositionSide.SHORT,
                "sl_price": sl_price,
                "tp1_price": bb_middle,
                "tp2_price": bb_lower,
                "engine": self.name,
            }

        return None

    def update_position_fast(self, pos: Position, curr: Dict[str, Any]) -> Dict[str, Any]:
        """포지션 상태 업데이트 및 50% 분할익절 판정"""
        high = curr['high']
        low = curr['low']

        if pos.side == PositionSide.LONG:
            # 1. 손절 체크
            if low <= pos.sl_price:
                return {"action": "STOP_LOSS", "exit_price": pos.sl_price, "closed_ratio": 1.0, "is_maker": False}

            # 2. 2차 전량 익절 (BB 상단)
            if pos.tp2_price and high >= pos.tp2_price:
                return {"action": "FULL_TP", "exit_price": pos.tp2_price, "closed_ratio": 1.0, "is_maker": True}

            # 3. 1차 50% 분할 익절 (BB 중심선 도달 시 50% 익절 + 본전컷 이동)
            if not pos.is_half_closed and pos.tp1_price and high >= pos.tp1_price:
                pos.is_half_closed = True
                pos.sl_price = pos.entry_price  # Breakeven
                return {"action": "HALF_TP", "exit_price": pos.tp1_price, "closed_ratio": 0.5, "is_maker": True}

        elif pos.side == PositionSide.SHORT:
            # 1. 손절 체크
            if high >= pos.sl_price:
                return {"action": "STOP_LOSS", "exit_price": pos.sl_price, "closed_ratio": 1.0, "is_maker": False}

            # 2. 2차 전량 익절 (BB 하단)
            if pos.tp2_price and low <= pos.tp2_price:
                return {"action": "FULL_TP", "exit_price": pos.tp2_price, "closed_ratio": 1.0, "is_maker": True}

            # 3. 1차 50% 분할 익절
            if not pos.is_half_closed and pos.tp1_price and low <= pos.tp1_price:
                pos.is_half_closed = True
                pos.sl_price = pos.entry_price  # Breakeven
                return {"action": "HALF_TP", "exit_price": pos.tp1_price, "closed_ratio": 0.5, "is_maker": True}

        return {"action": "NONE", "exit_price": 0.0, "closed_ratio": 0.0, "is_maker": False}
