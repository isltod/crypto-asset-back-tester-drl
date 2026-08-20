"""
엔진 2: 추세추종 엔진 (Trend Following Engine - 추세장 전용) [정석 비대칭 손익비 버전]
- A급 돌파 선별: 24봉 박스권 돌파 + ADX >= 25 + 거래량 1.5배 이상 폭발
- 손절: 1.5 * ATR
- 익절: ATR 트레일링 스탑 (3.0 * ATR) — 조기 컷 없이 대형 추세 무제한 추적
"""
from typing import Optional, Dict, Any, List
from python.risk.position_manager import Position, PositionSide


class TrendFollowingEngine:
    """추세 국면 대형 추세 추적(ATR Trailing Stop) 매매 엔진"""

    def __init__(
        self,
        adx_threshold: float = 25.0,
        breakout_lookback: int = 24,
        sl_atr_multiplier: float = 1.5,
        trailing_atr_multiplier: float = 3.0,  # 3.0 * ATR로 노이즈 버퍼 확보
        min_vol_mult: float = 0.5,            # 거래량 20봉 평균 대비 50% 이상 폭발
    ):
        self.name = "TREND_FOLLOWING"
        self.adx_threshold = adx_threshold
        self.breakout_lookback = breakout_lookback
        self.sl_atr_multiplier = sl_atr_multiplier
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.min_vol_mult = min_vol_mult

    def check_entry_signal_fast(self, i: int, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """고속 진입 시그널 검사"""
        if i < self.breakout_lookback:
            return None

        curr = records[i]
        if curr.get('is_cooldown', False) or curr.get('adx') is None:
            return None

        adx = curr['adx']
        plus_di = curr['plus_di']
        minus_di = curr['minus_di']
        atr = curr['atr']
        close = curr['close']
        vol_change = curr.get('vol_change', 0.0)

        # 직전 N봉의 최고가 및 최저가
        prev_slice = records[i - self.breakout_lookback : i]
        box_high = max(r['high'] for r in prev_slice)
        box_low = min(r['low'] for r in prev_slice)

        # 롱 돌파 조건: 종가가 박스권 상단 돌파 + ADX >= 25 + +DI > -DI + 거래량 1.5배 이상 폭발
        if close > box_high and adx >= self.adx_threshold and plus_di > minus_di and vol_change >= self.min_vol_mult:
            sl_price = close - (atr * self.sl_atr_multiplier)
            return {
                "side": PositionSide.LONG,
                "sl_price": sl_price,
                "tp1_price": None,
                "tp2_price": None,
                "engine": self.name,
            }

        # 숏 이탈 조건: 종가가 박스권 하단 이탈 + ADX >= 25 + -DI > +DI + 거래량 1.5배 이상 폭발
        if close < box_low and adx >= self.adx_threshold and minus_di > plus_di and vol_change >= self.min_vol_mult:
            sl_price = close + (atr * self.sl_atr_multiplier)
            return {
                "side": PositionSide.SHORT,
                "sl_price": sl_price,
                "tp1_price": None,
                "tp2_price": None,
                "engine": self.name,
            }

        return None

    def update_position_fast(self, pos: Position, curr: Dict[str, Any]) -> Dict[str, Any]:
        """ATR 트레일링 스탑 업데이트 (조기 청산 없이 대형 추세 지속 추적)"""
        high = curr['high']
        low = curr['low']
        atr = curr['atr']

        if pos.side == PositionSide.LONG:
            if high > pos.highest_price:
                pos.highest_price = high

            # 3.0 * ATR 트레일링 스탑
            trailing_sl = pos.highest_price - (atr * self.trailing_atr_multiplier)
            pos.sl_price = max(pos.sl_price, trailing_sl)

            if low <= pos.sl_price:
                return {"action": "TRAILING_STOP", "exit_price": pos.sl_price, "closed_ratio": 1.0, "is_maker": False}

        elif pos.side == PositionSide.SHORT:
            if low < pos.lowest_price:
                pos.lowest_price = low

            # 3.0 * ATR 트레일링 스탑
            trailing_sl = pos.lowest_price + (atr * self.trailing_atr_multiplier)
            pos.sl_price = min(pos.sl_price, trailing_sl)

            if high >= pos.sl_price:
                return {"action": "TRAILING_STOP", "exit_price": pos.sl_price, "closed_ratio": 1.0, "is_maker": False}

        return {"action": "NONE", "exit_price": 0.0, "closed_ratio": 0.0, "is_maker": False}
