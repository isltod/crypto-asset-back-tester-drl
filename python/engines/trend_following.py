"""
엔진 2: 추세추종 엔진 (Trend Following Engine - 추세장 전용) [상위 추세 필터 탑재판]
- 200 EMA 대세 추세 정렬 (롱은 200 EMA 위에서만, 숏은 200 EMA 아래에서만)
- 36봉(1.5일) 박스권 돌파
- 캔들 몸통 45% 이상 실체 돌파 확인
- ADX >= 25 + 거래량 1.5배 폭발
- 3.0 * ATR 트레일링 스탑
"""
from typing import Optional, Dict, Any, List
from python.risk.position_manager import Position, PositionSide


class TrendFollowingEngine:
    """추세 국면 200 EMA + 36봉 돌파 + ATR 트레일링 스탑 매매 엔진"""

    def __init__(
        self,
        adx_threshold: float = 25.0,
        breakout_lookback: int = 36,           # 36봉 (1.5일) 박스권
        sl_atr_multiplier: float = 1.5,
        trailing_atr_multiplier: float = 3.0,  # 3.0 * ATR 트레일링
        min_vol_mult: float = 0.5,             # 거래량 50% 이상 증가
        min_body_ratio: float = 0.45,          # 캔들 몸통 비율 45% 이상
    ):
        self.name = "TREND_FOLLOWING"
        self.adx_threshold = adx_threshold
        self.breakout_lookback = breakout_lookback
        self.sl_atr_multiplier = sl_atr_multiplier
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.min_vol_mult = min_vol_mult
        self.min_body_ratio = min_body_ratio

    def check_entry_signal_fast(self, i: int, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """고속 진입 시그널 검사"""
        if i < max(self.breakout_lookback, 200):
            return None

        curr = records[i]
        if curr.get('is_cooldown', False) or curr.get('adx') is None:
            return None

        close = curr['close']
        open_p = curr['open']
        high = curr['high']
        low = curr['low']
        atr = curr['atr']
        adx = curr['adx']
        plus_di = curr['plus_di']
        minus_di = curr['minus_di']
        vol_change = curr.get('vol_change', 0.0)
        ema200 = curr.get('ema200', close)

        # 캔들 몸통 비율 체크 (가짜 꼬리 돌파 필터링)
        candle_range = high - low
        body_size = abs(close - open_p)
        if candle_range > 0 and (body_size / candle_range) < self.min_body_ratio:
            return None

        # 직전 36봉의 최고가 및 최저가
        prev_slice = records[i - self.breakout_lookback : i]
        box_high = max(r['high'] for r in prev_slice)
        box_low = min(r['low'] for r in prev_slice)

        # 롱 돌파 조건:
        # 1) 200 EMA 위에 위치 (대세 상승장)
        # 2) 종가가 36봉 박스권 상단 돌파
        # 3) ADX >= 25 & +DI > -DI
        # 4) 거래량 1.5배 이상 폭발
        if close > ema200 and close > box_high and adx >= self.adx_threshold and plus_di > minus_di and vol_change >= self.min_vol_mult:
            sl_price = close - (atr * self.sl_atr_multiplier)
            return {
                "side": PositionSide.LONG,
                "sl_price": sl_price,
                "tp1_price": None,
                "tp2_price": None,
                "engine": self.name,
            }

        # 숏 이탈 조건:
        # 1) 200 EMA 아래에 위치 (대세 하락장)
        # 2) 종가가 36봉 박스권 하단 이탈
        # 3) ADX >= 25 & -DI > +DI
        # 4) 거래량 1.5배 이상 폭발
        if close < ema200 and close < box_low and adx >= self.adx_threshold and minus_di > plus_di and vol_change >= self.min_vol_mult:
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
        """ATR 트레일링 스탑 업데이트"""
        high = curr['high']
        low = curr['low']
        atr = curr['atr']

        if pos.side == PositionSide.LONG:
            if high > pos.highest_price:
                pos.highest_price = high

            trailing_sl = pos.highest_price - (atr * self.trailing_atr_multiplier)
            pos.sl_price = max(pos.sl_price, trailing_sl)

            if low <= pos.sl_price:
                return {"action": "TRAILING_STOP", "exit_price": pos.sl_price, "closed_ratio": 1.0, "is_maker": False}

        elif pos.side == PositionSide.SHORT:
            if low < pos.lowest_price:
                pos.lowest_price = low

            trailing_sl = pos.lowest_price + (atr * self.trailing_atr_multiplier)
            pos.sl_price = min(pos.sl_price, trailing_sl)

            if high >= pos.sl_price:
                return {"action": "TRAILING_STOP", "exit_price": pos.sl_price, "closed_ratio": 1.0, "is_maker": False}

        return {"action": "NONE", "exit_price": 0.0, "closed_ratio": 0.0, "is_maker": False}
