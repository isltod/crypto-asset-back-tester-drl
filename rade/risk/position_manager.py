"""
리스크 관리 및 포지션 관리자 모듈
- 1% Fixed Risk Model (손절폭 기반 포지션 사이징)
- 레버리지 및 격리 마진 관리
- Trailing OCO (1차 익절 후 본전컷 이동)
- ATR Trailing Stop (추세 추적)
- 일일 손실 한도 (Daily Loss Limit / Kill Switch)
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Position:
    """단일 선물 포지션 객체"""
    side: PositionSide
    entry_price: float
    size: float               # 계약 수량 (Base asset 수량, 예: BTC)
    sl_price: float           # 손절가
    tp1_price: Optional[float] = None  # 1차 익절가
    tp2_price: Optional[float] = None  # 2차 익절가
    is_half_closed: bool = False       # 50% 분할 익절 완료 여부
    highest_price: float = 0.0         # 진입 후 최고가 (롱 트레일링용)
    lowest_price: float = 0.0          # 진입 후 최저가 (숏 트레일링용)
    engine_name: str = ""              # "MEAN_REVERSION" or "TREND_FOLLOWING"
    entry_bar: int = 0
    entry_time: str = ""
    initial_risk_usdt: float = 0.0     # 진입 시 설정한 1회 최대 위험 금액

    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
        if self.lowest_price == 0.0:
            self.lowest_price = self.entry_price


class PositionManager:
    """리스크 관리 및 포지션 사이징 제어기"""

    def __init__(
        self,
        risk_per_trade_pct: float = 0.01,  # 1회 매매 최대 리스크 (1%)
        max_daily_loss_pct: float = 0.03,  # 일일 최대 손실률 (3% Kill Switch)
        default_leverage: float = 3.0,     # 기본 레버리지 (2~3x 권장)
        max_leverage: float = 5.0,
    ):
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.default_leverage = min(default_leverage, max_leverage)
        self.max_leverage = max_leverage

        self.daily_start_equity = 0.0
        self.current_day = ""
        self.is_kill_switch_active = False

    def update_day(self, current_day: str, equity: float):
        """날짜 변경 시 일일 손실 한도 리셋"""
        if current_day != self.current_day:
            self.current_day = current_day
            self.daily_start_equity = equity
            self.is_kill_switch_active = False

    def check_kill_switch(self, current_equity: float) -> bool:
        """일일 3% 초과 손실 발생 시 Kill Switch 발동"""
        if self.daily_start_equity <= 0:
            return False

        daily_drawdown = (self.daily_start_equity - current_equity) / self.daily_start_equity
        if daily_drawdown >= self.max_daily_loss_pct:
            self.is_kill_switch_active = True
            return True
        return self.is_kill_switch_active

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        sl_price: float,
        side: PositionSide,
        weight: float = 1.0,
    ) -> float:
        """
        1% Risk Model 기반 포지션 수량(BTC) 계산
        - 1회 감수할 최대 손실 = equity * risk_per_trade_pct * weight
        - 손절 폭(USDT per 1 BTC) = abs(entry_price - sl_price)
        - 진입 수량 = 최대 위험 금액 / 손절 폭
        - 레버리지 최대 증거금 한도 내로 클리핑
        """
        if entry_price <= 0 or sl_price <= 0 or equity <= 0:
            return 0.0

        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            return 0.0

        risk_budget = equity * self.risk_per_trade_pct * weight
        size = risk_budget / sl_distance

        # 레버리지 한도 체크: 총 포지션 가치(size * entry_price) <= equity * leverage
        max_position_value = equity * self.default_leverage
        max_size = max_position_value / entry_price
        final_size = min(size, max_size)

        return float(final_size)
