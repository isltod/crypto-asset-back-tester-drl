"""
RADE 시스템 공식 전략 프리셋 (Strategy Presets)
- 사용자가 검증된 전략 프로파일을 단 한 줄로 전환하여 사용할 수 있도록 제공합니다.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class StrategyConfig:
    preset_id: str
    name: str
    description: str
    hmm_window: int
    retrain_interval: int
    hmm_base_threshold: float
    hmm_bear_threshold: float
    bear_mode: str                     # "CASH" 또는 "SHORT"
    trailing_atr_multiplier: float
    max_trailing_atr: float
    mean_revert_max_holding: int
    risk_per_trade_pct: float
    leverage: float
    expected_4yr_return: str
    expected_mdd: str
    expected_pf: str
    expected_win_rate: str


STRATEGY_PRESETS: Dict[str, StrategyConfig] = {
    # 1. 기관급 안전 최우선 모델 (기본 권장 프로파일)
    "CONSERVATIVE_CASH": StrategyConfig(
        preset_id="CONSERVATIVE_CASH",
        name="안전 최우선 현금관망 모드 (기본 권장)",
        description="하락장(BEAR)에서 100% 현금 관망하여 MDD를 12%대로 철벽 방어하는 헤지펀드 스타일 안정형 전략",
        hmm_window=720,
        retrain_interval=168,
        hmm_base_threshold=0.74,
        hmm_bear_threshold=0.74,
        bear_mode="CASH",
        trailing_atr_multiplier=4.5,
        max_trailing_atr=4.5,
        mean_revert_max_holding=24,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        expected_4yr_return="+174.47% (+$17,446)",
        expected_mdd="12.76% (철벽 방어)",
        expected_pf="1.89",
        expected_win_rate="54.5%"
    ),

    # 2. 수익 극대화 비대칭 숏 모드 (공격형 프로파일)
    "AGGRESSIVE_SHORT": StrategyConfig(
        preset_id="AGGRESSIVE_SHORT",
        name="수익 극대화 비대칭 숏 공격 모드",
        description="하락 확신도 80% 이상의 대폭락장에서 추세 숏을 때려 2022년 하락장(+6,778$)을 최고 수익 연도로 반전시키는 230% 수익 극대화 전략",
        hmm_window=720,
        retrain_interval=168,
        hmm_base_threshold=0.74,
        hmm_bear_threshold=0.80,       # 하락장만 80% 이상 초고확신 시 진입
        bear_mode="SHORT",             # 추세 숏 가동
        trailing_atr_multiplier=4.5,
        max_trailing_atr=4.5,
        mean_revert_max_holding=24,
        risk_per_trade_pct=0.02,
        leverage=3.0,
        expected_4yr_return="+230.42% (+$23,042)",
        expected_mdd="26.27% (변동성 감수)",
        expected_pf="1.41",
        expected_win_rate="42.6%"
    )
}


def get_preset(preset_name: str = "CONSERVATIVE_CASH") -> StrategyConfig:
    """프리셋 이름으로 StrategyConfig 반환 (기본값: CONSERVATIVE_CASH)"""
    key = preset_name.upper()
    if key not in STRATEGY_PRESETS:
        available = ", ".join(STRATEGY_PRESETS.keys())
        raise ValueError(f"알 수 없는 프리셋: '{preset_name}'. 사용 가능한 프리셋: {available}")
    return STRATEGY_PRESETS[key]


def list_presets() -> Dict[str, str]:
    """사용 가능한 프리셋 목록 및 설명 반환"""
    return {k: f"{v.name} ({v.expected_4yr_return}, MDD {v.expected_mdd})" for k, v in STRATEGY_PRESETS.items()}
