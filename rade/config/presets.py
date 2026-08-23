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
    trend_risk_pct: float              # 추세장 리스크 비율
    mr_risk_pct: float                 # 횡보장 리스크 비율
    trailing_atr_multiplier: float
    max_trailing_atr: float
    mean_revert_max_holding: int
    leverage: float
    expected_4yr_return: str
    expected_mdd: str
    expected_pf: str
    expected_win_rate: str
    expected_calmar: str


STRATEGY_PRESETS: Dict[str, StrategyConfig] = {
    # 1. 황금 최적화 공식 표준 베이스라인 (Golden Standard Baseline ⭐⭐⭐⭐⭐)
    "STANDARD_GOLDEN": StrategyConfig(
        preset_id="STANDARD_GOLDEN",
        name="황금 최적화 표준 베이스 모드 (공식 기본값 ⭐)",
        description="추세장 2.5% x 횡보장 4.0% 비대칭 리스크 배팅으로 MDD 14%대를 유지하며 4년 수익률 +356%를 달성하는 RADE 공식 표준 모델",
        hmm_window=720,
        retrain_interval=168,
        hmm_base_threshold=0.74,
        hmm_bear_threshold=0.74,
        bear_mode="CASH",              # 하락장 100% 현금 관망
        trend_risk_pct=0.025,          # 추세 2.5%
        mr_risk_pct=0.040,             # 횡보 4.0%
        trailing_atr_multiplier=4.5,
        max_trailing_atr=4.5,
        mean_revert_max_holding=24,
        leverage=3.0,
        expected_4yr_return="+356.74% (+$35,674)",
        expected_mdd="14.95% (철벽 방어)",
        expected_pf="1.91",
        expected_win_rate="54.5%",
        expected_calmar="23.86 (전구간 1위)"
    ),

    # 2. 초안전 지향 프로파일 (Ultra Safe Mode)
    "ULTRA_SAFE": StrategyConfig(
        preset_id="ULTRA_SAFE",
        name="초안전 방어 모드 (MDD 10% 미만 극단적 통제)",
        description="추세 1.5% x 횡보 1.5% 보수적 배팅으로 MDD를 9.40%로 묶어두면서 4년 +124%를 달성하는 초안전 지향 전략",
        hmm_window=720,
        retrain_interval=168,
        hmm_base_threshold=0.74,
        hmm_bear_threshold=0.74,
        bear_mode="CASH",
        trend_risk_pct=0.015,
        mr_risk_pct=0.015,
        trailing_atr_multiplier=4.5,
        max_trailing_atr=4.5,
        mean_revert_max_holding=24,
        leverage=3.0,
        expected_4yr_return="+124.56% (+$12,455)",
        expected_mdd="9.40% (한 자리 수 극단적 방어)",
        expected_pf="1.90",
        expected_win_rate="54.5%",
        expected_calmar="13.26"
    ),

    # 3. 수익 극대화 비대칭 숏 모드 (Aggressive Short Mode)
    "AGGRESSIVE_SHORT": StrategyConfig(
        preset_id="AGGRESSIVE_SHORT",
        name="수익 극대화 비대칭 숏 공격 모드",
        description="하락 확신도 80% 이상의 대폭락장에서 추세 숏을 때려 2022년 하락장(+6,778$)을 최고 수익 연도로 반전시키는 230% 수익 극대화 전략",
        hmm_window=720,
        retrain_interval=168,
        hmm_base_threshold=0.74,
        hmm_bear_threshold=0.80,       # 하락장만 80% 이상 초고확신 시 진입
        bear_mode="SHORT",             # 추세 숏 가동
        trend_risk_pct=0.020,
        mr_risk_pct=0.020,
        trailing_atr_multiplier=4.5,
        max_trailing_atr=4.5,
        mean_revert_max_holding=24,
        leverage=3.0,
        expected_4yr_return="+230.42% (+$23,042)",
        expected_mdd="26.27% (변동성 감수)",
        expected_pf="1.41",
        expected_win_rate="42.6%",
        expected_calmar="8.77"
    )
}

# 기본 별칭 지원 (CONSERVATIVE_CASH -> STANDARD_GOLDEN)
STRATEGY_PRESETS["CONSERVATIVE_CASH"] = STRATEGY_PRESETS["STANDARD_GOLDEN"]


def get_preset(preset_name: str = "STANDARD_GOLDEN") -> StrategyConfig:
    """프리셋 이름으로 StrategyConfig 반환 (기본값: STANDARD_GOLDEN)"""
    key = preset_name.upper()
    if key not in STRATEGY_PRESETS:
        available = ", ".join(STRATEGY_PRESETS.keys())
        raise ValueError(f"알 수 없는 프리셋: '{preset_name}'. 사용 가능한 프리셋: {available}")
    return STRATEGY_PRESETS[key]


def list_presets() -> Dict[str, str]:
    """사용 가능한 프리셋 목록 및 설명 반환"""
    return {k: f"{v.name} ({v.expected_4yr_return}, MDD {v.expected_mdd})" for k, v in STRATEGY_PRESETS.items()}
