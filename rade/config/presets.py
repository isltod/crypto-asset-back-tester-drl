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
    # 1. 황금 균형 공식 표준 베이스라인 (Golden Standard Baseline ⭐⭐⭐⭐⭐ - 칼마 비율 1위)
    "STANDARD_GOLDEN": StrategyConfig(
        preset_id="STANDARD_GOLDEN",
        name="황금 균형 표준 베이스 모드 (공식 기본값 ⭐)",
        description="추세장 1.0% x 횡보장 8.0% 비대칭 배팅으로 칼마 비율 1위(13.19)를 달성하며 MDD 14%대로 4년 +187%를 만들어내는 RADE 공식 표준 모델",
        hmm_window=720,
        retrain_interval=168,
        hmm_base_threshold=0.74,
        hmm_bear_threshold=0.74,
        bear_mode="CASH",              # 하락장 100% 현금 관망
        trend_risk_pct=0.010,          # 추세 1.0%
        mr_risk_pct=0.080,             # 횡보 8.0%
        trailing_atr_multiplier=4.5,
        max_trailing_atr=4.5,
        mean_revert_max_holding=24,
        leverage=3.0,
        expected_4yr_return="+187.06% (+$18,706)",
        expected_mdd="14.18% (안정적 14%대 방어)",
        expected_pf="1.88",
        expected_win_rate="52.20%",
        expected_calmar="13.19 (전구간 1위)"
    ),

    # 2. 안정 성장형 프로파일 (Moderate Safe Mode - MDD 13%대 철벽 방어)
    "MODERATE_SAFE": StrategyConfig(
        preset_id="MODERATE_SAFE",
        name="안정 성장 모드 (MDD 13%대 철벽 방어)",
        description="추세 1.5% x 횡보 4.0% 배팅으로 MDD를 13.95%로 묶어두면서 4년 +214% 및 2022년 1천불 흑자를 달성하는 안정 지향형 전략",
        hmm_window=720,
        retrain_interval=168,
        hmm_base_threshold=0.74,
        hmm_bear_threshold=0.74,
        bear_mode="CASH",
        trend_risk_pct=0.015,
        mr_risk_pct=0.040,
        trailing_atr_multiplier=4.5,
        max_trailing_atr=4.5,
        mean_revert_max_holding=24,
        leverage=3.0,
        expected_4yr_return="+214.09% (+$21,409)",
        expected_mdd="13.95% (13%대 철벽 방어)",
        expected_pf="1.95",
        expected_win_rate="53.95%",
        expected_calmar="15.34"
    ),

    # 3. 고수익 300% 성장 프로파일 (High Growth 300 Mode - MDD 19%대)
    "HIGH_GROWTH_300": StrategyConfig(
        preset_id="HIGH_GROWTH_300",
        name="고수익 300% 성장 모드 (MDD 19%대 고수익형)",
        description="추세 2.5% x 횡보 4.0% 배팅으로 4년 총수익 +303.45%(+$30,345)를 달성하는 300% 고수익 지향 전략",
        hmm_window=720,
        retrain_interval=168,
        hmm_base_threshold=0.74,
        hmm_bear_threshold=0.74,
        bear_mode="CASH",
        trend_risk_pct=0.025,
        mr_risk_pct=0.040,
        trailing_atr_multiplier=4.5,
        max_trailing_atr=4.5,
        mean_revert_max_holding=24,
        leverage=3.0,
        expected_4yr_return="+303.45% (+$30,345)",
        expected_mdd="19.07% (20% 미만 방어)",
        expected_pf="1.96",
        expected_win_rate="53.95%",
        expected_calmar="15.92"
    ),

    # 4. 초안전 지향 프로파일 (Ultra Safe Mode - MDD 10% 미만 극단적 통제)
    "ULTRA_SAFE": StrategyConfig(
        preset_id="ULTRA_SAFE",
        name="초안전 방어 모드 (MDD 10% 미만 극단적 통제)",
        description="추세 1.5% x 횡보 1.5% 보수적 배팅으로 MDD를 9.70%로 묶어두면서 4년 +118%를 달성하는 초안전 지향 전략",
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
        expected_4yr_return="+118.40% (+$11,840)",
        expected_mdd="9.70% (한 자리 수 극단적 방어)",
        expected_pf="1.90",
        expected_win_rate="54.5%",
        expected_calmar="12.20"
    ),

    # 5. 수익 극대화 비대칭 숏 모드 (Aggressive Short Mode)
    "AGGRESSIVE_SHORT": StrategyConfig(
        preset_id="AGGRESSIVE_SHORT",
        name="수익 극대화 비대칭 숏 공격 모드",
        description="하락 확신도 80% 이상의 대폭락장에서 추세 숏을 때려 2022년 하락장(+5,586$)을 최고 수익 연도로 반전시키는 213% 수익 극대화 전략",
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
        expected_4yr_return="+213.28% (+$21,328)",
        expected_mdd="26.27% (변동성 감수)",
        expected_pf="1.41",
        expected_win_rate="43.2%",
        expected_calmar="8.12"
    ),

    # 6. 작은 몬스터 실전 공격형 모드 (Monster Mini Mode - TF 4% x MR 12% x 80% 숏 💥)
    "MONSTER_MINI": StrategyConfig(
        preset_id="MONSTER_MINI",
        name="작은 몬스터 실전 공격형 (100배 칼마 1위 ⚡)",
        description="레버리지 100배 개방 + 추세 4% x 횡보 12% + 80% 숏으로 MDD를 68.08%로 방어하면서 4년 +529.4%(6.3배) 및 100배 칼마 1위(7.78)를 달성하는 실전 최강 공격 모델",
        hmm_window=720,
        retrain_interval=168,
        hmm_base_threshold=0.74,
        hmm_bear_threshold=0.80,       # 80% 비대칭 숏
        bear_mode="SHORT",             # 추세 숏 가동
        trend_risk_pct=0.040,          # 추세 4.0%
        mr_risk_pct=0.120,             # 횡보 12.0%
        trailing_atr_multiplier=4.5,
        max_trailing_atr=4.5,
        mean_revert_max_holding=24,
        leverage=100.0,                # 100배 레버리지 개방
        expected_4yr_return="+529.42% (+$52,942)",
        expected_mdd="68.08% (60%대 방어)",
        expected_pf="1.31",
        expected_win_rate="43.5%",
        expected_calmar="7.78 (100x 1위 ⭐)"
    ),

    # 7. 몬스터 극한 100배 레버리지 모드 (Monster Extreme 100x Mode - 절대 화력 1위 피크)
    "MONSTER_EXTREME_100X": StrategyConfig(
        preset_id="MONSTER_EXTREME_100X",
        name="몬스터 극한 100x 모드 (절대 화력 1위 피크 🚀)",
        description="레버리지 100배 무제한 개방 + 추세 4.5% x 횡보 16% + 80% 숏으로 4년 +584.3%(6.84배)의 수학적 절대 피크를 기록하는 이론상 절대 한계 모델",
        hmm_window=720,
        retrain_interval=168,
        hmm_base_threshold=0.74,
        hmm_bear_threshold=0.80,       # 80% 비대칭 숏
        bear_mode="SHORT",             # 추세 숏 가동
        trend_risk_pct=0.045,          # 추세 4.5% (피크)
        mr_risk_pct=0.160,             # 횡보 16.0% (피크)
        trailing_atr_multiplier=4.5,
        max_trailing_atr=4.5,
        mean_revert_max_holding=24,
        leverage=100.0,                # 100배 레버리지 개방
        expected_4yr_return="+584.28% (+$58,428)",
        expected_mdd="78.07% (초고위험 계좌 78% 손실)",
        expected_pf="1.28",
        expected_win_rate="42.7%",
        expected_calmar="7.48"
    )
}

# 기본 별칭 지원
STRATEGY_PRESETS["CONSERVATIVE_CASH"] = STRATEGY_PRESETS["STANDARD_GOLDEN"]
STRATEGY_PRESETS["MONSTER"] = STRATEGY_PRESETS["MONSTER_MINI"]
STRATEGY_PRESETS["MONSTER_100X"] = STRATEGY_PRESETS["MONSTER_EXTREME_100X"]


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

