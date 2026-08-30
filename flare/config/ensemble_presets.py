"""
flare/config/ensemble_presets.py
- RADE x FLARE 앙상블 포트폴리오 공식 전략 프리셋 (Ensemble Presets)
- 사용자가 검증된 앙상블 자산 배분 및 리밸런싱 프로파일을 단 한 줄로 전환하여 사용할 수 있도록 제공합니다.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class EnsembleConfig:
    preset_id: str
    name: str
    description: str
    rade_ratio: float              # RADE 표준 배분 비율 (예: 0.80)
    flare_ratio: float             # FLARE 5x 배분 비율 (예: 0.20)
    rebalance_period: str          # 리밸런싱 주기 (기본: "3M" 분기)
    rade_preset: str               # 결합할 RADE 프리셋 (기본: "STANDARD_GOLDEN")
    expected_4yr_return: str       # 4개년 기대 수익률
    expected_mdd: str              # 기대 최대 낙폭 (MDD)
    expected_calmar: str           # 기대 칼마 비율
    monthly_loss_prob: str         # 월간 적자 확률
    max_losing_streak: str         # 최대 연속 적자 기간


ENSEMBLE_PRESETS: Dict[str, EnsembleConfig] = {
    # 1. 8:2 앙상블 황금 균형 모드 (공식 기본값 ⭐⭐⭐⭐⭐ - 수익 5배 x 20%대 방어)
    "ENSEMBLE_82_GOLDEN": EnsembleConfig(
        preset_id="ENSEMBLE_82_GOLDEN",
        name="8:2 앙상블 황금 균형 모드 (공식 기본값 ⭐)",
        description="RADE 표준 80%(BTC HMM 방패) + FLARE 20%(4대 코인 5배수 창) 분기 리밸런싱으로 4년 자산 5.0배(+399.25%) 복리 성장과 20%대 초반 MDD를 달성하는 앙상블 공식 표준 모델",
        rade_ratio=0.80,
        flare_ratio=0.20,
        rebalance_period="3M",
        rade_preset="STANDARD_GOLDEN",
        expected_4yr_return="+399.25% (+$39,925 / 5.0배)",
        expected_mdd="23.13% (20%대 초반 방어)",
        expected_calmar="17.26",
        monthly_loss_prob="21.3% (10달 중 2.1달만 적자 🏆)",
        max_losing_streak="2개월 연속 (최장 2달 이내 반등)"
    ),

    # 2. 9:1 앙상블 안심 방패 모드 (Safe Shield Mode - MDD 17%대 x 칼마 1위 🛡️)
    "ENSEMBLE_91_SAFE": EnsembleConfig(
        preset_id="ENSEMBLE_91_SAFE",
        name="9:1 앙상블 안심 방패 모드 (칼마 1위 🛡️)",
        description="RADE 표준 90% + FLARE 10% 분기 리밸런싱으로 MDD를 17.13%로 극단적 통제하면서 4년 +329.81%(4.3배) 및 칼마 비율 전체 1위(19.26)를 달성하는 극강 안정성 모델",
        rade_ratio=0.90,
        flare_ratio=0.10,
        rebalance_period="3M",
        rade_preset="STANDARD_GOLDEN",
        expected_4yr_return="+329.81% (+$32,981 / 4.3배)",
        expected_mdd="17.13% (17%대 철벽 방어)",
        expected_calmar="19.26 (전구간 1위 🏆)",
        monthly_loss_prob="25.5%",
        max_losing_streak="2개월 연속"
    ),

    # 3. 7:3 앙상블 고수익 성장 모드 (High Growth Mode - 화력 극대화 🚀)
    "ENSEMBLE_73_GROWTH": EnsembleConfig(
        preset_id="ENSEMBLE_73_GROWTH",
        name="7:3 앙상블 고수익 성장 모드 (화력 극대화 🚀)",
        description="RADE 표준 70% + FLARE 30% 분기 리밸런싱으로 4년 자산 5.7배(+469.70%)를 달성하는 공격형 성장 지향 모델",
        rade_ratio=0.70,
        flare_ratio=0.30,
        rebalance_period="3M",
        rade_preset="STANDARD_GOLDEN",
        expected_4yr_return="+469.70% (+$46,970 / 5.7배)",
        expected_mdd="28.97%",
        expected_calmar="16.22",
        monthly_loss_prob="23.4%",
        max_losing_streak="2개월 연속"
    ),
}

# 기본 별칭 지원
ENSEMBLE_PRESETS["DEFAULT"] = ENSEMBLE_PRESETS["ENSEMBLE_82_GOLDEN"]
ENSEMBLE_PRESETS["GOLDEN"] = ENSEMBLE_PRESETS["ENSEMBLE_82_GOLDEN"]
ENSEMBLE_PRESETS["SAFE"] = ENSEMBLE_PRESETS["ENSEMBLE_91_SAFE"]
ENSEMBLE_PRESETS["GROWTH"] = ENSEMBLE_PRESETS["ENSEMBLE_73_GROWTH"]


def get_ensemble_preset(preset_name: str = "ENSEMBLE_82_GOLDEN") -> EnsembleConfig:
    """앙상블 프리셋 이름으로 EnsembleConfig 반환 (기본값: ENSEMBLE_82_GOLDEN)"""
    key = preset_name.upper()
    if key not in ENSEMBLE_PRESETS:
        available = ", ".join(ENSEMBLE_PRESETS.keys())
        raise ValueError(f"알 수 없는 앙상블 프리셋: '{preset_name}'. 사용 가능한 프리셋: {available}")
    return ENSEMBLE_PRESETS[key]


def list_ensemble_presets() -> Dict[str, str]:
    """사용 가능한 앙상블 프리셋 목록 및 설명 반환"""
    return {k: f"{v.name} ({v.expected_4yr_return}, MDD {v.expected_mdd})" for k, v in ENSEMBLE_PRESETS.items()}
