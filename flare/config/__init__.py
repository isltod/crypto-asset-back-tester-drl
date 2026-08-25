"""
flare.config

FLARE 시스템 설정 및 공식 프리셋 패키지
"""

from flare.config.presets import (
    FlarePresetConfig,
    SWING_PURE_PRESET,
    SNIPER_PURE_PRESET,
    SWING_DYNAMIC_PRESET,
    SNIPER_ML_PRESET,
    get_preset,
    list_presets
)

__all__ = [
    "FlarePresetConfig",
    "SWING_PURE_PRESET",
    "SNIPER_PURE_PRESET",
    "SWING_DYNAMIC_PRESET",
    "SNIPER_ML_PRESET",
    "get_preset",
    "list_presets"
]
