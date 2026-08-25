"""
flare.config.presets

FLARE 엔진의 검증된 듀얼 모드 공식 최적 시스템 프리셋(Preset) 정의 모듈
- Mode 2.1: FLARE-Swing-Pure (24시간 스윙 최적 설정)
- Mode 1.1: FLARE-Sniper-Pure (4시간 단기 저격 최적 설정)
- Mode 2.2: FLARE-Swing-Dynamic (동적 SL 조이기 하이브리드)
- Mode 1.2: FLARE-Sniper-ML (LightGBM 게이트 결합)
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass(frozen=True)
class FlarePresetConfig:
    preset_name: str
    description: str
    mode_type: str                  # "SWING" or "SNIPER"
    horizon_hours: int              # 만기 보유 시간 (시간 단위)
    horizon_bars: int               # 만기 봉 수 (5분봉 기준: hours * 12)
    sl_pct: float                   # 방어 손절선 (-%)
    tp_pct: Optional[float]         # 목표 익절선 (+%, None이면 만기 종가 청산)
    funding_rsi_threshold: float    # 펀딩비 30일 RSI 과열 기준선 (0.05 or 0.10)
    vol_ratio_threshold: float      # 거래량 배수 기준선 (스나이퍼용)
    wick_ratio_threshold: float     # 아래꼬리 비중 기준선 (스나이퍼용)
    recommended_leverage: float     # 권장 레버리지 배수
    fee_maker_pct: float = 0.02     # 지정가 수수료 (%)
    fee_taker_pct: float = 0.05     # 시장가 수수료 (%)
    slippage_pct: float = 0.02      # 슬리피지 (%)
    enable_ml_gate: bool = False    # ML 진입 게이트 활성화 여부
    ml_confidence_threshold: float = 0.45
    enable_dynamic_sl: bool = False # 매 4시간 동적 SL 조이기 활성화 여부
    target_symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"])


# =============================================================================
# 🏆 공식 확정 프리셋 정의
# =============================================================================

# 1. Mode 2.1: FLARE-Swing-Pure (최고 수익률 & 샤프 1.13 달성 👑)
SWING_PURE_PRESET = FlarePresetConfig(
    preset_name="swing_pure",
    description="8시간 펀딩비 하위 5% 과열 진입, SL -4.0% 방어선, 24시간 만기 종가 청산 (No TP)",
    mode_type="SWING",
    horizon_hours=24,
    horizon_bars=288,
    sl_pct=4.0,
    tp_pct=None, # 만기 종가 청산
    funding_rsi_threshold=0.05,
    vol_ratio_threshold=1.0,
    wick_ratio_threshold=0.0,
    recommended_leverage=3.0,
    enable_ml_gate=False,
    enable_dynamic_sl=False
)

# 2. Mode 1.1: FLARE-Sniper-Pure (극강의 MDD 9.68% 방어 & 승률 57.4% 🛡️)
SNIPER_PURE_PRESET = FlarePresetConfig(
    preset_name="sniper_pure",
    description="펀딩비 하위 10% + 5분봉 거래량 3x 청산 꼬리 진입, SL -3.0% 방어선, 4시간 만기 종가 청산",
    mode_type="SNIPER",
    horizon_hours=4,
    horizon_bars=48,
    sl_pct=3.0,
    tp_pct=None, # 4시간 만기 종가 청산
    funding_rsi_threshold=0.10,
    vol_ratio_threshold=3.0,
    wick_ratio_threshold=0.55,
    recommended_leverage=4.0,
    enable_ml_gate=False,
    enable_dynamic_sl=False
)

# 3. Mode 2.2: FLARE-Swing-Dynamic (매 4시간 ML 점검 동적 SL 조이기 🛡️)
SWING_DYNAMIC_PRESET = FlarePresetConfig(
    preset_name="swing_dynamic",
    description="8시간 펀딩비 하위 5% 진입 + 매 4시간마다 ML 점검 후 하방 위험 시 SL을 본전/타이트하게 조임",
    mode_type="SWING",
    horizon_hours=24,
    horizon_bars=288,
    sl_pct=4.0,
    tp_pct=None,
    funding_rsi_threshold=0.05,
    vol_ratio_threshold=1.0,
    wick_ratio_threshold=0.0,
    recommended_leverage=3.0,
    enable_ml_gate=False,
    enable_dynamic_sl=True
)

# 4. Mode 1.2: FLARE-Sniper-ML (LightGBM 게이트 결합 정예 저격)
SNIPER_ML_PRESET = FlarePresetConfig(
    preset_name="sniper_ml",
    description="펀딩비 하위 10% + 청산 꼬리 + LightGBM 4h 롱 확신도(Prob >= 45%) 통과 시에만 진입",
    mode_type="SNIPER",
    horizon_hours=4,
    horizon_bars=48,
    sl_pct=3.0,
    tp_pct=None,
    funding_rsi_threshold=0.10,
    vol_ratio_threshold=3.0,
    wick_ratio_threshold=0.55,
    recommended_leverage=5.0,
    enable_ml_gate=True,
    ml_confidence_threshold=0.45,
    enable_dynamic_sl=False
)


_PRESETS = {
    "swing_pure": SWING_PURE_PRESET,
    "swing": SWING_PURE_PRESET, # Alias
    "sniper_pure": SNIPER_PURE_PRESET,
    "sniper": SNIPER_PURE_PRESET, # Alias
    "swing_dynamic": SWING_DYNAMIC_PRESET,
    "sniper_ml": SNIPER_ML_PRESET
}


def get_preset(name: str = "swing_pure") -> FlarePresetConfig:
    """이름으로 공식 프리셋 설정을 조회합니다."""
    key = name.lower().strip()
    if key not in _PRESETS:
        raise ValueError(f"알 수 없는 프리셋 이름입니다: '{name}'. 가능한 목록: {list(_PRESETS.keys())}")
    return _PRESETS[key]


def list_presets() -> dict[str, str]:
    """사용 가능한 프리셋 목록과 설명을 반환합니다."""
    return {k: v.description for k, v in _PRESETS.items() if "_" in k}
