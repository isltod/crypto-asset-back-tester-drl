"""
룰 기반 국면 지표(ADX, Choppiness Index) 정규화 및 스코어 계산 모듈
"""
import numpy as np
import pandas as pd


def normalize_min_max(val: float or pd.Series, min_val: float, max_val: float) -> float or pd.Series:
    """값을 [min_val, max_val] 범위 기준으로 0.0 ~ 1.0으로 정규화 (클리핑 포함)"""
    if isinstance(val, pd.Series):
        norm = (val - min_val) / (max_val - min_val + 1e-10)
        return norm.clip(lower=0.0, upper=1.0)
    else:
        norm = (val - min_val) / (max_val - min_val + 1e-10)
        return float(np.clip(norm, 0.0, 1.0))


class RuleRegimeCalculator:
    """ADX 및 Choppiness Index 기반 룰 국면 스코어 계산기"""

    def __init__(self, ci_min: float = 30.0, ci_max: float = 70.0, adx_min: float = 10.0, adx_max: float = 40.0):
        self.ci_min = ci_min
        self.ci_max = ci_max
        self.adx_min = adx_min
        self.adx_max = adx_max

    def compute_rule_trend_probability(self, df: pd.DataFrame) -> pd.Series:
        """
        데이터프레임의 choppiness, adx 컬럼으로부터 룰 기반 추세 확률 (0.0~1.0) 계산
        - CI가 높을수록 횡보 -> ci_norm은 횡보 점수 (0: 추세, 1: 횡보)
        - ADX가 높을수록 추세 -> adx_norm은 횡보 점수 (1 - norm)
        - rule_trend_prob = 1.0 - rule_range_score
        """
        ci_norm = normalize_min_max(df['choppiness'], self.ci_min, self.ci_max)
        adx_trend_norm = normalize_min_max(df['adx'], self.adx_min, self.adx_max)
        adx_range_norm = 1.0 - adx_trend_norm

        # 횡보 가중합 스코어 (0.0: 강한 추세, 1.0: 강한 횡보)
        rule_range_score = 0.5 * ci_norm + 0.5 * adx_range_norm
        
        # 추세 확률로 변환 (0.0: 횡보, 1.0: 추세)
        rule_trend_prob = 1.0 - rule_range_score
        return rule_trend_prob
