"""
RADE 국면 관리자 (Regime Manager)
- HMM 모델과 룰 기반 지표(ADX, CI)의 결합
- 히스테리시스(Hysteresis) 기반 채터링 방지 상태 전이
- 쿨다운(Cooldown) 관리 및 엔진별 포지션 비중(Weight) 산출
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from python.regime.hmm_detector import HMMRegimeDetector
from python.regime.rule_indicators import RuleRegimeCalculator


class RegimeState:
    RANGE = "RANGE"      # 횡보 국면 (엔진 1 가동)
    TREND = "TREND"      # 추세 국면 (엔진 2 가동)
    TRANSITION = "TRANSITION"  # 전이/불확실 국면


class RegimeManager:
    """국면 탐지, 히스테리시스, 전이 프로토콜 통합 관리자"""

    def __init__(
        self,
        hmm_window: int = 720,          # HMM 학습 윈도우 (봉 수, 720봉 = 약 30일)
        retrain_interval: int = 168,    # HMM 재학습 주기 (168봉 = 1주일)
        hysteresis_upper: float = 0.65, # 횡보 -> 추세 전환 임계값
        hysteresis_lower: float = 0.35, # 추세 -> 횡보 복귀 임계값
        cooldown_bars: int = 3,         # 국면 전환 후 신규 진입 대기 봉 수
    ):
        self.hmm_window = hmm_window
        self.retrain_interval = retrain_interval
        self.hysteresis_upper = hysteresis_upper
        self.hysteresis_lower = hysteresis_lower
        self.cooldown_bars = cooldown_bars

        self.hmm_detector = HMMRegimeDetector(n_components=2)
        self.rule_calculator = RuleRegimeCalculator()

        self.current_state = RegimeState.RANGE
        self.bars_since_transition = 999
        self.last_trained_idx = -999

    def fit_hmm(self, df_window: pd.DataFrame):
        """HMM 모델을 주어진 윈도우 데이터로 학습"""
        self.hmm_detector.fit(df_window)

    def calculate_regime_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터프레임 전체에 대해 시점별 국면 확률 및 상태를 시뮬레이션 계산 (Look-Ahead Bias 방지)
        - 학습 윈도우(hmm_window)가 누적된 이후부터 순차 계산
        """
        data = df.copy()
        n = len(data)

        hmm_probs = np.full(n, np.nan)
        rule_probs = self.rule_calculator.compute_rule_trend_probability(data).values
        combined_probs = np.full(n, np.nan)
        regime_states = [RegimeState.RANGE] * n
        is_cooldown = [False] * n

        curr_state = RegimeState.RANGE
        bars_since_trans = 999

        # 순차 롤링 시뮬레이션
        for i in range(self.hmm_window, n):
            # 주기적 HMM 재학습 (i 시점까지의 데이터만 사용)
            if (i - self.last_trained_idx) >= self.retrain_interval or self.last_trained_idx < 0:
                train_slice = data.iloc[i - self.hmm_window:i]
                try:
                    self.fit_hmm(train_slice)
                    self.last_trained_idx = i
                except Exception as e:
                    pass

            # i 시점의 HMM 추세 확률 (직전 슬라이스 기반)
            recent_slice = data.iloc[max(0, i - 100):i + 1]
            try:
                p_hmm = self.hmm_detector.get_latest_trend_probability(recent_slice)
            except Exception:
                p_hmm = 0.5
            hmm_probs[i] = p_hmm

            p_rule = rule_probs[i]

            # 결합 로직: HMM과 룰 지표 일치 여부 확인
            if abs(p_hmm - p_rule) < 0.25:
                p_comb = 0.6 * p_hmm + 0.4 * p_rule
            else:
                p_comb = 0.5  # 전이/불확실 구간
            combined_probs[i] = p_comb

            # 히스테리시스 상태 전이
            prev_state = curr_state
            if curr_state == RegimeState.RANGE:
                if p_comb >= self.hysteresis_upper:
                    curr_state = RegimeState.TREND
            elif curr_state == RegimeState.TREND:
                if p_comb <= self.hysteresis_lower:
                    curr_state = RegimeState.RANGE

            # 상태 전환 감지 및 쿨다운 카운터
            if curr_state != prev_state:
                bars_since_trans = 0
            else:
                bars_since_trans += 1

            regime_states[i] = curr_state
            is_cooldown[i] = (bars_since_trans < self.cooldown_bars)

        data['hmm_trend_prob'] = hmm_probs
        data['rule_trend_prob'] = rule_probs
        data['regime_trend_prob'] = combined_probs
        data['regime_state'] = regime_states
        data['is_cooldown'] = is_cooldown

        # 엔진별 비중 (Weight) 계산
        data['mean_revert_weight'] = (1.0 - data['regime_trend_prob']).clip(0.0, 1.0).fillna(0.5)
        data['trend_follow_weight'] = data['regime_trend_prob'].clip(0.0, 1.0).fillna(0.5)

        return data
