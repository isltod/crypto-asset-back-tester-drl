"""
RADE 국면 관리자 (Regime Manager)
- HMM 모델과 룰 기반 지표(ADX, CI)의 결합
- 히스테리시스(Hysteresis) 기반 채터링 방지 상태 전이
- 쿨다운(Cooldown) 관리 및 엔진별 포지션 비중(Weight) 산출
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from rade.regime.hmm_detector import HMMRegimeDetector
from rade.regime.rule_indicators import RuleRegimeCalculator


class RegimeState:
    RANGE = "RANGE"              # 평온 횡보 국면 (평균회귀 가동)
    BULL_TREND = "BULL_TREND"    # 상승 추세 국면 (추세추종 롱 가동)
    BEAR_PANIC = "BEAR_PANIC"    # 위험/패닉 국면 (현금 관망 / Cash Mode)


class RegimeManager:
    """3-State HMM 국면 탐지, 히스테리시스, 전이 프로토콜 통합 관리자"""

    def __init__(
        self,
        hmm_window: int = 720,          # HMM 학습 윈도우 (720봉 = 약 30일)
        retrain_interval: int = 168,    # HMM 재학습 주기 (168봉 = 1주일)
        trans_threshold: float = 0.45,  # 상태 전환 최소 사후확률 임계값
        cooldown_bars: int = 3,         # 국면 전환 후 신규 진입 대기 봉 수
    ):
        self.hmm_window = hmm_window
        self.retrain_interval = retrain_interval
        self.trans_threshold = trans_threshold
        self.cooldown_bars = cooldown_bars

        self.hmm_detector = HMMRegimeDetector(n_components=3)
        self.last_trained_idx = -999

    def fit_hmm(self, df_window: pd.DataFrame):
        """HMM 모델을 주어진 윈도우 데이터로 학습"""
        self.hmm_detector.fit(df_window)

    def calculate_regime_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터프레임 전체에 대해 시점별 3-State 국면 확률 및 상태를 롤링 시뮬레이션 계산 (Look-Ahead Bias 방지)
        """
        data = df.copy()
        n = len(data)

        p_ranges = np.full(n, np.nan)
        p_bulls = np.full(n, np.nan)
        p_bears = np.full(n, np.nan)
        regime_states = [RegimeState.RANGE] * n
        is_cooldown = [False] * n

        curr_state = RegimeState.RANGE
        bars_since_trans = 999

        for i in range(self.hmm_window, n):
            # 주기적 HMM 재학습 (i 시점까지의 과거 데이터만 사용)
            if (i - self.last_trained_idx) >= self.retrain_interval or self.last_trained_idx < 0:
                train_slice = data.iloc[i - self.hmm_window : i]
                try:
                    self.fit_hmm(train_slice)
                    self.last_trained_idx = i
                except Exception:
                    pass

            recent_slice = data.iloc[max(0, i - 100) : i + 1]
            try:
                p_r, p_u, p_d = self.hmm_detector.get_latest_probabilities(recent_slice)
            except Exception:
                p_r, p_u, p_d = 0.34, 0.33, 0.33

            p_ranges[i] = p_r
            p_bulls[i] = p_u
            p_bears[i] = p_d

            # 3-State 히스테리시스 전이 로직
            prev_state = curr_state
            probs = {
                RegimeState.RANGE: p_r,
                RegimeState.BULL_TREND: p_u,
                RegimeState.BEAR_PANIC: p_d,
            }
            max_state = max(probs, key=probs.get)

            # 지배적 확률이 임계값(0.45) 이상일 때만 전환 허용 (채터링 방지)
            if probs[max_state] >= self.trans_threshold:
                curr_state = max_state

            if curr_state != prev_state:
                bars_since_trans = 0
            else:
                bars_since_trans += 1

            regime_states[i] = curr_state
            is_cooldown[i] = (bars_since_trans < self.cooldown_bars)

        data['p_range'] = p_ranges
        data['p_bull'] = p_bulls
        data['p_bear'] = p_bears
        data['regime_state'] = regime_states
        data['is_cooldown'] = is_cooldown

        return data
