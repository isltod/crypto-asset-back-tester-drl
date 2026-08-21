"""
HMM(은닉 마르코프 모델) 기반 국면 분류기
- 관측 벡터: [수익률(return), 변동성 비율(atr_ratio), 거래량 변화율(vol_change)]
- 2개 은닉 상태: 0 = 횡보(Range), 1 = 추세(Trend)
- Unsupervised 학습 후 변동성/분산 기반으로 상태 0, 1의 일관성 보장(State Alignment)
"""
import numpy as np
import pandas as pd
from typing import Tuple
from hmmlearn.hmm import GaussianHMM


class HMMRegimeDetector:
    """Gaussian HMM 기반 3-State(Range, Bull, Bear/Panic) 국면 확률 추정기"""

    def __init__(self, n_components: int = 3, covariance_type: str = "full", random_state: int = 42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self.range_idx = 0
        self.bull_idx = 1
        self.bear_idx = 2

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """관측치 피처 행렬 생성: [return, atr_ratio, vol_change]"""
        features = df[["return", "atr_ratio", "vol_change"]].copy()
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return features.values

    def fit(self, df: pd.DataFrame):
        """슬라이딩 윈도우 데이터로 HMM 학습 및 3개 상태 자동 정렬(State Alignment)"""
        X = self._prepare_features(df)
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=100,
            random_state=self.random_state,
        )
        self.model.fit(X)

        mean_returns = self.model.means_[:, 0]
        mean_atrs = self.model.means_[:, 1]

        # 1) 평균 수익률이 가장 높은 상태 -> BULL (상승)
        bull_candidate = int(np.argmax(mean_returns))

        # 2) 나머지 2개 중 변동성(ATR)이 더 낮은 상태 -> RANGE (횡보), 높은 상태 -> BEAR/PANIC
        remaining = [i for i in range(self.n_components) if i != bull_candidate]
        if mean_atrs[remaining[0]] < mean_atrs[remaining[1]]:
            range_candidate = remaining[0]
            bear_candidate = remaining[1]
        else:
            range_candidate = remaining[1]
            bear_candidate = remaining[0]

        self.bull_idx = bull_candidate
        self.range_idx = range_candidate
        self.bear_idx = bear_candidate

    def predict_state_probabilities(self, df: pd.DataFrame) -> np.ndarray:
        """각 시점의 (P(Range), P(Bull), P(Bear)) 확률 반환"""
        if self.model is None:
            raise ValueError("HMM 모델이 아직 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        X = self._prepare_features(df)
        posteriors = self.model.predict_proba(X)
        p_range = posteriors[:, self.range_idx]
        p_bull = posteriors[:, self.bull_idx]
        p_bear = posteriors[:, self.bear_idx]
        return np.column_stack([p_range, p_bull, p_bear])

    def get_latest_probabilities(self, df_window: pd.DataFrame) -> Tuple[float, float, float]:
        """가장 최근(마지막) 봉의 3-State 확률 반환"""
        probs = self.predict_state_probabilities(df_window)
        last_p = probs[-1]
        return float(last_p[0]), float(last_p[1]), float(last_p[2])
