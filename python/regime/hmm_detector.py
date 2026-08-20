"""
HMM(은닉 마르코프 모델) 기반 국면 분류기
- 관측 벡터: [수익률(return), 변동성 비율(atr_ratio), 거래량 변화율(vol_change)]
- 2개 은닉 상태: 0 = 횡보(Range), 1 = 추세(Trend)
- Unsupervised 학습 후 변동성/분산 기반으로 상태 0, 1의 일관성 보장(State Alignment)
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class HMMRegimeDetector:
    """Gaussian HMM 기반 시장 국면 확률 추정기"""

    def __init__(self, n_components: int = 2, covariance_type: str = "full", random_state: int = 42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self.trend_state_idx = 1  # 추세 상태 인덱스 (기본 1)

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """관측치 피처 행렬 생성: [return, atr_ratio, vol_change]"""
        features = df[["return", "atr_ratio", "vol_change"]].copy()
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return features.values

    def fit(self, df: pd.DataFrame):
        """
        슬라이딩 윈도우 데이터로 HMM 학습 및 상태 정렬(Alignment)
        """
        X = self._prepare_features(df)
        
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=100,
            random_state=self.random_state,
        )
        self.model.fit(X)

        # 상태 정렬: 각 상태별 변동성(수익률 표준편차 또는 ATR) 비교
        # 변동성(또는 분산)이 더 큰 상태를 '추세(Trend, index 1)'로 지정
        # variances: [n_components, n_features]
        # features[1]이 atr_ratio, features[0]이 return
        state_volatilities = []
        for i in range(self.n_components):
            # covariance matrix의 대각원소(분산) 합 또는 atr_ratio 평균값
            mean_atr = self.model.means_[i, 1]
            state_volatilities.append(mean_atr)

        # ATR이 더 높은 상태를 Trend로 설정
        if state_volatilities[0] > state_volatilities[1]:
            self.trend_state_idx = 0
            self.range_state_idx = 1
        else:
            self.trend_state_idx = 1
            self.range_state_idx = 0

    def predict_trend_probability(self, df: pd.DataFrame) -> np.ndarray:
        """
        주어진 데이터 구간에 대해 각 시점의 추세 확률 P(Trend) 반환 (0.0 ~ 1.0)
        """
        if self.model is None:
            raise ValueError("HMM 모델이 아직 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        X = self._prepare_features(df)
        posteriors = self.model.predict_proba(X)
        trend_probs = posteriors[:, self.trend_state_idx]
        return trend_probs

    def get_latest_trend_probability(self, df_window: pd.DataFrame) -> float:
        """가장 최근(마지막) 봉의 추세 확률 반환"""
        probs = self.predict_trend_probability(df_window)
        return float(probs[-1])
