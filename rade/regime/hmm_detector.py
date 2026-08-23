import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from hmmlearn.hmm import GaussianHMM


class HMMRegimeDetector:
    """Gaussian HMM 기반 3-State(Range, Bull, Bear/Panic) 국면 확률 추정기 (exp16 기준 표준)"""

    def __init__(self, n_components: int = 3, covariance_type: str = "full", min_covar: float = 1e-3, random_state: int = 42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.random_state = random_state
        self.model: Optional[GaussianHMM] = None
        self.range_idx = 0
        self.bull_idx = 1
        self.bear_idx = 2

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def save_model(self, filepath: str):
        """학습된 HMM 모델 및 상태 인덱스를 파일로 영속화"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "model": self.model,
            "bull_idx": self.bull_idx,
            "range_idx": self.range_idx,
            "bear_idx": self.bear_idx,
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "min_covar": self.min_covar,
            "random_state": self.random_state,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_model(self, filepath: str) -> bool:
        """파일에서 HMM 모델 및 상태 인덱스 로드"""
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.bull_idx = data["bull_idx"]
            self.range_idx = data["range_idx"]
            self.bear_idx = data["bear_idx"]
            self.n_components = data.get("n_components", 3)
            self.covariance_type = data.get("covariance_type", "full")
            self.min_covar = data.get("min_covar", 1e-3)
            self.random_state = data.get("random_state", 42)
            return True
        except Exception:
            return False

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """관측치 피처 행렬 생성: [return, atr_ratio, vol_change]"""
        features = df[["return", "atr_ratio", "vol_change"]].copy()
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return features.values

    def fit(self, df: pd.DataFrame):
        """슬라이딩 윈도우 데이터로 HMM 학습 및 3개 상태 자동 정렬(State Alignment)"""
        X = self._prepare_features(df)
        if len(X) < 10:
            return

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

    def get_latest_probabilities(self, df_window: pd.DataFrame) -> Tuple[float, float, float]:
        """가장 최근(마지막) 봉의 3-State 확률 반환"""
        if self.model is None:
            return 0.33, 0.33, 0.33

        X = self._prepare_features(df_window)
        posteriors = self.model.predict_proba(X)
        last_p = posteriors[-1]
        return float(last_p[self.range_idx]), float(last_p[self.bull_idx]), float(last_p[self.bear_idx])
