"""
RADE 국면 관리자 (Regime Manager)
- HMM 모델과 룰 기반 지표(ADX, CI)의 결합
- 히스테리시스(Hysteresis) 기반 채터링 방지 상태 전이
- 쿨다운(Cooldown) 관리 및 엔진별 포지션 비중(Weight) 산출
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from rade.regime.hmm_detector import HMMRegimeDetector
from rade.regime.rule_indicators import RuleRegimeCalculator


class RegimeState:
    RANGE = "RANGE"              # 평온 횡보 국면 (평균회귀 가동)
    BULL_TREND = "BULL_TREND"    # 상승 추세 국면 (추세추종 롱 가동)
    BEAR_PANIC = "BEAR_PANIC"    # 위험/패닉 국면 (현금 관망 / Cash Mode)


class RegimeManager:
    """3-State HMM 국면 탐지, 캘린더 앵커링, 히스테리시스 전이 프로토콜 통합 관리자"""

    def __init__(
        self,
        hmm_window: int = 720,          # HMM 학습 윈도우 (720봉 = 약 30일)
        retrain_interval: int = 168,    # HMM 재학습 주기 (168봉 = 1주일)
        anchor_dayofweek: int = 6,      # 캘린더 앵커 요일 (6 = 일요일, 00:00 UTC = 09:00 KST)
        trans_threshold: float = 0.45,  # 상태 전환 최소 사후확률 임계값
        cooldown_bars: int = 0,         # 국면 전환 후 신규 진입 대기 봉 수 (0: 즉시 진입)
    ):
        self.hmm_window = hmm_window
        self.retrain_interval = retrain_interval
        self.anchor_dayofweek = anchor_dayofweek
        self.trans_threshold = trans_threshold
        self.cooldown_bars = cooldown_bars

        self.hmm_detector = HMMRegimeDetector(n_components=3, min_covar=1e-3)
        self.last_trained_idx = -999

    def fit_hmm(self, df_window: pd.DataFrame):
        """HMM 모델을 주어진 윈도우 데이터로 학습"""
        self.hmm_detector.fit(df_window)

    def calculate_regime_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [백테스트용] 캘린더 앵커(매주 일요일 00:00 UTC) 기준 HMM 재학습 + 매시간 최신 사후확률 추론
        """
        data = df.copy()
        n = len(data)
        dts = pd.to_datetime(data["datetime"], utc=True)

        p_ranges = np.full(n, np.nan)
        p_bulls = np.full(n, np.nan)
        p_bears = np.full(n, np.nan)
        regime_states = [RegimeState.RANGE] * n
        is_cooldown = [False] * n

        curr_state = RegimeState.RANGE
        bars_since_trans = 999

        for i in range(self.hmm_window, n):
            curr_dt = dts.iloc[i]
            is_anchor = (curr_dt.dayofweek == self.anchor_dayofweek and curr_dt.hour == 0)

            # 1. 캘린더 앵커 시점(매주 일요일 00:00 UTC)에만 HMM 모델 재학습
            if not self.hmm_detector.is_fitted or is_anchor:
                train_slice = data.iloc[i - self.hmm_window : i]
                try:
                    self.fit_hmm(train_slice)
                    self.last_trained_idx = i
                except Exception:
                    pass

            # 2. 매 1시간마다 최신 캔들 피처로 실시간 사후확률 추론
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
        data['regime_trend_prob'] = p_bulls
        data['regime_mr_prob'] = p_ranges

        return data

    def update_live_regime(self, df_recent: pd.DataFrame, model_path: Optional[str] = None) -> Tuple[Dict[str, Any], bool]:
        """
        [실전 페이퍼/라이브 트레이딩용]
        - model_path의 모델이 없으면 즉시 fit 및 저장
        - 최신 봉이 일요일 00:00 UTC (09:00 KST)이면 주간 정기 재학습 및 모델 파일 갱신
        - 매 정시에는 기존 모델로 최신 캔들에 대해 실시간 사후확률 추론(predict_proba)만 수행
        - 반환: (최신 국면 딕셔너리, 재학습 여부 bool)
        """
        n = len(df_recent)
        if n < self.hmm_window:
            raise ValueError(f"최근 데이터 봉 수가 부족합니다 ({n} < {self.hmm_window})")

        dts = pd.to_datetime(df_recent["datetime"], utc=True)
        latest_dt = dts.iloc[-1]
        is_anchor_time = (latest_dt.dayofweek == self.anchor_dayofweek and latest_dt.hour == 0)

        # 모델 로드 시도
        model_loaded = False
        if model_path and os.path.exists(model_path):
            model_loaded = self.hmm_detector.load_model(model_path)

        retrained = False
        # 1. 모델이 없거나, 주간 앵커 시점(일요일 자정)이면 재학습 수행
        if not model_loaded or not self.hmm_detector.is_fitted or is_anchor_time:
            train_slice = df_recent.iloc[-self.hmm_window - 1 : -1] # 방금 마감된 직전 720봉
            try:
                self.fit_hmm(train_slice)
                retrained = True
                if model_path:
                    self.hmm_detector.save_model(model_path)
            except Exception as e:
                # 피팅 실패 시 기존 로드된 모델 유지
                if not self.hmm_detector.is_fitted and model_path and os.path.exists(model_path):
                    self.hmm_detector.load_model(model_path)

        # 2. 최신 100봉으로 실시간 사후확률 추론
        recent_slice = df_recent.iloc[-100:]
        p_r, p_u, p_d = self.hmm_detector.get_latest_probabilities(recent_slice)

        probs = {
            RegimeState.RANGE: p_r,
            RegimeState.BULL_TREND: p_u,
            RegimeState.BEAR_PANIC: p_d,
        }
        max_state = max(probs, key=probs.get)
        curr_state = max_state if probs[max_state] >= self.trans_threshold else RegimeState.RANGE

        regime_info = {
            "regime_state": curr_state,
            "p_range": p_r,
            "p_bull": p_u,
            "p_bear": p_d,
            "is_anchor_time": is_anchor_time,
            "retrained": retrained,
        }
        return regime_info, retrained
