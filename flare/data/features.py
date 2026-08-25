"""
flare.data.features

5분봉 OHLCV 캔들 데이터와 8시간 펀딩비 데이터를 결합하여,
FLARE 듀얼 모드(Swing & Sniper) 및 LightGBM 모델 학습을 위한
수급, 미시구조(꼬리/거래량), 변동성, 모멘텀, 시간 주기 피처를 통합 생성하는 모듈
"""

import numpy as np
import pandas as pd


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """캔들 구조(몸통, 아래꼬리, 윗꼬리) 및 거래량 수급 피처를 계산합니다."""
    # 캔들 기본 구조
    body = (df["close"] - df["open"]).abs()
    total_range = df["high"] - df["low"]
    safe_range = np.where(total_range == 0, 1e-9, total_range)
    safe_body = np.where(body == 0, 1e-9, body)
    
    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    upper_wick = df["high"] - np.maximum(df["open"], df["close"])
    
    df["feat_body_ratio"] = body / safe_range
    df["feat_lower_wick_ratio"] = lower_wick / safe_range
    df["feat_upper_wick_ratio"] = upper_wick / safe_range
    df["feat_lower_wick_to_body"] = np.clip(lower_wick / safe_body, 0, 20.0)
    df["feat_upper_wick_to_body"] = np.clip(upper_wick / safe_body, 0, 20.0)
    
    # 거래량 24시간(288개 5분봉) 이동평균 및 Z-Score
    vol_sma288 = df["volume"].rolling(window=288, min_periods=72).mean()
    vol_std288 = df["volume"].rolling(window=288, min_periods=72).std()
    safe_vol_std = np.where(vol_std288 == 0, 1e-9, vol_std288)
    
    df["feat_vol_ratio_24h"] = df["volume"] / np.where(vol_sma288 == 0, 1e-9, vol_sma288)
    df["feat_vol_zscore_24h"] = np.clip((df["volume"] - vol_sma288) / safe_vol_std, -5.0, 15.0)
    
    # 롱/숏 청산 빔 프록시 플래그 (지표 피처)
    df["feat_is_lower_wick_spike"] = (
        (df["feat_vol_ratio_24h"] >= 3.0) & 
        (df["feat_lower_wick_ratio"] >= 0.55) & 
        (df["feat_lower_wick_to_body"] >= 1.5)
    ).astype(float)
    
    df["feat_is_upper_wick_spike"] = (
        (df["feat_vol_ratio_24h"] >= 3.0) & 
        (df["feat_upper_wick_ratio"] >= 0.55) & 
        (df["feat_upper_wick_to_body"] >= 1.5)
    ).astype(float)
    
    return df


def compute_funding_features(df: pd.DataFrame) -> pd.DataFrame:
    """펀딩비 수급 불균형 및 롤링 백분위/Z-Score 피처를 계산합니다."""
    # 펀딩비가 없는 경우 기본값 처리
    if "fundingRate" not in df.columns:
        df["fundingRate"] = 0.0001
        
    fr = df["fundingRate"]
    df["feat_funding_rate"] = fr * 100.0  # % 단위
    
    # 최근 30일(30 * 288 = 8,640개 5분봉) 롤링 펀딩비 상대강도/위치 (0.0 ~ 1.0)
    fr_min_30d = fr.rolling(window=8640, min_periods=288).min()
    fr_max_30d = fr.rolling(window=8640, min_periods=288).max()
    fr_range = np.where(fr_max_30d == fr_min_30d, 1e-9, fr_max_30d - fr_min_30d)
    
    df["feat_funding_rsi_30d"] = np.clip((fr - fr_min_30d) / fr_range, 0.0, 1.0)
    
    # 펀딩비 변동 표준화 (CV / Z-Score 성격)
    fr_mean_30d = fr.rolling(window=8640, min_periods=288).mean()
    fr_std_30d = fr.rolling(window=8640, min_periods=288).std()
    safe_fr_std = np.where(fr_std_30d == 0, 1e-9, fr_std_30d)
    df["feat_funding_cv_30d"] = np.clip((fr - fr_mean_30d) / safe_fr_std, -5.0, 5.0)
    
    # 펀딩비 과열 상태 플래그
    df["feat_is_funding_negative"] = (fr <= 0.0).astype(float)
    df["feat_is_funding_high"] = (fr >= 0.0003).astype(float)  # +0.03% 이상
    
    return df


def compute_volatility_and_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """변동성(ATR, Parkinson) 및 모멘텀(수익률, RSI, EMA 이격도) 피처를 계산합니다."""
    close = df["close"]
    
    # 1. 과거 N개 봉 수익률 (5m, 15m, 1h, 4h)
    df["feat_ret_1bar"] = (close - close.shift(1)) / close.shift(1) * 100.0
    df["feat_ret_3bar"] = (close - close.shift(3)) / close.shift(3) * 100.0
    df["feat_ret_12bar"] = (close - close.shift(12)) / close.shift(12) * 100.0
    df["feat_ret_48bar"] = (close - close.shift(48)) / close.shift(48) * 100.0
    
    # 2. ATR(14) 정규화
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - close.shift(1)).abs()
    tr3 = (df["low"] - close.shift(1)).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = tr.rolling(window=14, min_periods=14).mean()
    df["feat_atr_norm"] = (atr14 / close) * 100.0
    
    # 3. Parkinson 변동성 (2시간 = 24개 봉)
    log_hl = np.log(df["high"] / np.where(df["low"] == 0, 1e-9, df["low"])) ** 2
    parkinson = np.sqrt(log_hl.rolling(window=24, min_periods=24).mean() / (4.0 * np.log(2.0))) * 100.0
    df["feat_parkinson_vol_2h"] = parkinson
    
    # 4. RSI(14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=14).mean()
    rs = gain / np.where(loss == 0, 1e-9, loss)
    df["feat_rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    
    # 5. 4시간 EMA(48봉) 이격도
    ema48 = close.ewm(span=48, adjust=False).mean()
    df["feat_ema48_dist_norm"] = (close - ema48) / np.where(atr14 == 0, 1e-9, atr14)
    
    return df


def compute_time_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """시간대(Hour of day) 삼각함수 인코딩 및 글로벌 거래 세션 피처를 계산합니다."""
    if "datetime" not in df.columns:
        return df
        
    hour = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0
    
    # 24시간 주기 삼각함수 인코딩
    df["feat_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["feat_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    
    # 글로벌 거래 세션 원핫 인코딩 (UTC 기준)
    # 아시아 세션 (00:00 ~ 08:00 UTC)
    df["feat_session_asia"] = ((hour >= 0.0) & (hour < 8.0)).astype(float)
    # 런던 세션 (08:00 ~ 16:00 UTC)
    df["feat_session_london"] = ((hour >= 8.0) & (hour < 16.0)).astype(float)
    # 뉴욕 세션 (13:00 ~ 21:00 UTC)
    df["feat_session_ny"] = ((hour >= 13.0) & (hour < 21.0)).astype(float)
    
    return df


def generate_all_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """모든 피처를 파이프라인으로 생성하고 유효한 피처 컬럼 목록을 반환합니다."""
    df = compute_microstructure_features(df)
    df = compute_funding_features(df)
    df = compute_volatility_and_momentum_features(df)
    df = compute_time_cyclical_features(df)
    
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    return df, feature_cols
