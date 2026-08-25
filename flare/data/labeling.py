"""
flare.data.labeling

향후 Horizon(예: 4시간=48봉, 24시간=288봉) 동안의
최대 상방 진폭(MFE)과 최대 하방 진폭(MAE)을 계산하여,
통계적 손익비 우위가 있는 3-Class 비대칭 타깃 라벨을 생성하는 모듈
"""

import numpy as np
import pandas as pd


def compute_forward_mfe_mae(
    df: pd.DataFrame, 
    horizon_bars: int = 48
) -> pd.DataFrame:
    """
    현재 시점 이후 향후 horizon_bars 동안의 MFE(상방), MAE(하방)를 계산합니다.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    # 롤링 미래 최고가(High) 및 최저가(Low) 역방향 롤링
    rolling_max_high = high.iloc[::-1].rolling(window=horizon_bars, min_periods=horizon_bars).max().iloc[::-1]
    rolling_min_low = low.iloc[::-1].rolling(window=horizon_bars, min_periods=horizon_bars).min().iloc[::-1]
    
    # MFE / MAE (% 단위)
    df[f"target_mfe_{horizon_bars}"] = (rolling_max_high - close) / close * 100.0
    df[f"target_mae_{horizon_bars}"] = (close - rolling_min_low) / close * 100.0
    
    # 만기 시점 종가 수익률 (% 단위)
    future_close = close.shift(-horizon_bars)
    df[f"target_ret_{horizon_bars}"] = (future_close - close) / close * 100.0
    
    return df


def create_asymmetric_labels(
    df: pd.DataFrame,
    horizon_bars: int = 48,
    min_mfe_pct: float = 1.0,
    ratio_threshold: float = 1.3
) -> tuple[pd.DataFrame, str]:
    """
    MFE / MAE의 비대칭 비율을 기반으로 3-Class 라벨을 생성합니다.
    
    라벨 정의:
      - 1 (Long / Bullish Asym): MFE >= min_mfe_pct AND MFE >= ratio_threshold * MAE
      - 2 (Short / Bearish Asym): MAE >= min_mfe_pct AND MAE >= ratio_threshold * MFE
      - 0 (Neutral / Noise): 대칭적 횡보 또는 최소 진폭 미달 구간
    """
    df = compute_forward_mfe_mae(df, horizon_bars=horizon_bars)
    
    mfe = df[f"target_mfe_{horizon_bars}"]
    mae = df[f"target_mae_{horizon_bars}"]
    
    label_col = f"label_asym_{horizon_bars}"
    
    # 기본값 0 (Neutral)
    df[label_col] = 0
    
    # 🟢 Long (Class 1)
    cond_long = (mfe >= min_mfe_pct) & (mfe >= ratio_threshold * mae)
    # 🔴 Short (Class 2)
    cond_short = (mae >= min_mfe_pct) & (mae >= ratio_threshold * mfe)
    
    df.loc[cond_long, label_col] = 1
    df.loc[cond_short, label_col] = 2
    
    return df, label_col
