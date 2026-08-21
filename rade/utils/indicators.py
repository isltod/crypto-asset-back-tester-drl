"""
기술적 지표 계산 모듈
RSI, MACD, Bollinger Bands, ATR, Choppiness Index, ADX 등 계산
"""
import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Wilder's Exponential Moving Average
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    """Bollinger Bands (Middle, Upper, Lower, Bandwidth, %B)"""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    bandwidth = (upper - lower) / (middle + 1e-10)
    percent_b = (series - lower) / (upper - lower + 1e-10)
    return middle, upper, lower, bandwidth, percent_b


def compute_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range (TR)"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (ATR) using Wilder's EMA"""
    tr = compute_true_range(high, low, close)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Choppiness Index (CI)
    CI = 100 * LOG10( SUM(ATR(1), n) / (MaxHigh(n) - MinLow(n)) ) / LOG10(n)
    CI > 61.8 : 횡보(Choppy) / CI < 38.2 : 추세(Trending)
    """
    tr1 = compute_true_range(high, low, close)
    sum_tr = tr1.rolling(window=period).sum()
    max_high = high.rolling(window=period).max()
    min_low = low.rolling(window=period).min()
    range_hl = max_high - min_low

    ci = 100.0 * np.log10((sum_tr + 1e-10) / (range_hl + 1e-10)) / np.log10(period)
    return ci


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """
    Average Directional Index (ADX) & +DI, -DI
    ADX < 20 : 횡보(Weak Trend) / ADX > 25 : 강한 추세(Strong Trend)
    """
    tr = compute_true_range(high, low, close)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_series = pd.Series(plus_dm, index=high.index)
    minus_dm_series = pd.Series(minus_dm, index=low.index)

    smooth_plus_dm = plus_dm_series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    smooth_minus_dm = minus_dm_series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * (smooth_plus_dm / (atr + 1e-10))
    minus_di = 100.0 * (smooth_minus_dm / (atr + 1e-10))

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return adx, plus_di, minus_di


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """모든 기본 및 국면 지표를 데이터프레임에 추가"""
    data = df.copy()
    
    # 수익률 및 변동성
    data['return'] = data['close'].pct_change()
    data['atr'] = compute_atr(data['high'], data['low'], data['close'], period=14)
    data['atr_ma50'] = data['atr'].rolling(window=50).mean()
    data['atr_ratio'] = data['atr'] / (data['close'] + 1e-10)
    
    # 거래량 변화율 (20봉 이동평균 대비)
    vol_ma20 = data['volume'].rolling(window=20).mean()
    data['vol_change'] = (data['volume'] - vol_ma20) / (vol_ma20 + 1e-10)

    # RSI
    data['rsi'] = compute_rsi(data['close'], period=14)

    # 볼린저 밴드
    bb_mid, bb_upper, bb_lower, bb_bw, bb_pct = compute_bollinger_bands(data['close'], period=20, num_std=2.0)
    data['bb_middle'] = bb_mid
    data['bb_upper'] = bb_upper
    data['bb_lower'] = bb_lower
    data['bb_bandwidth'] = bb_bw
    data['bb_bandwidth_ma50'] = bb_bw.rolling(window=50).mean()
    data['bb_percent_b'] = bb_pct

    # 장기 추세 필터 (200 EMA)
    data['ema200'] = data['close'].ewm(span=200, adjust=False).mean()

    # 국면 지표 (Choppiness Index, ADX)
    data['choppiness'] = compute_choppiness_index(data['high'], data['low'], data['close'], period=14)
    data['adx'], data['plus_di'], data['minus_di'] = compute_adx(data['high'], data['low'], data['close'], period=14)

    return data
