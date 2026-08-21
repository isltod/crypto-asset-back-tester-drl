"""
바이낸스 선물 OHLCV 과거 데이터 수집 모듈
- Binance Futures Public REST API 활용 (API 키 없이 과거 데이터 수집 가능)
- 1H 봉 페이지네이션 수집 및 CSV 캐싱
"""
import os
import time
from datetime import datetime, timezone
import pandas as pd
import requests
from tqdm import tqdm


class BinanceFuturesFetcher:
    """바이낸스 선물 과거 klines(OHLCV) 다운로더"""

    BASE_URL = "https://fapi.binance.com/fapi/v1/klines"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_recent_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 800,
    ) -> pd.DataFrame:
        """
        바이낸스 선물에서 가장 최근 N개(최대 1500) klines를 단일 REST 요청으로 즉시 반환 (페이퍼/실시간 전용)
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1500),
        }
        res = requests.get(self.BASE_URL, params=params, timeout=10)
        res.raise_for_status()
        raw_data = res.json()

        if not raw_data:
            return pd.DataFrame()

        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ]
        df = pd.DataFrame(raw_data, columns=columns)
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        for col in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
            df[col] = df[col].astype(float)

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_time_str: str = "2023-01-01 00:00:00",
        end_time_str: str = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        """
        start_time부터 end_time까지의 klines를 반복 요청하여 온전한 DataFrame으로 반환
        """
        # UTC 타임스탬프 (ms) 변환
        dt_start = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        start_ms = int(dt_start.timestamp() * 1000)

        if end_time_str:
            dt_end = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            end_ms = int(dt_end.timestamp() * 1000)
        else:
            end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        all_klines = []
        curr_start = start_ms

        print(f"[{symbol}] {interval} 데이터 수집 시작 ({start_time_str} ~ {end_time_str or '현재'})...")

        with tqdm(desc=f"Fetching {symbol}") as pbar:
            while curr_start < end_ms:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": curr_start,
                    "endTime": end_ms,
                    "limit": limit,
                }
                
                try:
                    res = requests.get(self.BASE_URL, params=params, timeout=10)
                    res.raise_for_status()
                    data = res.json()
                except Exception as e:
                    print(f"\n[Warning] API 요청 에러: {e}. 3초 후 재시도...")
                    time.sleep(3)
                    continue

                if not data:
                    break

                all_klines.extend(data)
                pbar.update(len(data))

                # 다음 청크 시작 시간은 마지막 캔들의 openTime + 1ms
                last_open_time = data[-1][0]
                curr_start = last_open_time + 1

                if len(data) < limit:
                    break

                # Rate limit 방지
                time.sleep(0.1)

        if not all_klines:
            print(f"[Error] 수집된 데이터가 없습니다.")
            return pd.DataFrame()

        # DataFrame 가공
        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ]
        df = pd.DataFrame(all_klines, columns=columns)
        
        # 중복 제거 및 타입 변환
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        for col in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
            df[col] = df[col].astype(float)

        # Datetime 인덱스 추가 (가독성 및 분석용)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        print(f"총 {len(df)}개 봉 수집 완료 ({df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]})")
        return df

    def get_or_download_data(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_time_str: str = "2023-01-01 00:00:00",
        end_time_str: str = None,
        force_download: bool = False,
    ) -> pd.DataFrame:
        """로컬 파일이 있으면 로드, 없거나 force_download=True이면 새로 다운로드"""
        filename = f"{symbol}_{interval}.csv"
        filepath = os.path.join(self.data_dir, filename)

        if os.path.exists(filepath) and not force_download:
            print(f"[Cache] 로컬 캐시 데이터 로드: {filepath}")
            df = pd.read_csv(filepath)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            return df

        df = self.fetch_klines(symbol, interval, start_time_str, end_time_str)
        if not df.empty:
            df.to_csv(filepath, index=False)
            print(f"[Saved] 로컬 저장 완료: {filepath}")
        return df


if __name__ == "__main__":
    fetcher = BinanceFuturesFetcher(data_dir="data")
    # 최근 약 2년치 1시간봉 테스트 다운로드
    df = fetcher.get_or_download_data(symbol="BTCUSDT", interval="1h", start_time_str="2023-01-01 00:00:00")
    print(df.head())
    print(df.tail())
