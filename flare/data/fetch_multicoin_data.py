"""
flare.data.fetch_multicoin_data

BTC, ETH, SOL, DOGE, XRP 5대 메이저 코인 과거 펀딩비 및 캔들 데이터 수집기
- 바이낸스 선물 Public REST API 활용
- 2021-01-01 ~ 2024-12-31 4개년 데이터
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")


def fetch_funding_history(symbol: str, data_dir: Path):
    output_file = data_dir / f"{symbol.lower()}_funding_rate.csv"
    if output_file.exists():
        print(f"[*] [{symbol}] 이미 펀딩비 파일이 존재합니다: {output_file.name}")
        return output_file
        
    base_url = "https://fapi.binance.com/fapi/v1/fundingRate"
    dt_start = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    dt_end = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    start_ms = int(dt_start.timestamp() * 1000)
    end_ms = int(dt_end.timestamp() * 1000)
    
    all_records = []
    curr_start = start_ms
    limit = 1000
    
    print(f"[*] [{symbol}] 4개년 펀딩비 데이터 수집 시작...")
    
    while curr_start < end_ms:
        params = {
            "symbol": symbol,
            "startTime": curr_start,
            "endTime": end_ms,
            "limit": limit
        }
        try:
            res = requests.get(base_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            print(f"[!] 에러 재시도 중: {e}")
            time.sleep(1)
            continue
            
        if not data:
            break
            
        all_records.extend(data)
        last_time = data[-1]["fundingTime"]
        curr_start = last_time + 1
        time.sleep(0.05)
        
    df = pd.DataFrame(all_records)
    df.drop_duplicates(subset=["fundingTime"], inplace=True)
    df.sort_values(by="fundingTime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df.to_csv(output_file, index=False)
    print(f"[*] [{symbol}] 펀딩비 수집 완료: 총 {len(df):,}건 저장됨.")
    return output_file


def fetch_klines_1h(symbol: str, data_dir: Path):
    output_file = data_dir / f"{symbol}_1h_4years_full.csv"
    if output_file.exists():
        print(f"[*] [{symbol}] 이미 1h 캔들 파일이 존재합니다: {output_file.name}")
        return output_file
        
    base_url = "https://fapi.binance.com/fapi/v1/klines"
    dt_start = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    dt_end = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    start_ms = int(dt_start.timestamp() * 1000)
    end_ms = int(dt_end.timestamp() * 1000)
    
    all_klines = []
    curr_start = start_ms
    limit = 1500
    
    print(f"[*] [{symbol}] 4개년 1시간봉 데이터 수집 시작...")
    
    while curr_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": curr_start,
            "endTime": end_ms,
            "limit": limit
        }
        try:
            res = requests.get(base_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            print(f"[!] 에러 재시도 중: {e}")
            time.sleep(1)
            continue
            
        if not data:
            break
            
        all_klines.extend(data)
        last_close = data[-1][6]
        curr_start = last_close + 1
        time.sleep(0.05)
        
    columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    df = pd.DataFrame(all_klines, columns=columns)
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
        df[col] = df[col].astype(float)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.to_csv(output_file, index=False)
    print(f"[*] [{symbol}] 1h 캔들 수집 완료: 총 {len(df):,}개 캔들 저장됨.")
    return output_file


def main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
    
    for sym in symbols:
        fetch_funding_history(sym, data_dir)
        fetch_klines_1h(sym, data_dir)


if __name__ == "__main__":
    main()
