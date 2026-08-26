"""
flare.data.fetch_2021_5m

2021년 BTCUSDT 5분봉(5m) 선물 과거 캔들 데이터 다운로더
- Binance Futures Public REST API 활용
- 2021-01-01 00:00:00 ~ 2022-01-01 00:00:00 (1년 치 OOS 전용)
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

# 콘솔 UTF-8 출력 강제
sys.stdout.reconfigure(encoding="utf-8")


def fetch_2021_5m_klines():
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    output_file = data_dir / "BTCUSDT_5m_2021.csv"
    
    if output_file.exists():
        print(f"[*] 이미 2021년 5분봉 파일이 존재합니다: {output_file.name}")
        df = pd.read_csv(output_file)
        print(f"    - 데이터 행 수: {len(df):,}개 | 시작: {df['datetime'].iloc[0]} | 끝: {df['datetime'].iloc[-1]}")
        return output_file
        
    base_url = "https://fapi.binance.com/fapi/v1/klines"
    
    dt_start = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    dt_end = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    start_ms = int(dt_start.timestamp() * 1000)
    end_ms = int(dt_end.timestamp() * 1000)
    
    all_klines = []
    curr_start = start_ms
    limit = 1500
    
    print(f"[*] 2021년 BTCUSDT 5분봉 데이터 다운로드 시작 (2021-01-01 ~ 2021-12-31)...")
    
    while curr_start < end_ms:
        params = {
            "symbol": "BTCUSDT",
            "interval": "5m",
            "startTime": curr_start,
            "endTime": end_ms,
            "limit": limit
        }
        
        try:
            res = requests.get(base_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            print(f"[!] 다운로드 재시도 중... 에러: {e}")
            time.sleep(2)
            continue
            
        if not data:
            break
            
        all_klines.extend(data)
        last_close_time = data[-1][6]
        curr_start = last_close_time + 1
        
        # 진행상황 표시
        pct = (curr_start - start_ms) / (end_ms - start_ms) * 100
        print(f"    ➔ 수집 중: {len(all_klines):,}개 캔들 완료 ({pct:.1f}%)", end="\r")
        time.sleep(0.1)
        
    print(f"\n[*] 다운로드 완료! 총 {len(all_klines):,}개 5분봉 수집됨.")
    
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
    
    # 저장
    df.to_csv(output_file, index=False)
    print(f"[*] 파일 저장 성공: {output_file.name} ({output_file.stat().st_size / 1024 / 1024:.2f} MB)")
    return output_file


if __name__ == "__main__":
    fetch_2021_5m_klines()
