"""
flare.data.fetch_funding_rate

바이낸스 선물 API(fapi/v1/fundingRate)로부터 BTCUSDT 펀딩비 이력 전수를 수집하여 저장하는 모듈
"""

import time
import datetime
from pathlib import Path
import requests
import pandas as pd


BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch_all_funding_rates(
    symbol: str = "BTCUSDT",
    start_time_ms: int = 1567296000000,  # 2019-09-01 (바이낸스 선물 개장 시점)
    limit: int = 1000,
    delay_sec: float = 0.2
) -> pd.DataFrame:
    """
    바이낸스 선물 API를 페이징 호출하여 전체 펀딩비 이력을 수집합니다.
    """
    all_records = []
    current_start = start_time_ms
    now_ms = int(time.time() * 1000)

    print(f"[*] 바이낸스 {symbol} 펀딩비 데이터 수집 시작...")

    while current_start < now_ms:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "limit": limit
        }
        
        try:
            resp = requests.get(BINANCE_FUNDING_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[!] API 호출 오류 발생: {e}. 2초 후 재시도...")
            time.sleep(2)
            continue

        if not data:
            print("[*] 더 이상 수집할 데이터가 없습니다.")
            break

        all_records.extend(data)
        last_time = data[-1]["fundingTime"]
        
        # 마지막 타임스탬프보다 1ms 뒤부터 다음 수집
        current_start = last_time + 1
        
        last_dt = datetime.datetime.fromtimestamp(last_time / 1000, tz=datetime.timezone.utc)
        print(f"    - 수집 누적: {len(all_records)}건 (최근 시점: {last_dt.strftime('%Y-%m-%d %H:%M:%S UTC')})")

        if len(data) < limit:
            # 마지막 페이지 도달
            break

        time.sleep(delay_sec)

    if not all_records:
        raise ValueError("수집된 펀딩비 데이터가 없습니다.")

    df = pd.DataFrame(all_records)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    if "markPrice" in df.columns:
        df["markPrice"] = pd.to_numeric(df["markPrice"], errors="coerce")

    # 중복 제거 및 정렬
    df = df.drop_duplicates(subset=["fundingTime"]).sort_values("fundingTime").reset_index(drop=True)
    
    print(f"[+] 총 {len(df)}건의 펀딩비 데이터 수집 완료! ({df['fundingTime'].min()} ~ {df['fundingTime'].max()})")
    return df


def save_funding_rates(df: pd.DataFrame, output_path: Path) -> Path:
    """수집된 펀딩비 데이터를 CSV 파일로 저장합니다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] 데이터 저장 완료: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


def main():
    base_dir = Path(__file__).resolve().parent
    output_file = base_dir / "btcusdt_funding_rate.csv"
    
    df = fetch_all_funding_rates(symbol="BTCUSDT")
    save_funding_rates(df, output_file)


if __name__ == "__main__":
    main()
