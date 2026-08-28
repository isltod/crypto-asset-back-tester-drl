import sys
import os
import json
import urllib.request
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

# Windows 콘솔 UTF-8 출력 보장
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def fetch_fng_data():
    print("1. Alternative.me Fear & Greed Index 다운로드 중...")
    url = "https://api.alternative.me/fng/?limit=0"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    
    df = pd.DataFrame(data['data'])
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
    df['fng_value'] = pd.to_numeric(df['value'])
    df = df[['date', 'fng_value']].sort_values('date').drop_duplicates('date').reset_index(drop=True)
    print(f"   -> F&G 데이터 {len(df)}일 확보 ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
    return df

def fetch_binance_btc_daily():
    print("2. 바이낸스 BTC/USDT 일봉 시세 다운로드 중...")
    all_klines = []
    start_ts = int(datetime(2018, 1, 1).timestamp() * 1000)
    
    while True:
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&startTime={start_ts}&limit=1000"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            klines = json.loads(response.read().decode())
        
        if not klines:
            break
        all_klines.extend(klines)
        if len(klines) < 1000:
            break
        start_ts = klines[-1][0] + 86400000
    
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    df['date'] = pd.to_datetime(df['open_time'], unit='ms').dt.strftime('%Y-%m-%d')
    df['close'] = pd.to_numeric(df['close'])
    df = df[['date', 'close']].sort_values('date').drop_duplicates('date').reset_index(drop=True)
    print(f"   -> 바이낸스 BTC 일봉 {len(df)}일 확보 ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
    return df

def analyze_period(df, period_name):
    print("\n" + "="*80)
    print(f" ■ 분석 대상 기간: {period_name} (총 {len(df)} 영업일)")
    print("="*80)
    
    horizons = [1, 3, 7, 14, 30]
    for h in horizons:
        df[f'fwd_ret_{h}d'] = (df['close'].shift(-h) / df['close'] - 1.0) * 100.0

    # 1단계: 선형 및 순위 상관분석
    print("\n[1단계: 전체 연속형 시차 상관분석 (Linear & Rank Correlation)]")
    corr_records = []
    for h in horizons:
        valid_df = df.dropna(subset=[f'fwd_ret_{h}d', 'fng_value'])
        p_corr, p_val = stats.pearsonr(valid_df['fng_value'], valid_df[f'fwd_ret_{h}d'])
        s_corr, s_val = stats.spearmanr(valid_df['fng_value'], valid_df[f'fwd_ret_{h}d'])
        
        corr_records.append({
            '미래 기간': f'{h}일 후 (t+{h})',
            'Pearson r': f"{p_corr:+.4f}",
            'Pearson p-val': f"{p_val:.4f} {'***' if p_val < 0.01 else '**' if p_val < 0.05 else ''}",
            'Spearman rho': f"{s_corr:+.4f}",
            'Spearman p-val': f"{s_val:.4f} {'***' if s_val < 0.01 else '**' if s_val < 0.05 else ''}"
        })
    print(pd.DataFrame(corr_records).to_string(index=False))
    
    # 2단계: 10분위수(Decile) 분석 (7일 및 14일 미래 수익률 기준)
    print("\n[2단계: F&G 지수 10분위수(Decile)별 미래 7일 및 14일 수익률 분포]")
    bins = list(range(0, 101, 10))
    labels = [f"{i}~{i+10}" for i in range(0, 100, 10)]
    df['decile'] = pd.cut(df['fng_value'], bins=bins, labels=labels, right=True, include_lowest=True)
    
    decile_summary = []
    for label in labels:
        sub = df[df['decile'] == label]
        n_samples = len(sub)
        if n_samples == 0:
            continue
        
        r7 = sub['fwd_ret_7d'].dropna()
        r14 = sub['fwd_ret_14d'].dropna()
        
        decile_summary.append({
            'F&G 구간': label,
            '표본수': n_samples,
            '7일 평균수익률': f"{r7.mean():+.2f}%" if len(r7) > 0 else "N/A",
            '7일 승률': f"{(r7 > 0).mean()*100:.1f}%" if len(r7) > 0 else "N/A",
            '14일 평균수익률': f"{r14.mean():+.2f}%" if len(r14) > 0 else "N/A",
            '14일 승률': f"{(r14 > 0).mean()*100:.1f}%" if len(r14) > 0 else "N/A"
        })
    print(pd.DataFrame(decile_summary).to_string(index=False))

def main():
    fng_df = fetch_fng_data()
    btc_df = fetch_binance_btc_daily()
    
    merged = pd.merge(fng_df, btc_df, on='date').sort_values('date').reset_index(drop=True)
    print(f"\n최종 결합된 일봉 데이터: {len(merged)}일 ({merged['date'].iloc[0]} ~ {merged['date'].iloc[-1]})")
    
    # 1. 전체 기간 분석 (2018 ~ 현재)
    analyze_period(merged.copy(), "전체 기간 (2018 ~ 2026)")
    
    # 2. 최근 구조적 변화 기간 (2022 ~ 현재: 금리인상 & FTX & ETF)
    df_post_2022 = merged[merged['date'] >= '2022-01-01'].copy().reset_index(drop=True)
    analyze_period(df_post_2022, "최신 시장 환경 (2022 ~ 2026)")
    
    # 3. 비트코인 현물 ETF 승인 이후 (2024 ~ 현재)
    df_post_etf = merged[merged['date'] >= '2024-01-10'].copy().reset_index(drop=True)
    analyze_period(df_post_etf, "ETF 승인 이후 기관화 시장 (2024 ~ 2026)")

if __name__ == '__main__':
    main()
