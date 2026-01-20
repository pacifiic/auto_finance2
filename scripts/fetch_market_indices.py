"""
Market Indices Fetcher
시장 지수 데이터 수집 스크립트 (VIX, Nasdaq)

yfinance를 사용하여 주요 시장 지수 데이터를 수집합니다.
- VIX (CBOE Volatility Index): 시장 공포 지수
- IXIC (Nasdaq Composite): 기술주 중심 지수 (코인과 상관관계 높음)
- GSPC (S&P 500): 미국 주식 시장 대표 지수
- DX-Y.NYB (US Dollar Index): 달러 인덱스 (코인과 역상관)
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 저장 경로
DATA_DIR = Path(__file__).parent.parent / 'data' / 'macro'

# 수집할 종목
INDICES = {
    '^VIX': 'vix',           # Volatility Index
    '^IXIC': 'nasdaq',       # Nasdaq Composite
    '^GSPC': 'sp500',        # S&P 500
    'DX-Y.NYB': 'dollar',    # US Dollar Index
}

def fetch_indices(start_date: str = '2018-01-01'):
    """yfinance를 통해 지수 데이터 수집"""
    print(f"Fetching market indices from {start_date}...")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for symbol, name in INDICES.items():
        print(f"\n[{name.upper()}] ({symbol})")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, interval='1d')
            
            if df.empty:
                print(f"  ⚠️ No data found for {symbol}")
                continue
            
            # 필요한 컬럼만 선택 및 정리
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = [c.lower() for c in df.columns]
            df.index.name = 'date'
            
            # 시간대 제거 (날짜만 유지)
            df.index = df.index.tz_localize(None)
            
            # 저장
            filepath = DATA_DIR / f'{name}_daily.csv'
            df.to_csv(filepath)
            
            print(f"  ✅ Saved: {filepath}")
            print(f"  Rows: {len(df)}")
            print(f"  Latest: {df.index.max().date()} = {df['close'].iloc[-1]:.2f}")
            
        except Exception as e:
            print(f"  ❌ Error fetching {symbol}: {e}")

def merge_macro_data():
    """수집된 모든 거시/지수 데이터를 하나의 CSV로 병합"""
    print(f"\n[Merging Macro Data]")
    
    all_dfs = []
    
    # 1. Market Indices
    for _, name in INDICES.items():
        filepath = DATA_DIR / f'{name}_daily.csv'
        if filepath.exists():
            df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
            df = df[['close']].rename(columns={'close': name})
            all_dfs.append(df)
            
    # 2. Interest Rates (기존 fetch_interest_rates.py 결과)
    rate_files = {
        'us_treasury_10y.csv': 'rate_10y',
        'korea_base_rate.csv': 'rate_kr',
    }
    
    for fname, col in rate_files.items():
        filepath = DATA_DIR / fname
        if filepath.exists():
            df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
            if 'value' in df.columns:
                df = df[['value']].rename(columns={'value': col})
                all_dfs.append(df)
    
    # 3. Fear & Greed (Sentiment)
    sentiment_dir = Path(__file__).parent.parent / 'data' / 'sentiment'
    fng_path = sentiment_dir / 'fear_greed_index.csv'
    if fng_path.exists():
        df = pd.read_csv(fng_path, parse_dates=['timestamp'], index_col='timestamp')
        df.index.name = 'date'
        df = df[['value']].rename(columns={'value': 'fng'})
        all_dfs.append(df)

    if not all_dfs:
        print("No data to merge.")
        return

    # 병합 (Outer Join)
    merged = pd.concat(all_dfs, axis=1).sort_index()
    merged = merged.ffill() # 주말/휴일 데이터 채우기
    
    # 2018년 이후 데이터만 저장
    merged = merged[merged.index >= '2018-01-01']
    
    save_path = DATA_DIR / 'all_macro_indicators.csv'
    merged.to_csv(save_path)
    print(f"✅ Merged Data Saved: {save_path}")
    print(f"  Rows: {len(merged)}")
    print(f"  Columns: {list(merged.columns)}")

if __name__ == '__main__':
    fetch_indices()
    merge_macro_data()
