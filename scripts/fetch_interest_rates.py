"""
Interest Rate Data Fetcher
금리 데이터 수집 스크립트

Fred API를 통해 미국 연방기금금리(Fed Funds Rate)와 
한국 기준금리 데이터를 시계열로 수집합니다.
"""

import os
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path

# 저장 경로
DATA_DIR = Path(__file__).parent.parent / 'data' / 'macro'

# FRED API (무료, 등록 필요: https://fred.stlouisfed.org/docs/api/api_key.html)
# 환경변수에서 키 로드, 없으면 공개 데이터 사용
FRED_API_KEY = os.environ.get('FRED_API_KEY', None)

# 수집할 금리 시리즈
RATE_SERIES = {
    'fed_funds': {
        'name': 'Federal Funds Rate',
        'fred_id': 'FEDFUNDS',  # 월간 평균
        'description': 'US Federal Funds Effective Rate (Monthly)',
    },
    'fed_funds_daily': {
        'name': 'Federal Funds Rate (Daily)',
        'fred_id': 'DFF',  # 일간
        'description': 'US Federal Funds Effective Rate (Daily)',
    },
    'treasury_10y': {
        'name': '10-Year Treasury Rate',
        'fred_id': 'DGS10',  # 일간
        'description': 'US 10-Year Treasury Constant Maturity Rate',
    },
    'treasury_2y': {
        'name': '2-Year Treasury Rate',
        'fred_id': 'DGS2',  # 일간
        'description': 'US 2-Year Treasury Constant Maturity Rate',
    },
}


def fetch_fred_data(series_id: str, start_date: str = '2015-01-01') -> pd.DataFrame:
    """
    FRED API에서 데이터 수집
    
    Args:
        series_id: FRED 시리즈 ID
        start_date: 시작일
        
    Returns:
        DataFrame with date and value
    """
    if FRED_API_KEY:
        # API 키가 있으면 API 사용
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'observation_start': start_date,
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'observations' not in data:
            print(f"  Error fetching {series_id}: {data.get('error_message', 'Unknown error')}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[['date', 'value']].dropna()
        
    else:
        # API 키 없으면 대체 방법 사용 (pandas-datareader)
        try:
            import pandas_datareader as pdr
            df = pdr.get_data_fred(series_id, start=start_date)
            df = df.reset_index()
            df.columns = ['date', 'value']
        except ImportError:
            print("  pandas-datareader not installed. Installing...")
            os.system("pip install pandas-datareader -q")
            import pandas_datareader as pdr
            df = pdr.get_data_fred(series_id, start=start_date)
            df = df.reset_index()
            df.columns = ['date', 'value']
        except Exception as e:
            print(f"  Error: {e}")
            return pd.DataFrame()
    
    return df


def fetch_korea_rate() -> pd.DataFrame:
    """
    한국은행 기준금리 데이터 수집 (공개 데이터)
    
    한국은행 경제통계시스템(ECOS)에서 기준금리 수집
    """
    try:
        # Bank of Korea base rate history (수동 데이터)
        # 실제로는 ECOS API 사용 권장
        korea_rates = [
            ('2015-01-01', 2.00), ('2015-03-12', 1.75), ('2015-06-11', 1.50),
            ('2016-06-09', 1.25), ('2017-11-30', 1.50), ('2018-11-30', 1.75),
            ('2019-07-18', 1.50), ('2019-10-16', 1.25), ('2020-03-16', 0.75),
            ('2020-05-28', 0.50), ('2021-08-26', 0.75), ('2021-11-25', 1.00),
            ('2022-01-14', 1.25), ('2022-02-24', 1.25), ('2022-04-14', 1.50),
            ('2022-05-26', 1.75), ('2022-07-13', 2.25), ('2022-08-25', 2.50),
            ('2022-10-12', 3.00), ('2022-11-24', 3.25), ('2023-01-13', 3.50),
            ('2023-02-23', 3.50), ('2024-10-11', 3.25), ('2024-11-28', 3.00),
            ('2025-01-16', 2.75),
        ]
        
        df = pd.DataFrame(korea_rates, columns=['date', 'value'])
        df['date'] = pd.to_datetime(df['date'])
        
        # 일별로 확장 (forward fill)
        date_range = pd.date_range(start='2015-01-01', end=datetime.now(), freq='D')
        df_daily = pd.DataFrame({'date': date_range})
        df_daily = df_daily.merge(df, on='date', how='left')
        df_daily['value'] = df_daily['value'].ffill()
        
        return df_daily.dropna()
        
    except Exception as e:
        print(f"  Error fetching Korea rate: {e}")
        return pd.DataFrame()


def classify_economic_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    경제 상황 4단계 분류
    
    1. 저금리: 금리 < 2%
    2. 고금리: 금리 >= 4%
    3. 확장기: 금리 하락 추세
    4. 침체기: 금리 상승 추세
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # 금리 수준 분류
    df['rate_level'] = pd.cut(
        df['value'], 
        bins=[-1, 2, 4, 100],
        labels=['low', 'mid', 'high']
    )
    
    # 금리 추세 (30일 이동평균 기준)
    df['value_ma'] = df['value'].rolling(30, min_periods=1).mean()
    df['trend'] = df['value_ma'].diff(30)
    df['trend_direction'] = df['trend'].apply(
        lambda x: 'falling' if x < -0.1 else ('rising' if x > 0.1 else 'stable')
    )
    
    # 4단계 분류
    def classify(row):
        if row['rate_level'] == 'low':
            return 'low_rate'
        elif row['rate_level'] == 'high':
            return 'high_rate'
        elif row['trend_direction'] == 'falling':
            return 'expansion'
        elif row['trend_direction'] == 'rising':
            return 'contraction'
        else:
            return 'stable'
    
    df['regime'] = df.apply(classify, axis=1)
    
    return df


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Interest Rate Data Fetcher")
    print("=" * 60)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. US Fed Funds Rate (Daily)
    print("\n[1] US Federal Funds Rate (Daily)")
    df_fed = fetch_fred_data('DFF', '2015-01-01')
    if not df_fed.empty:
        df_fed = classify_economic_regime(df_fed)
        filepath = DATA_DIR / 'us_fed_funds_daily.csv'
        df_fed.to_csv(filepath, index=False)
        print(f"  Saved: {filepath} ({len(df_fed)} rows)")
        print(f"  Latest: {df_fed['date'].max().date()} = {df_fed['value'].iloc[-1]:.2f}%")
    
    # 2. US 10-Year Treasury
    print("\n[2] US 10-Year Treasury Rate")
    df_10y = fetch_fred_data('DGS10', '2015-01-01')
    if not df_10y.empty:
        filepath = DATA_DIR / 'us_treasury_10y.csv'
        df_10y.to_csv(filepath, index=False)
        print(f"  Saved: {filepath} ({len(df_10y)} rows)")
    
    # 3. US 2-Year Treasury
    print("\n[3] US 2-Year Treasury Rate")
    df_2y = fetch_fred_data('DGS2', '2015-01-01')
    if not df_2y.empty:
        filepath = DATA_DIR / 'us_treasury_2y.csv'
        df_2y.to_csv(filepath, index=False)
        print(f"  Saved: {filepath} ({len(df_2y)} rows)")
    
    # 4. Yield Curve (10Y - 2Y spread, 경기 침체 예측 지표)
    if not df_10y.empty and not df_2y.empty:
        print("\n[4] Yield Curve Spread (10Y - 2Y)")
        df_spread = df_10y.merge(df_2y, on='date', suffixes=('_10y', '_2y'))
        df_spread['spread'] = df_spread['value_10y'] - df_spread['value_2y']
        df_spread['inverted'] = df_spread['spread'] < 0  # 역전시 경기침체 신호
        filepath = DATA_DIR / 'us_yield_curve_spread.csv'
        df_spread.to_csv(filepath, index=False)
        print(f"  Saved: {filepath} ({len(df_spread)} rows)")
        print(f"  Current spread: {df_spread['spread'].iloc[-1]:.2f}%")
        print(f"  Inverted: {df_spread['inverted'].iloc[-1]}")
    
    # 5. Korea Base Rate
    print("\n[5] Korea Base Rate")
    df_korea = fetch_korea_rate()
    if not df_korea.empty:
        df_korea = classify_economic_regime(df_korea)
        filepath = DATA_DIR / 'korea_base_rate.csv'
        df_korea.to_csv(filepath, index=False)
        print(f"  Saved: {filepath} ({len(df_korea)} rows)")
        print(f"  Latest: {df_korea['date'].max().date()} = {df_korea['value'].iloc[-1]:.2f}%")
    
    # 6. 메타데이터 저장
    metadata = {
        'fetched_at': datetime.now().isoformat(),
        'sources': {
            'us_rates': 'FRED (Federal Reserve Economic Data)',
            'korea_rate': 'Bank of Korea (manual)',
        },
        'files': [
            'us_fed_funds_daily.csv',
            'us_treasury_10y.csv',
            'us_treasury_2y.csv',
            'us_yield_curve_spread.csv',
            'korea_base_rate.csv',
        ]
    }
    
    metadata_path = DATA_DIR / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Interest Rate Data Collection Complete!")
    print(f"Data saved to: {DATA_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
