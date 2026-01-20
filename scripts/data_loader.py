"""
Historical Data Loader
과거 데이터 로더

수집된 암호화폐 OHLCV 데이터를 로드하고 관리하는 유틸리티
"""

import os
import json
import pandas as pd
from typing import List, Optional, Dict
from pathlib import Path


# 데이터 경로
DATA_DIR = Path(__file__).parent.parent / 'data' / 'historical'


def get_available_symbols() -> List[str]:
    """사용 가능한 심볼 목록 반환"""
    metadata_path = DATA_DIR / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f).get('symbols', [])
    return []


def get_available_timeframes() -> List[str]:
    """사용 가능한 타임프레임 목록 반환"""
    metadata_path = DATA_DIR / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f).get('timeframes', [])
    return []


def get_metadata() -> Dict:
    """수집 메타데이터 반환"""
    metadata_path = DATA_DIR / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return {}


def load_ohlcv(
    symbol: str,
    timeframe: str = '1h',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    OHLCV 데이터 로드
    
    Args:
        symbol: 심볼 (예: 'BTC/USDT' 또는 'BTC_USDT')
        timeframe: 타임프레임 ('1h', '4h', '1d')
        start_date: 시작일 (YYYY-MM-DD 형식)
        end_date: 종료일 (YYYY-MM-DD 형식)
    
    Returns:
        DataFrame with columns: open, high, low, close, volume
        Index: timestamp (datetime)
    
    Example:
        >>> df = load_ohlcv('BTC/USDT', '4h')
        >>> df = load_ohlcv('ETH_USDT', '1d', start_date='2024-06-01')
    """
    # 심볼 정규화
    safe_symbol = symbol.replace('/', '_')
    
    filename = f"{safe_symbol}_{timeframe}.csv"
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data not found: {filepath}\n"
            f"Available symbols: {get_available_symbols()}\n"
            f"Available timeframes: {get_available_timeframes()}"
        )
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # 날짜 필터링
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    return df


def load_multiple(
    symbols: List[str],
    timeframe: str = '1h',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    여러 심볼의 OHLCV 데이터 로드
    
    Args:
        symbols: 심볼 리스트
        timeframe: 타임프레임
        start_date: 시작일
        end_date: 종료일
    
    Returns:
        Dict[symbol, DataFrame]
    
    Example:
        >>> data = load_multiple(['BTC/USDT', 'ETH/USDT'], '4h')
        >>> btc = data['BTC/USDT']
    """
    result = {}
    for symbol in symbols:
        try:
            result[symbol] = load_ohlcv(symbol, timeframe, start_date, end_date)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    return result


def load_all(
    timeframe: str = '1h',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    모든 심볼의 OHLCV 데이터 로드
    
    Args:
        timeframe: 타임프레임
        start_date: 시작일
        end_date: 종료일
    
    Returns:
        Dict[symbol, DataFrame]
    """
    symbols = get_available_symbols()
    return load_multiple(symbols, timeframe, start_date, end_date)


def print_data_summary():
    """수집된 데이터 요약 출력"""
    metadata = get_metadata()
    
    print("=" * 60)
    print("Historical Data Summary")
    print("=" * 60)
    print(f"Source: {metadata.get('source', 'N/A')}")
    print(f"Period: {metadata.get('start_date', 'N/A')} ~ {metadata.get('end_date', 'N/A')}")
    print(f"Fetched: {metadata.get('fetched_at', 'N/A')}")
    print(f"\nSymbols ({len(get_available_symbols())}):")
    for sym in get_available_symbols():
        print(f"  - {sym}")
    print(f"\nTimeframes: {get_available_timeframes()}")
    print("=" * 60)
    
    # 각 심볼별 데이터 크기
    print("\nData Details:")
    print("-" * 60)
    for symbol in get_available_symbols():
        for tf in get_available_timeframes():
            try:
                df = load_ohlcv(symbol, tf)
                print(f"{symbol:12} {tf:4}: {len(df):6} rows  ({df.index.min().date()} ~ {df.index.max().date()})")
            except Exception as e:
                print(f"{symbol:12} {tf:4}: Error - {e}")


if __name__ == '__main__':
    print_data_summary()
