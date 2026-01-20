"""
Cryptocurrency Historical Data Fetcher
암호화폐 과거 차트 데이터 수집 스크립트

ccxt 라이브러리를 사용하여 Binance에서 OHLCV 데이터를 수집합니다.
"""

import ccxt
import pandas as pd
import os
from datetime import datetime, timedelta
import time
from typing import List, Optional
import json


# 수집할 코인 목록 (Top 50 시가총액 + DeFi + Layer2 + Meme 코인들)
SYMBOLS = [
    # Top 15 by Market Cap
    'BTC/USDT',   # Bitcoin
    'ETH/USDT',   # Ethereum
    'BNB/USDT',   # Binance Coin
    'SOL/USDT',   # Solana
    'XRP/USDT',   # Ripple
    'ADA/USDT',   # Cardano
    'DOGE/USDT',  # Dogecoin
    'AVAX/USDT',  # Avalanche
    'LINK/USDT',  # Chainlink
    'MATIC/USDT', # Polygon
    'DOT/USDT',   # Polkadot
    'LTC/USDT',   # Litecoin
    'TRX/USDT',   # Tron
    'SHIB/USDT',  # Shiba Inu
    'BCH/USDT',   # Bitcoin Cash
    
    # DeFi Tokens
    'UNI/USDT',   # Uniswap
    'AAVE/USDT',  # Aave
    'MKR/USDT',   # Maker
    'CRV/USDT',   # Curve
    'SNX/USDT',   # Synthetix
    'COMP/USDT',  # Compound
    'SUSHI/USDT', # SushiSwap
    '1INCH/USDT', # 1inch
    
    # Layer 1 & Layer 2
    'ATOM/USDT',  # Cosmos
    'NEAR/USDT',  # Near Protocol
    'FTM/USDT',   # Fantom
    'ALGO/USDT',  # Algorand
    'XLM/USDT',   # Stellar
    'ETC/USDT',   # Ethereum Classic
    'HBAR/USDT',  # Hedera
    'VET/USDT',   # VeChain
    'ICP/USDT',   # Internet Computer
    'FIL/USDT',   # Filecoin
    'ARB/USDT',   # Arbitrum
    'OP/USDT',    # Optimism
    'APT/USDT',   # Aptos
    'SUI/USDT',   # Sui
    
    # Infrastructure & Gaming
    'INJ/USDT',   # Injective
    'RENDER/USDT', # Render
    'GRT/USDT',   # The Graph
    'IMX/USDT',   # Immutable X
    'GALA/USDT',  # Gala
    'AXS/USDT',   # Axie Infinity
    'SAND/USDT',  # The Sandbox
    'MANA/USDT',  # Decentraland
    
    # Meme & Others
    'PEPE/USDT',  # Pepe
    'WIF/USDT',   # dogwifhat
    'BONK/USDT',  # Bonk
]

# 수집 설정
TIMEFRAMES = ['1h', '4h', '1d']  # 수집할 타임프레임
START_DATE = '2020-01-01T00:00:00Z'  # 시작일 (6년 데이터)
END_DATE = '2026-01-20T00:00:00Z'   # 종료일

# 저장 경로
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'historical')


def create_exchange():
    """Binance 거래소 인스턴스 생성"""
    exchange = ccxt.binance({
        'enableRateLimit': True,  # Rate limit 자동 적용
        'options': {
            'defaultType': 'spot',
        }
    })
    return exchange


def fetch_ohlcv_data(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: int,
    until: int
) -> pd.DataFrame:
    """
    OHLCV 데이터 수집
    
    Args:
        exchange: ccxt 거래소 인스턴스
        symbol: 심볼 (예: 'BTC/USDT')
        timeframe: 타임프레임 (예: '1h', '4h', '1d')
        since: 시작 timestamp (ms)
        until: 종료 timestamp (ms)
    
    Returns:
        DataFrame with OHLCV data
    """
    all_ohlcv = []
    current_since = since
    
    # 타임프레임별 밀리초 계산
    timeframe_ms = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
    }
    
    ms_per_candle = timeframe_ms.get(timeframe, 60 * 60 * 1000)
    limit = 1000  # Binance 최대 한도
    
    print(f"  Fetching {symbol} {timeframe}...")
    
    while current_since < until:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since,
                limit=limit
            )
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # 다음 배치 시작점
            last_timestamp = ohlcv[-1][0]
            current_since = last_timestamp + ms_per_candle
            
            # 진행 상황 표시
            progress_date = datetime.fromtimestamp(last_timestamp / 1000)
            print(f"    → {progress_date.strftime('%Y-%m-%d')} ({len(all_ohlcv)} candles)")
            
            # Rate limit 대기
            time.sleep(0.1)
            
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(1)
            continue
    
    if not all_ohlcv:
        return pd.DataFrame()
    
    # DataFrame 생성
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # 중복 제거
    df = df[~df.index.duplicated(keep='first')]
    
    # 종료일 필터링
    until_dt = datetime.fromtimestamp(until / 1000)
    df = df[df.index <= until_dt]
    
    return df


def save_data(df: pd.DataFrame, symbol: str, timeframe: str):
    """데이터를 CSV로 저장"""
    if df.empty:
        print(f"  No data for {symbol} {timeframe}")
        return
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 파일명 생성 (슬래시를 언더스코어로 변환)
    safe_symbol = symbol.replace('/', '_')
    filename = f"{safe_symbol}_{timeframe}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    df.to_csv(filepath)
    print(f"  Saved: {filepath} ({len(df)} rows)")


def create_metadata(symbols: List[str], timeframes: List[str]):
    """수집 메타데이터 저장"""
    metadata = {
        'symbols': symbols,
        'timeframes': timeframes,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'fetched_at': datetime.now().isoformat(),
        'source': 'binance',
    }
    
    metadata_path = os.path.join(DATA_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved: {metadata_path}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Cryptocurrency Historical Data Fetcher")
    print("=" * 60)
    print(f"Symbols: {len(SYMBOLS)} coins")
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Period: {START_DATE} ~ {END_DATE}")
    print("=" * 60)
    
    # 거래소 초기화
    exchange = create_exchange()
    
    # 타임스탬프 변환
    since_ts = int(datetime.fromisoformat(START_DATE.replace('Z', '+00:00')).timestamp() * 1000)
    until_ts = int(datetime.fromisoformat(END_DATE.replace('Z', '+00:00')).timestamp() * 1000)
    
    # 데이터 디렉토리 생성
    os.makedirs(DATA_DIR, exist_ok=True)
    
    successful = []
    failed = []
    
    for symbol in SYMBOLS:
        print(f"\n[{symbol}]")
        
        for timeframe in TIMEFRAMES:
            try:
                df = fetch_ohlcv_data(exchange, symbol, timeframe, since_ts, until_ts)
                save_data(df, symbol, timeframe)
                successful.append(f"{symbol}_{timeframe}")
            except Exception as e:
                print(f"  Failed {symbol} {timeframe}: {e}")
                failed.append(f"{symbol}_{timeframe}")
    
    # 메타데이터 저장
    create_metadata(SYMBOLS, TIMEFRAMES)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed items: {failed}")
    print(f"\nData saved to: {DATA_DIR}")


if __name__ == '__main__':
    main()
