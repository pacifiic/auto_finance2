"""
Data Utilities
데이터 유틸리티

샘플 데이터 생성 및 데이터 로딩 기능을 제공합니다.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_data(
    n_samples: int = 500,
    start_price: float = 50000.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    테스트용 샘플 OHLCV 데이터 생성
    
    Args:
        n_samples: 데이터 포인트 수
        start_price: 시작 가격
        volatility: 변동성 (일일 표준편차)
        trend: 추세 (양수: 상승, 음수: 하락)
        seed: 랜덤 시드
        
    Returns:
        OHLCV 데이터프레임
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 날짜 생성
    dates = pd.date_range(
        end=datetime.now(),
        periods=n_samples,
        freq='4h'  # 4시간봉
    )
    
    # 가격 생성 (기하 브라운 운동)
    returns = np.random.normal(trend, volatility, n_samples)
    price_multipliers = np.exp(returns)
    close_prices = start_price * np.cumprod(price_multipliers)
    
    # OHLC 생성
    data = []
    for i, close in enumerate(close_prices):
        # 일중 변동폭
        intraday_vol = volatility * np.random.uniform(0.5, 1.5)
        
        # High, Low, Open 생성
        high_low_range = close * intraday_vol
        high = close + np.random.uniform(0, high_low_range)
        low = close - np.random.uniform(0, high_low_range)
        
        if i == 0:
            open_price = start_price
        else:
            open_price = close_prices[i - 1] * (1 + np.random.uniform(-0.005, 0.005))
        
        # 일관성 보장
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # 거래량 (가격 변동과 양의 상관관계)
        base_volume = 1000000
        volume_multiplier = 1 + abs(returns[i]) * 10
        volume = base_volume * volume_multiplier * np.random.uniform(0.8, 1.2)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'timestamp'
    
    return df


def generate_trending_data(
    n_samples: int = 500,
    start_price: float = 50000.0,
    trend_type: str = 'up',
    trend_strength: float = 0.001,
    volatility: float = 0.015
) -> pd.DataFrame:
    """
    추세가 있는 샘플 데이터 생성
    
    Args:
        trend_type: 'up', 'down', 'sideways', 'reversal'
        trend_strength: 추세 강도
    """
    if trend_type == 'up':
        trend = trend_strength
    elif trend_type == 'down':
        trend = -trend_strength
    elif trend_type == 'sideways':
        trend = 0
    elif trend_type == 'reversal':
        # 전반부 상승, 후반부 하락
        df1 = generate_sample_data(n_samples // 2, start_price, volatility, trend_strength)
        end_price = df1['close'].iloc[-1]
        df2 = generate_sample_data(n_samples - n_samples // 2, end_price, volatility, -trend_strength)
        
        # 날짜 조정
        df2.index = pd.date_range(
            start=df1.index[-1] + timedelta(hours=4),
            periods=len(df2),
            freq='4h'
        )
        
        return pd.concat([df1, df2])
    else:
        trend = 0
    
    return generate_sample_data(n_samples, start_price, volatility, trend)


class DataLoader:
    """
    데이터 로더
    
    다양한 소스에서 OHLCV 데이터를 로드합니다.
    """
    
    @staticmethod
    def from_csv(filepath: str, 
                 date_column: str = 'timestamp',
                 date_format: Optional[str] = None) -> pd.DataFrame:
        """
        CSV 파일에서 데이터 로드
        
        Args:
            filepath: CSV 파일 경로
            date_column: 날짜 컬럼 이름
            date_format: 날짜 형식
            
        Returns:
            OHLCV 데이터프레임
        """
        df = pd.read_csv(filepath)
        
        # 날짜 파싱
        if date_format:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
        else:
            df[date_column] = pd.to_datetime(df[date_column])
        
        df.set_index(date_column, inplace=True)
        
        # 컬럼 이름 표준화
        df.columns = df.columns.str.lower()
        
        # 필수 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df[required_columns]
    
    @staticmethod
    def from_dict(data: dict) -> pd.DataFrame:
        """
        딕셔너리에서 데이터 로드
        
        Args:
            data: OHLCV 데이터 딕셔너리
            
        Returns:
            OHLCV 데이터프레임
        """
        df = pd.DataFrame(data)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        df.columns = df.columns.str.lower()
        
        return df
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> Tuple[bool, list]:
        """
        데이터 유효성 검사
        
        Returns:
            (유효 여부, 에러 메시지 리스트)
        """
        errors = []
        
        # 필수 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing column: {col}")
        
        if errors:
            return False, errors
        
        # 데이터 일관성 확인
        if (df['high'] < df['low']).any():
            errors.append("High < Low detected")
        
        if (df['high'] < df['close']).any():
            errors.append("High < Close detected")
        
        if (df['low'] > df['close']).any():
            errors.append("Low > Close detected")
        
        if (df['high'] < df['open']).any():
            errors.append("High < Open detected")
        
        if (df['low'] > df['open']).any():
            errors.append("Low > Open detected")
        
        # NaN 확인
        if df.isnull().any().any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            errors.append(f"NaN values in columns: {nan_cols}")
        
        # 음수 확인
        if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
            errors.append("Negative values detected")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        타임프레임 리샘플링
        
        Args:
            df: OHLCV 데이터프레임
            timeframe: 새 타임프레임 ('1h', '4h', '1d' 등)
            
        Returns:
            리샘플된 데이터프레임
        """
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        return df.resample(timeframe).agg(ohlc_dict).dropna()
