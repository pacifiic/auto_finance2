"""
Base Indicator Class
기본 지표 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


class BaseIndicator(ABC):
    """
    모든 기술적 지표의 기본 클래스
    
    Attributes:
        name: 지표 이름
        params: 지표 파라미터
        weight: 신호 가중치
        enabled: 활성화 여부
    """
    
    def __init__(self, name: str, params: Dict[str, Any], weight: float = 1.0, enabled: bool = True):
        self.name = name
        self.params = params
        self.weight = weight
        self.enabled = enabled
        self._values: Optional[pd.DataFrame] = None
        self._signals: Optional[pd.Series] = None
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        지표 값 계산
        
        Args:
            df: OHLCV 데이터프레임 (open, high, low, close, volume)
            
        Returns:
            계산된 지표 값이 포함된 데이터프레임
        """
        pass
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        매매 신호 생성
        
        Args:
            df: 지표 값이 포함된 데이터프레임
            
        Returns:
            신호 시리즈 (-1 ~ 1, 음수: 매도, 양수: 매수)
        """
        pass
    
    def get_signal_with_weight(self, df: pd.DataFrame) -> pd.Series:
        """
        가중치가 적용된 신호 반환
        """
        if not self.enabled:
            return pd.Series(0, index=df.index)
        
        signals = self.generate_signals(df)
        return signals * self.weight
    
    @staticmethod
    def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        상향 교차 감지 (series1이 series2를 위로 돌파)
        """
        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))
    
    @staticmethod
    def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        하향 교차 감지 (series1이 series2를 아래로 돌파)
        """
        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))
    
    @staticmethod
    def detect_divergence(price: pd.Series, indicator: pd.Series, 
                          lookback: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        다이버전스 감지 (가격과 지표의 괴리)
        
        Returns:
            (bullish_divergence, bearish_divergence) 튜플
        """
        # 로컬 고점/저점 찾기
        price_highs = price.rolling(lookback, center=True).max() == price
        price_lows = price.rolling(lookback, center=True).min() == price
        
        ind_highs = indicator.rolling(lookback, center=True).max() == indicator
        ind_lows = indicator.rolling(lookback, center=True).min() == indicator
        
        # Bullish divergence: 가격 저점 하락 but 지표 저점 상승
        bullish_div = pd.Series(False, index=price.index)
        bearish_div = pd.Series(False, index=price.index)
        
        for i in range(lookback, len(price)):
            if price_lows.iloc[i]:
                # 이전 저점 찾기
                prev_lows = price_lows.iloc[max(0, i-lookback*2):i]
                if prev_lows.any():
                    prev_low_idx = prev_lows[prev_lows].index[-1]
                    # 가격은 낮아졌는데 지표는 높아졌으면 bullish
                    if (price.iloc[i] < price.loc[prev_low_idx] and 
                        indicator.iloc[i] > indicator.loc[prev_low_idx]):
                        bullish_div.iloc[i] = True
            
            if price_highs.iloc[i]:
                prev_highs = price_highs.iloc[max(0, i-lookback*2):i]
                if prev_highs.any():
                    prev_high_idx = prev_highs[prev_highs].index[-1]
                    # 가격은 높아졌는데 지표는 낮아졌으면 bearish
                    if (price.iloc[i] > price.loc[prev_high_idx] and 
                        indicator.iloc[i] < indicator.loc[prev_high_idx]):
                        bearish_div.iloc[i] = True
        
        return bullish_div, bearish_div
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.weight}, enabled={self.enabled})"
