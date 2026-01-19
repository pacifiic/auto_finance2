"""
Trend Indicators
추세 지표 모듈 - SMA, EMA, MACD, ADX
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from .base import BaseIndicator


class TrendIndicators:
    """
    추세 관련 기술적 지표 모음
    
    포함 지표:
    - SMA (Simple Moving Average)
    - EMA (Exponential Moving Average)
    - MACD (Moving Average Convergence Divergence)
    - ADX (Average Directional Index)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: YAML 설정에서 로드된 trend_indicators 섹션
        """
        self.config = config or self._default_config()
        self.indicators = {}
        self._initialize_indicators()
    
    def _default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'sma': {
                'enabled': True,
                'weight': 1.0,
                'params': {'short_period': 20, 'medium_period': 50, 'long_period': 200},
                'signals': {'golden_cross': 1.0, 'death_cross': -1.0, 
                           'price_above_ma': 0.5, 'price_below_ma': -0.5}
            },
            'ema': {
                'enabled': True,
                'weight': 1.2,
                'params': {'short_period': 12, 'medium_period': 26, 'long_period': 50},
                'signals': {'golden_cross': 1.0, 'death_cross': -1.0, 
                           'trend_strength_multiplier': 0.3}
            },
            'macd': {
                'enabled': True,
                'weight': 1.5,
                'params': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                'signals': {'bullish_crossover': 1.0, 'bearish_crossover': -1.0,
                           'histogram_positive': 0.3, 'histogram_negative': -0.3,
                           'divergence_multiplier': 0.5}
            },
            'adx': {
                'enabled': True,
                'weight': 0.8,
                'params': {'period': 14, 'strong_trend_threshold': 25, 
                          'weak_trend_threshold': 20},
                'signals': {'strong_uptrend': 0.5, 'strong_downtrend': -0.5, 
                           'no_trend': 0.0}
            }
        }
    
    def _initialize_indicators(self):
        """지표 초기화"""
        self.indicators = {
            'sma': SMAIndicator(self.config.get('sma', {})),
            'ema': EMAIndicator(self.config.get('ema', {})),
            'macd': MACDIndicator(self.config.get('macd', {})),
            'adx': ADXIndicator(self.config.get('adx', {}))
        }
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 추세 지표 계산
        
        Args:
            df: OHLCV 데이터프레임
            
        Returns:
            모든 지표가 추가된 데이터프레임
        """
        result = df.copy()
        for name, indicator in self.indicators.items():
            if indicator.enabled:
                result = indicator.calculate(result)
        return result
    
    def generate_all_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        모든 추세 지표의 신호 생성
        
        Returns:
            {지표명: 신호 시리즈} 딕셔너리
        """
        signals = {}
        for name, indicator in self.indicators.items():
            if indicator.enabled:
                signals[name] = indicator.get_signal_with_weight(df)
        return signals
    
    def get_combined_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        모든 추세 지표의 조합된 신호 반환
        """
        signals = self.generate_all_signals(df)
        if not signals:
            return pd.Series(0, index=df.index)
        
        combined = pd.concat(signals.values(), axis=1)
        total_weight = sum(ind.weight for ind in self.indicators.values() if ind.enabled)
        
        return combined.sum(axis=1) / total_weight if total_weight > 0 else combined.sum(axis=1)


class SMAIndicator(BaseIndicator):
    """Simple Moving Average 지표"""
    
    def __init__(self, config: Dict[str, Any]):
        params = config.get('params', {})
        super().__init__(
            name='SMA',
            params=params,
            weight=config.get('weight', 1.0),
            enabled=config.get('enabled', True)
        )
        self.signals_config = config.get('signals', {})
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """SMA 계산"""
        result = df.copy()
        
        short = self.params.get('short_period', 20)
        medium = self.params.get('medium_period', 50)
        long_ = self.params.get('long_period', 200)
        
        result[f'sma_{short}'] = df['close'].rolling(window=short).mean()
        result[f'sma_{medium}'] = df['close'].rolling(window=medium).mean()
        result[f'sma_{long_}'] = df['close'].rolling(window=long_).mean()
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """SMA 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        short = self.params.get('short_period', 20)
        medium = self.params.get('medium_period', 50)
        long_ = self.params.get('long_period', 200)
        
        sma_short = df.get(f'sma_{short}', df['close'].rolling(short).mean())
        sma_medium = df.get(f'sma_{medium}', df['close'].rolling(medium).mean())
        sma_long = df.get(f'sma_{long_}', df['close'].rolling(long_).mean())
        
        # Golden Cross (단기 > 장기 상향 교차)
        golden_cross = self.crossover(sma_short, sma_long)
        signals[golden_cross] += self.signals_config.get('golden_cross', 1.0)
        
        # Death Cross (단기 < 장기 하향 교차)
        death_cross = self.crossunder(sma_short, sma_long)
        signals[death_cross] += self.signals_config.get('death_cross', -1.0)
        
        # 가격이 MA 위/아래
        price_above = df['close'] > sma_medium
        price_below = df['close'] < sma_medium
        
        signals[price_above] += self.signals_config.get('price_above_ma', 0.5)
        signals[price_below] += self.signals_config.get('price_below_ma', -0.5)
        
        # 정규화 (-1 ~ 1)
        return signals.clip(-1, 1)


class EMAIndicator(BaseIndicator):
    """Exponential Moving Average 지표"""
    
    def __init__(self, config: Dict[str, Any]):
        params = config.get('params', {})
        super().__init__(
            name='EMA',
            params=params,
            weight=config.get('weight', 1.2),
            enabled=config.get('enabled', True)
        )
        self.signals_config = config.get('signals', {})
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMA 계산"""
        result = df.copy()
        
        short = self.params.get('short_period', 12)
        medium = self.params.get('medium_period', 26)
        long_ = self.params.get('long_period', 50)
        
        result[f'ema_{short}'] = df['close'].ewm(span=short, adjust=False).mean()
        result[f'ema_{medium}'] = df['close'].ewm(span=medium, adjust=False).mean()
        result[f'ema_{long_}'] = df['close'].ewm(span=long_, adjust=False).mean()
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """EMA 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        short = self.params.get('short_period', 12)
        medium = self.params.get('medium_period', 26)
        long_ = self.params.get('long_period', 50)
        
        ema_short = df.get(f'ema_{short}', df['close'].ewm(span=short, adjust=False).mean())
        ema_medium = df.get(f'ema_{medium}', df['close'].ewm(span=medium, adjust=False).mean())
        ema_long = df.get(f'ema_{long_}', df['close'].ewm(span=long_, adjust=False).mean())
        
        # Golden Cross / Death Cross
        golden_cross = self.crossover(ema_short, ema_long)
        death_cross = self.crossunder(ema_short, ema_long)
        
        signals[golden_cross] += self.signals_config.get('golden_cross', 1.0)
        signals[death_cross] += self.signals_config.get('death_cross', -1.0)
        
        # 추세 강도 (EMA 정렬 상태)
        trend_strength = self.signals_config.get('trend_strength_multiplier', 0.3)
        
        # 완벽한 상승 정렬: short > medium > long
        bullish_alignment = (ema_short > ema_medium) & (ema_medium > ema_long)
        bearish_alignment = (ema_short < ema_medium) & (ema_medium < ema_long)
        
        signals[bullish_alignment] += trend_strength
        signals[bearish_alignment] -= trend_strength
        
        return signals.clip(-1, 1)


class MACDIndicator(BaseIndicator):
    """MACD (Moving Average Convergence Divergence) 지표"""
    
    def __init__(self, config: Dict[str, Any]):
        params = config.get('params', {})
        super().__init__(
            name='MACD',
            params=params,
            weight=config.get('weight', 1.5),
            enabled=config.get('enabled', True)
        )
        self.signals_config = config.get('signals', {})
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD 계산"""
        result = df.copy()
        
        fast = self.params.get('fast_period', 12)
        slow = self.params.get('slow_period', 26)
        signal = self.params.get('signal_period', 9)
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        result['macd_line'] = ema_fast - ema_slow
        result['macd_signal'] = result['macd_line'].ewm(span=signal, adjust=False).mean()
        result['macd_histogram'] = result['macd_line'] - result['macd_signal']
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """MACD 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        # 필요한 컬럼 계산
        if 'macd_line' not in df.columns:
            df = self.calculate(df)
        
        macd_line = df['macd_line']
        macd_signal = df['macd_signal']
        macd_hist = df['macd_histogram']
        
        # Bullish crossover (MACD가 Signal 위로)
        bullish_cross = self.crossover(macd_line, macd_signal)
        bearish_cross = self.crossunder(macd_line, macd_signal)
        
        signals[bullish_cross] += self.signals_config.get('bullish_crossover', 1.0)
        signals[bearish_cross] += self.signals_config.get('bearish_crossover', -1.0)
        
        # 히스토그램 상태
        hist_positive = macd_hist > 0
        hist_negative = macd_hist < 0
        
        signals[hist_positive] += self.signals_config.get('histogram_positive', 0.3)
        signals[hist_negative] += self.signals_config.get('histogram_negative', -0.3)
        
        # 다이버전스 감지
        div_mult = self.signals_config.get('divergence_multiplier', 0.5)
        bullish_div, bearish_div = self.detect_divergence(df['close'], macd_hist)
        
        signals[bullish_div] += div_mult
        signals[bearish_div] -= div_mult
        
        return signals.clip(-1, 1)


class ADXIndicator(BaseIndicator):
    """ADX (Average Directional Index) 지표"""
    
    def __init__(self, config: Dict[str, Any]):
        params = config.get('params', {})
        super().__init__(
            name='ADX',
            params=params,
            weight=config.get('weight', 0.8),
            enabled=config.get('enabled', True)
        )
        self.signals_config = config.get('signals', {})
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """ADX 계산"""
        result = df.copy()
        period = self.params.get('period', 14)
        
        # True Range 계산
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = tr.rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        result['adx'] = adx
        result['plus_di'] = plus_di
        result['minus_di'] = minus_di
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """ADX 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        if 'adx' not in df.columns:
            df = self.calculate(df)
        
        adx = df['adx']
        plus_di = df['plus_di']
        minus_di = df['minus_di']
        
        strong_threshold = self.params.get('strong_trend_threshold', 25)
        weak_threshold = self.params.get('weak_trend_threshold', 20)
        
        # 강한 상승 추세: ADX > 25 & +DI > -DI
        strong_uptrend = (adx > strong_threshold) & (plus_di > minus_di)
        strong_downtrend = (adx > strong_threshold) & (minus_di > plus_di)
        no_trend = adx < weak_threshold
        
        signals[strong_uptrend] += self.signals_config.get('strong_uptrend', 0.5)
        signals[strong_downtrend] += self.signals_config.get('strong_downtrend', -0.5)
        signals[no_trend] += self.signals_config.get('no_trend', 0.0)
        
        return signals.clip(-1, 1)
