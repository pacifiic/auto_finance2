"""
Volatility Indicators
변동성 지표 모듈 - Bollinger Bands, ATR, Keltner Channel
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from .base import BaseIndicator


class VolatilityIndicators:
    """
    변동성 관련 기술적 지표 모음
    
    포함 지표:
    - Bollinger Bands
    - ATR (Average True Range)
    - Keltner Channel
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.indicators = {}
        self._initialize_indicators()
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'bollinger': {
                'enabled': True,
                'weight': 1.2,
                'params': {'period': 20, 'std_dev': 2.0, 'squeeze_threshold': 0.05},
                'signals': {'lower_band_touch': 0.8, 'upper_band_touch': -0.8,
                           'squeeze_breakout_up': 1.0, 'squeeze_breakout_down': -1.0,
                           'mean_reversion': 0.3}
            },
            'atr': {
                'enabled': True,
                'weight': 0.6,
                'params': {'period': 14, 'multiplier': 2.0},
                'signals': {'high_volatility_caution': -0.2, 
                           'low_volatility_opportunity': 0.2}
            },
            'keltner': {
                'enabled': True,
                'weight': 0.7,
                'params': {'ema_period': 20, 'atr_period': 10, 'multiplier': 2.0},
                'signals': {'lower_band_touch': 0.6, 'upper_band_touch': -0.6,
                           'channel_breakout': 0.8}
            }
        }
    
    def _initialize_indicators(self):
        self.indicators = {
            'bollinger': BollingerBandsIndicator(self.config.get('bollinger', {})),
            'atr': ATRIndicator(self.config.get('atr', {})),
            'keltner': KeltnerChannelIndicator(self.config.get('keltner', {}))
        }
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for name, indicator in self.indicators.items():
            if indicator.enabled:
                result = indicator.calculate(result)
        return result
    
    def generate_all_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        signals = {}
        for name, indicator in self.indicators.items():
            if indicator.enabled:
                signals[name] = indicator.get_signal_with_weight(df)
        return signals
    
    def get_combined_signal(self, df: pd.DataFrame) -> pd.Series:
        signals = self.generate_all_signals(df)
        if not signals:
            return pd.Series(0, index=df.index)
        
        combined = pd.concat(signals.values(), axis=1)
        total_weight = sum(ind.weight for ind in self.indicators.values() if ind.enabled)
        
        return combined.sum(axis=1) / total_weight if total_weight > 0 else combined.sum(axis=1)


class BollingerBandsIndicator(BaseIndicator):
    """Bollinger Bands 지표"""
    
    def __init__(self, config: Dict[str, Any]):
        params = config.get('params', {})
        super().__init__(
            name='Bollinger Bands',
            params=params,
            weight=config.get('weight', 1.2),
            enabled=config.get('enabled', True)
        )
        self.signals_config = config.get('signals', {})
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands 계산"""
        result = df.copy()
        
        period = self.params.get('period', 20)
        std_dev = self.params.get('std_dev', 2.0)
        
        result['bb_middle'] = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        
        result['bb_upper'] = result['bb_middle'] + (rolling_std * std_dev)
        result['bb_lower'] = result['bb_middle'] - (rolling_std * std_dev)
        
        # 밴드 폭 (Bandwidth)
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        
        # %B (가격의 밴드 내 위치)
        result['bb_percent'] = (df['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'] + 1e-10)
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Bollinger Bands 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        if 'bb_middle' not in df.columns:
            df = self.calculate(df)
        
        close = df['close']
        bb_upper = df['bb_upper']
        bb_lower = df['bb_lower']
        bb_middle = df['bb_middle']
        bb_width = df['bb_width']
        
        squeeze_threshold = self.params.get('squeeze_threshold', 0.05)
        
        # 하단 밴드 터치 → 매수 고려
        lower_touch = close <= bb_lower
        signals[lower_touch] += self.signals_config.get('lower_band_touch', 0.8)
        
        # 상단 밴드 터치 → 매도 고려
        upper_touch = close >= bb_upper
        signals[upper_touch] += self.signals_config.get('upper_band_touch', -0.8)
        
        # 스퀴즈 감지 (밴드 폭이 좁아짐)
        is_squeeze = bb_width < squeeze_threshold
        squeeze_end = is_squeeze.shift(1) & ~is_squeeze
        
        # 스퀴즈 후 상방/하방 돌파
        breakout_up = squeeze_end & (close > bb_middle)
        breakout_down = squeeze_end & (close < bb_middle)
        
        signals[breakout_up] += self.signals_config.get('squeeze_breakout_up', 1.0)
        signals[breakout_down] += self.signals_config.get('squeeze_breakout_down', -1.0)
        
        # 중앙선 회귀 신호
        far_from_mean = abs(close - bb_middle) / bb_middle > 0.02
        moving_to_mean = (
            ((close > bb_middle) & (close < close.shift(1))) |
            ((close < bb_middle) & (close > close.shift(1)))
        )
        mean_reversion = far_from_mean & moving_to_mean
        
        # 중앙으로 회귀 시 방향에 따른 신호
        reversion_signal = np.where(close > bb_middle, -1, 1) * self.signals_config.get('mean_reversion', 0.3)
        signals[mean_reversion] += reversion_signal[mean_reversion]
        
        return signals.clip(-1, 1)


class ATRIndicator(BaseIndicator):
    """ATR (Average True Range) 지표"""
    
    def __init__(self, config: Dict[str, Any]):
        params = config.get('params', {})
        super().__init__(
            name='ATR',
            params=params,
            weight=config.get('weight', 0.6),
            enabled=config.get('enabled', True)
        )
        self.signals_config = config.get('signals', {})
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR 계산"""
        result = df.copy()
        period = self.params.get('period', 14)
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result['atr'] = tr.rolling(window=period).mean()
        
        # ATR 백분율 (가격 대비)
        result['atr_percent'] = result['atr'] / close * 100
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """ATR 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        if 'atr' not in df.columns:
            df = self.calculate(df)
        
        atr_pct = df['atr_percent']
        
        # 평균 ATR%
        avg_atr_pct = atr_pct.rolling(window=50).mean()
        
        # 높은 변동성 (평균의 1.5배 이상) → 주의
        high_volatility = atr_pct > avg_atr_pct * 1.5
        signals[high_volatility] += self.signals_config.get('high_volatility_caution', -0.2)
        
        # 낮은 변동성 (평균의 0.5배 이하) → 기회 (돌파 대기)
        low_volatility = atr_pct < avg_atr_pct * 0.5
        signals[low_volatility] += self.signals_config.get('low_volatility_opportunity', 0.2)
        
        return signals.clip(-1, 1)
    
    def get_stop_loss_level(self, df: pd.DataFrame, position_type: str = 'long') -> pd.Series:
        """
        ATR 기반 손절가 계산
        
        Args:
            df: 데이터프레임
            position_type: 'long' 또는 'short'
            
        Returns:
            손절가 시리즈
        """
        if 'atr' not in df.columns:
            df = self.calculate(df)
        
        multiplier = self.params.get('multiplier', 2.0)
        atr = df['atr']
        close = df['close']
        
        if position_type == 'long':
            return close - (atr * multiplier)
        else:
            return close + (atr * multiplier)


class KeltnerChannelIndicator(BaseIndicator):
    """Keltner Channel 지표"""
    
    def __init__(self, config: Dict[str, Any]):
        params = config.get('params', {})
        super().__init__(
            name='Keltner Channel',
            params=params,
            weight=config.get('weight', 0.7),
            enabled=config.get('enabled', True)
        )
        self.signals_config = config.get('signals', {})
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keltner Channel 계산"""
        result = df.copy()
        
        ema_period = self.params.get('ema_period', 20)
        atr_period = self.params.get('atr_period', 10)
        multiplier = self.params.get('multiplier', 2.0)
        
        # EMA 중심선
        result['kc_middle'] = df['close'].ewm(span=ema_period, adjust=False).mean()
        
        # ATR 계산
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()
        
        result['kc_upper'] = result['kc_middle'] + (atr * multiplier)
        result['kc_lower'] = result['kc_middle'] - (atr * multiplier)
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Keltner Channel 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        if 'kc_middle' not in df.columns:
            df = self.calculate(df)
        
        close = df['close']
        kc_upper = df['kc_upper']
        kc_lower = df['kc_lower']
        
        # 하단 밴드 터치 → 매수
        lower_touch = close <= kc_lower
        signals[lower_touch] += self.signals_config.get('lower_band_touch', 0.6)
        
        # 상단 밴드 터치 → 매도
        upper_touch = close >= kc_upper
        signals[upper_touch] += self.signals_config.get('upper_band_touch', -0.6)
        
        # 채널 돌파
        breakout_up = self.crossover(close, kc_upper)
        breakout_down = self.crossunder(close, kc_lower)
        
        signals[breakout_up] += self.signals_config.get('channel_breakout', 0.8)
        signals[breakout_down] -= self.signals_config.get('channel_breakout', 0.8)
        
        return signals.clip(-1, 1)
