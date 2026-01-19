"""
Momentum Indicators
모멘텀 지표 모듈 - RSI, Stochastic, Williams %R, CCI
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from .base import BaseIndicator


class MomentumIndicators:
    """
    모멘텀 관련 기술적 지표 모음
    
    포함 지표:
    - RSI (Relative Strength Index)
    - Stochastic Oscillator
    - Williams %R
    - CCI (Commodity Channel Index)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.indicators = {}
        self._initialize_indicators()
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'rsi': {
                'enabled': True,
                'weight': 1.3,
                'params': {'period': 14, 'overbought': 70, 'oversold': 30,
                          'extreme_overbought': 80, 'extreme_oversold': 20},
                'signals': {'oversold_buy': 1.0, 'overbought_sell': -1.0,
                           'extreme_oversold': 1.5, 'extreme_overbought': -1.5,
                           'neutral_zone': 0.0, 'divergence_multiplier': 0.7}
            },
            'stochastic': {
                'enabled': True,
                'weight': 1.0,
                'params': {'k_period': 14, 'd_period': 3, 'smooth_k': 3,
                          'overbought': 80, 'oversold': 20},
                'signals': {'oversold_crossover': 1.0, 'overbought_crossover': -1.0,
                           'bullish_divergence': 0.8, 'bearish_divergence': -0.8}
            },
            'williams_r': {
                'enabled': True,
                'weight': 0.7,
                'params': {'period': 14, 'overbought': -20, 'oversold': -80},
                'signals': {'oversold': 0.8, 'overbought': -0.8}
            },
            'cci': {
                'enabled': True,
                'weight': 0.8,
                'params': {'period': 20, 'overbought': 100, 'oversold': -100},
                'signals': {'oversold': 0.7, 'overbought': -0.7,
                           'zero_cross_up': 0.3, 'zero_cross_down': -0.3}
            }
        }
    
    def _initialize_indicators(self):
        self.indicators = {
            'rsi': RSIIndicator(self.config.get('rsi', {})),
            'stochastic': StochasticIndicator(self.config.get('stochastic', {})),
            'williams_r': WilliamsRIndicator(self.config.get('williams_r', {})),
            'cci': CCIIndicator(self.config.get('cci', {}))
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


class RSIIndicator(BaseIndicator):
    """RSI (Relative Strength Index) 지표"""
    
    def __init__(self, config: Dict[str, Any]):
        params = config.get('params', {})
        super().__init__(
            name='RSI',
            params=params,
            weight=config.get('weight', 1.3),
            enabled=config.get('enabled', True)
        )
        self.signals_config = config.get('signals', {})
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI 계산"""
        result = df.copy()
        period = self.params.get('period', 14)
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        result['rsi'] = 100 - (100 / (1 + rs))
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """RSI 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        if 'rsi' not in df.columns:
            df = self.calculate(df)
        
        rsi = df['rsi']
        
        overbought = self.params.get('overbought', 70)
        oversold = self.params.get('oversold', 30)
        extreme_overbought = self.params.get('extreme_overbought', 80)
        extreme_oversold = self.params.get('extreme_oversold', 20)
        
        # 과매도 → 매수 신호
        oversold_zone = rsi < oversold
        extreme_oversold_zone = rsi < extreme_oversold
        
        # 과매수 → 매도 신호
        overbought_zone = rsi > overbought
        extreme_overbought_zone = rsi > extreme_overbought
        
        signals[oversold_zone] += self.signals_config.get('oversold_buy', 1.0)
        signals[extreme_oversold_zone] += self.signals_config.get('extreme_oversold', 0.5)
        
        signals[overbought_zone] += self.signals_config.get('overbought_sell', -1.0)
        signals[extreme_overbought_zone] += self.signals_config.get('extreme_overbought', -0.5)
        
        # 다이버전스
        div_mult = self.signals_config.get('divergence_multiplier', 0.7)
        bullish_div, bearish_div = self.detect_divergence(df['close'], rsi)
        
        signals[bullish_div] += div_mult
        signals[bearish_div] -= div_mult
        
        return signals.clip(-1, 1)


class StochasticIndicator(BaseIndicator):
    """Stochastic Oscillator 지표"""
    
    def __init__(self, config: Dict[str, Any]):
        params = config.get('params', {})
        super().__init__(
            name='Stochastic',
            params=params,
            weight=config.get('weight', 1.0),
            enabled=config.get('enabled', True)
        )
        self.signals_config = config.get('signals', {})
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stochastic 계산"""
        result = df.copy()
        
        k_period = self.params.get('k_period', 14)
        d_period = self.params.get('d_period', 3)
        smooth_k = self.params.get('smooth_k', 3)
        
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        # Fast %K
        fast_k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
        
        # Slow %K (smoothed)
        result['stoch_k'] = fast_k.rolling(window=smooth_k).mean()
        
        # %D (signal line)
        result['stoch_d'] = result['stoch_k'].rolling(window=d_period).mean()
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Stochastic 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        if 'stoch_k' not in df.columns:
            df = self.calculate(df)
        
        k = df['stoch_k']
        d = df['stoch_d']
        
        overbought = self.params.get('overbought', 80)
        oversold = self.params.get('oversold', 20)
        
        # 과매도 구간에서 K가 D를 상향 돌파 → 매수
        oversold_zone = k < oversold
        k_cross_d_up = self.crossover(k, d)
        
        oversold_crossover = oversold_zone & k_cross_d_up
        signals[oversold_crossover] += self.signals_config.get('oversold_crossover', 1.0)
        
        # 과매수 구간에서 K가 D를 하향 돌파 → 매도
        overbought_zone = k > overbought
        k_cross_d_down = self.crossunder(k, d)
        
        overbought_crossover = overbought_zone & k_cross_d_down
        signals[overbought_crossover] += self.signals_config.get('overbought_crossover', -1.0)
        
        # 다이버전스
        bullish_div, bearish_div = self.detect_divergence(df['close'], k)
        
        signals[bullish_div] += self.signals_config.get('bullish_divergence', 0.8)
        signals[bearish_div] += self.signals_config.get('bearish_divergence', -0.8)
        
        return signals.clip(-1, 1)


class WilliamsRIndicator(BaseIndicator):
    """Williams %R 지표"""
    
    def __init__(self, config: Dict[str, Any]):
        params = config.get('params', {})
        super().__init__(
            name='Williams %R',
            params=params,
            weight=config.get('weight', 0.7),
            enabled=config.get('enabled', True)
        )
        self.signals_config = config.get('signals', {})
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Williams %R 계산"""
        result = df.copy()
        period = self.params.get('period', 14)
        
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        result['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-10)
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Williams %R 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        if 'williams_r' not in df.columns:
            df = self.calculate(df)
        
        wr = df['williams_r']
        
        overbought = self.params.get('overbought', -20)
        oversold = self.params.get('oversold', -80)
        
        # 과매도 (< -80) → 매수
        oversold_zone = wr < oversold
        signals[oversold_zone] += self.signals_config.get('oversold', 0.8)
        
        # 과매수 (> -20) → 매도
        overbought_zone = wr > overbought
        signals[overbought_zone] += self.signals_config.get('overbought', -0.8)
        
        return signals.clip(-1, 1)


class CCIIndicator(BaseIndicator):
    """CCI (Commodity Channel Index) 지표"""
    
    def __init__(self, config: Dict[str, Any]):
        params = config.get('params', {})
        super().__init__(
            name='CCI',
            params=params,
            weight=config.get('weight', 0.8),
            enabled=config.get('enabled', True)
        )
        self.signals_config = config.get('signals', {})
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """CCI 계산"""
        result = df.copy()
        period = self.params.get('period', 20)
        
        # Typical Price
        tp = (df['high'] + df['low'] + df['close']) / 3
        
        # SMA of TP
        sma_tp = tp.rolling(window=period).mean()
        
        # Mean Deviation
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        
        result['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """CCI 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        if 'cci' not in df.columns:
            df = self.calculate(df)
        
        cci = df['cci']
        
        overbought = self.params.get('overbought', 100)
        oversold = self.params.get('oversold', -100)
        
        # 과매도 → 매수
        oversold_zone = cci < oversold
        signals[oversold_zone] += self.signals_config.get('oversold', 0.7)
        
        # 과매수 → 매도
        overbought_zone = cci > overbought
        signals[overbought_zone] += self.signals_config.get('overbought', -0.7)
        
        # 0 교차
        zero_cross_up = self.crossover(cci, pd.Series(0, index=df.index))
        zero_cross_down = self.crossunder(cci, pd.Series(0, index=df.index))
        
        signals[zero_cross_up] += self.signals_config.get('zero_cross_up', 0.3)
        signals[zero_cross_down] += self.signals_config.get('zero_cross_down', -0.3)
        
        return signals.clip(-1, 1)
