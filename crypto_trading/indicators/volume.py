"""
Volume Indicators
거래량 지표 모듈 - OBV, Volume MA, MFI, VWAP
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from .base import BaseIndicator


class VolumeIndicators:
    """
    거래량 관련 기술적 지표 모음
    
    포함 지표:
    - OBV (On-Balance Volume)
    - Volume MA
    - MFI (Money Flow Index)
    - VWAP (Volume Weighted Average Price)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.indicators = {}
        self._initialize_indicators()
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'obv': {
                'enabled': True,
                'weight': 0.9,
                'params': {'ma_period': 20},
                'signals': {'bullish_divergence': 0.7, 'bearish_divergence': -0.7,
                           'trend_confirmation': 0.4}
            },
            'volume_ma': {
                'enabled': True,
                'weight': 0.8,
                'params': {'short_period': 10, 'long_period': 20, 'spike_threshold': 2.0},
                'signals': {'volume_spike_up': 0.5, 'volume_spike_down': -0.5,
                           'volume_dry_up': -0.2}
            },
            'mfi': {
                'enabled': True,
                'weight': 0.9,
                'params': {'period': 14, 'overbought': 80, 'oversold': 20},
                'signals': {'oversold': 0.8, 'overbought': -0.8, 'divergence': 0.6}
            },
            'vwap': {
                'enabled': True,
                'weight': 0.7,
                'params': {'period': 20},
                'signals': {'price_above_vwap': 0.4, 'price_below_vwap': -0.4}
            }
        }
    
    def _initialize_indicators(self):
        self.indicators = {
            'obv': OBVIndicator(self.config.get('obv', {})),
            'volume_ma': VolumeMAIndicator(self.config.get('volume_ma', {})),
            'mfi': MFIIndicator(self.config.get('mfi', {})),
            'vwap': VWAPIndicator(self.config.get('vwap', {}))
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


class OBVIndicator(BaseIndicator):
    """OBV (On-Balance Volume) 지표"""
    
    def __init__(self, config: Dict[str, Any]):
        params = config.get('params', {})
        super().__init__(
            name='OBV',
            params=params,
            weight=config.get('weight', 0.9),
            enabled=config.get('enabled', True)
        )
        self.signals_config = config.get('signals', {})
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """OBV 계산"""
        result = df.copy()
        ma_period = self.params.get('ma_period', 20)
        
        # OBV 계산
        close_diff = df['close'].diff()
        volume = df['volume']
        
        obv = pd.Series(0, index=df.index, dtype=float)
        obv[close_diff > 0] = volume[close_diff > 0]
        obv[close_diff < 0] = -volume[close_diff < 0]
        
        result['obv'] = obv.cumsum()
        result['obv_ma'] = result['obv'].rolling(window=ma_period).mean()
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """OBV 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        if 'obv' not in df.columns:
            df = self.calculate(df)
        
        obv = df['obv']
        obv_ma = df['obv_ma']
        close = df['close']
        
        # OBV 추세 확인
        obv_uptrend = obv > obv_ma
        obv_downtrend = obv < obv_ma
        
        price_uptrend = close > close.shift(5)
        price_downtrend = close < close.shift(5)
        
        # 추세 확인 (가격과 OBV 방향 일치)
        trend_confirm_bull = obv_uptrend & price_uptrend
        trend_confirm_bear = obv_downtrend & price_downtrend
        
        signals[trend_confirm_bull] += self.signals_config.get('trend_confirmation', 0.4)
        signals[trend_confirm_bear] -= self.signals_config.get('trend_confirmation', 0.4)
        
        # 다이버전스
        bullish_div, bearish_div = self.detect_divergence(close, obv)
        
        signals[bullish_div] += self.signals_config.get('bullish_divergence', 0.7)
        signals[bearish_div] += self.signals_config.get('bearish_divergence', -0.7)
        
        return signals.clip(-1, 1)


class VolumeMAIndicator(BaseIndicator):
    """Volume MA 지표"""
    
    def __init__(self, config: Dict[str, Any]):
        params = config.get('params', {})
        super().__init__(
            name='Volume MA',
            params=params,
            weight=config.get('weight', 0.8),
            enabled=config.get('enabled', True)
        )
        self.signals_config = config.get('signals', {})
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume MA 계산"""
        result = df.copy()
        
        short_period = self.params.get('short_period', 10)
        long_period = self.params.get('long_period', 20)
        
        result['vol_ma_short'] = df['volume'].rolling(window=short_period).mean()
        result['vol_ma_long'] = df['volume'].rolling(window=long_period).mean()
        result['vol_ratio'] = df['volume'] / result['vol_ma_long']
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Volume MA 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        if 'vol_ma_long' not in df.columns:
            df = self.calculate(df)
        
        volume = df['volume']
        vol_ma = df['vol_ma_long']
        vol_ratio = df['vol_ratio']
        close = df['close']
        
        spike_threshold = self.params.get('spike_threshold', 2.0)
        
        # 거래량 급증 + 가격 상승 → 매수 신호
        volume_spike = vol_ratio > spike_threshold
        price_up = close > close.shift(1)
        price_down = close < close.shift(1)
        
        spike_up = volume_spike & price_up
        spike_down = volume_spike & price_down
        
        signals[spike_up] += self.signals_config.get('volume_spike_up', 0.5)
        signals[spike_down] += self.signals_config.get('volume_spike_down', -0.5)
        
        # 거래량 급감 (추세 약화 신호)
        volume_dry = vol_ratio < 0.5
        signals[volume_dry] += self.signals_config.get('volume_dry_up', -0.2)
        
        return signals.clip(-1, 1)


class MFIIndicator(BaseIndicator):
    """MFI (Money Flow Index) 지표"""
    
    def __init__(self, config: Dict[str, Any]):
        params = config.get('params', {})
        super().__init__(
            name='MFI',
            params=params,
            weight=config.get('weight', 0.9),
            enabled=config.get('enabled', True)
        )
        self.signals_config = config.get('signals', {})
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """MFI 계산"""
        result = df.copy()
        period = self.params.get('period', 14)
        
        # Typical Price
        tp = (df['high'] + df['low'] + df['close']) / 3
        
        # Raw Money Flow
        raw_mf = tp * df['volume']
        
        # Money Flow Direction
        tp_diff = tp.diff()
        positive_mf = raw_mf.where(tp_diff > 0, 0)
        negative_mf = raw_mf.where(tp_diff < 0, 0)
        
        # Money Flow Ratio
        positive_mf_sum = positive_mf.rolling(window=period).sum()
        negative_mf_sum = negative_mf.rolling(window=period).sum()
        
        mf_ratio = positive_mf_sum / (negative_mf_sum + 1e-10)
        
        result['mfi'] = 100 - (100 / (1 + mf_ratio))
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """MFI 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        if 'mfi' not in df.columns:
            df = self.calculate(df)
        
        mfi = df['mfi']
        close = df['close']
        
        overbought = self.params.get('overbought', 80)
        oversold = self.params.get('oversold', 20)
        
        # 과매도 → 매수
        oversold_zone = mfi < oversold
        signals[oversold_zone] += self.signals_config.get('oversold', 0.8)
        
        # 과매수 → 매도
        overbought_zone = mfi > overbought
        signals[overbought_zone] += self.signals_config.get('overbought', -0.8)
        
        # 다이버전스
        bullish_div, bearish_div = self.detect_divergence(close, mfi)
        div_signal = self.signals_config.get('divergence', 0.6)
        
        signals[bullish_div] += div_signal
        signals[bearish_div] -= div_signal
        
        return signals.clip(-1, 1)


class VWAPIndicator(BaseIndicator):
    """VWAP (Volume Weighted Average Price) 지표"""
    
    def __init__(self, config: Dict[str, Any]):
        params = config.get('params', {})
        super().__init__(
            name='VWAP',
            params=params,
            weight=config.get('weight', 0.7),
            enabled=config.get('enabled', True)
        )
        self.signals_config = config.get('signals', {})
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """VWAP 계산 (Rolling VWAP)"""
        result = df.copy()
        period = self.params.get('period', 20)
        
        # Typical Price
        tp = (df['high'] + df['low'] + df['close']) / 3
        
        # Rolling VWAP
        tp_vol = tp * df['volume']
        result['vwap'] = tp_vol.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """VWAP 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        if 'vwap' not in df.columns:
            df = self.calculate(df)
        
        close = df['close']
        vwap = df['vwap']
        
        # 가격이 VWAP 위 → 매수 우세
        price_above = close > vwap
        signals[price_above] += self.signals_config.get('price_above_vwap', 0.4)
        
        # 가격이 VWAP 아래 → 매도 우세
        price_below = close < vwap
        signals[price_below] += self.signals_config.get('price_below_vwap', -0.4)
        
        return signals.clip(-1, 1)
