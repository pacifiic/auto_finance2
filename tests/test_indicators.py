"""
Test Indicators
기술적 지표 테스트
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading.indicators import TrendIndicators, MomentumIndicators, VolatilityIndicators, VolumeIndicators
from crypto_trading.utils import generate_sample_data


@pytest.fixture
def sample_data():
    """테스트용 샘플 데이터"""
    return generate_sample_data(n_samples=100, seed=42)


class TestTrendIndicators:
    """추세 지표 테스트"""
    
    def test_initialization(self):
        """초기화 테스트"""
        trend = TrendIndicators()
        assert trend is not None
        assert 'sma' in trend.indicators
        assert 'ema' in trend.indicators
        assert 'macd' in trend.indicators
        assert 'adx' in trend.indicators
    
    def test_calculate_all(self, sample_data):
        """모든 지표 계산 테스트"""
        trend = TrendIndicators()
        result = trend.calculate_all(sample_data)
        
        # SMA 컬럼 확인
        assert 'sma_20' in result.columns
        assert 'sma_50' in result.columns
        assert 'sma_200' in result.columns
        
        # MACD 컬럼 확인
        assert 'macd_line' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_histogram' in result.columns
    
    def test_generate_signals(self, sample_data):
        """신호 생성 테스트"""
        trend = TrendIndicators()
        signals = trend.generate_all_signals(sample_data)
        
        assert len(signals) > 0
        for name, signal in signals.items():
            assert len(signal) == len(sample_data)
            assert signal.min() >= -1
            assert signal.max() <= 1
    
    def test_combined_signal(self, sample_data):
        """조합 신호 테스트"""
        trend = TrendIndicators()
        combined = trend.get_combined_signal(sample_data)
        
        assert len(combined) == len(sample_data)
        assert combined.min() >= -1
        assert combined.max() <= 1


class TestMomentumIndicators:
    """모멘텀 지표 테스트"""
    
    def test_rsi_calculation(self, sample_data):
        """RSI 계산 테스트"""
        momentum = MomentumIndicators()
        result = momentum.calculate_all(sample_data)
        
        assert 'rsi' in result.columns
        # RSI는 0-100 범위
        valid_rsi = result['rsi'].dropna()
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100
    
    def test_stochastic_calculation(self, sample_data):
        """Stochastic 계산 테스트"""
        momentum = MomentumIndicators()
        result = momentum.calculate_all(sample_data)
        
        assert 'stoch_k' in result.columns
        assert 'stoch_d' in result.columns


class TestVolatilityIndicators:
    """변동성 지표 테스트"""
    
    def test_bollinger_bands(self, sample_data):
        """볼린저 밴드 테스트"""
        vol = VolatilityIndicators()
        result = vol.calculate_all(sample_data)
        
        assert 'bb_upper' in result.columns
        assert 'bb_middle' in result.columns
        assert 'bb_lower' in result.columns
        
        # upper > middle > lower
        valid_idx = result['bb_upper'].notna()
        assert (result.loc[valid_idx, 'bb_upper'] >= result.loc[valid_idx, 'bb_middle']).all()
        assert (result.loc[valid_idx, 'bb_middle'] >= result.loc[valid_idx, 'bb_lower']).all()
    
    def test_atr_calculation(self, sample_data):
        """ATR 계산 테스트"""
        vol = VolatilityIndicators()
        result = vol.calculate_all(sample_data)
        
        assert 'atr' in result.columns
        valid_atr = result['atr'].dropna()
        assert (valid_atr >= 0).all()


class TestVolumeIndicators:
    """거래량 지표 테스트"""
    
    def test_obv_calculation(self, sample_data):
        """OBV 계산 테스트"""
        volume = VolumeIndicators()
        result = volume.calculate_all(sample_data)
        
        assert 'obv' in result.columns
        assert 'obv_ma' in result.columns
    
    def test_mfi_calculation(self, sample_data):
        """MFI 계산 테스트"""
        volume = VolumeIndicators()
        result = volume.calculate_all(sample_data)
        
        assert 'mfi' in result.columns
        valid_mfi = result['mfi'].dropna()
        assert valid_mfi.min() >= 0
        assert valid_mfi.max() <= 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
