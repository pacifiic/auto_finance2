"""
Test Strategy Engine
전략 엔진 테스트
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading.strategy import StrategyEngine, HyperParameterManager
from crypto_trading.utils import generate_sample_data


@pytest.fixture
def sample_data():
    """테스트용 샘플 데이터"""
    return generate_sample_data(n_samples=200, seed=42)


class TestHyperParameterManager:
    """하이퍼파라미터 관리자 테스트"""
    
    def test_initialization(self):
        """초기화 테스트"""
        hp = HyperParameterManager()
        assert hp is not None
        assert hp.params is not None
    
    def test_get_parameter(self):
        """파라미터 가져오기 테스트"""
        hp = HyperParameterManager()
        
        rsi_period = hp.get('momentum.rsi.period')
        assert rsi_period == 14
        
        default = hp.get('nonexistent.param', 999)
        assert default == 999
    
    def test_set_parameter(self):
        """파라미터 설정 테스트"""
        hp = HyperParameterManager()
        
        hp.set('momentum.rsi.period', 21)
        assert hp.get('momentum.rsi.period') == 21
    
    def test_apply_preset(self):
        """프리셋 적용 테스트"""
        hp = HyperParameterManager()
        
        hp.apply_preset('aggressive')
        assert hp.get('momentum.rsi.period') == 7
        assert hp.get('momentum.rsi.overbought') == 75
    
    def test_list_presets(self):
        """프리셋 목록 테스트"""
        hp = HyperParameterManager()
        
        presets = hp.list_presets()
        assert len(presets) >= 4
        
        preset_names = [p['name'] for p in presets]
        assert 'aggressive' in preset_names
        assert 'conservative' in preset_names
        assert 'swing' in preset_names
    
    def test_get_all_weights(self):
        """모든 가중치 가져오기 테스트"""
        hp = HyperParameterManager()
        
        weights = hp.get_all_weights()
        assert len(weights) > 0
        assert 'category.trend' in weights


class TestStrategyEngine:
    """전략 엔진 테스트"""
    
    def test_initialization(self):
        """초기화 테스트"""
        engine = StrategyEngine()
        assert engine is not None
        assert engine.signal_generator is not None
        assert engine.signal_combiner is not None
    
    def test_analyze(self, sample_data):
        """분석 테스트"""
        engine = StrategyEngine()
        analysis = engine.analyze(sample_data)
        
        assert 'current_price' in analysis
        assert 'signal' in analysis
        assert 'signals_by_category' in analysis
        assert 'recommendation' in analysis
        assert 'risk_parameters' in analysis
        
        # 신호 값 범위 확인
        assert -1 <= analysis['signal']['value'] <= 1
        
        # 가격이 양수인지 확인
        assert analysis['current_price'] > 0
    
    def test_generate_trade_signal(self, sample_data):
        """거래 신호 생성 테스트"""
        engine = StrategyEngine()
        signal = engine.generate_trade_signal(sample_data)
        
        assert signal is not None
        assert signal.price > 0
        assert -1 <= signal.signal_value <= 1
        assert signal.stop_loss < signal.price
        assert signal.take_profit > signal.price
    
    def test_apply_preset(self, sample_data):
        """프리셋 적용 테스트"""
        engine = StrategyEngine()
        
        # 기본 분석
        analysis1 = engine.analyze(sample_data)
        
        # 공격적 프리셋 적용
        engine.apply_preset('aggressive')
        analysis2 = engine.analyze(sample_data)
        
        # 결과가 다를 수 있음 (파라미터가 다르므로)
        assert analysis1 is not None
        assert analysis2 is not None
    
    def test_update_hyperparameters(self, sample_data):
        """하이퍼파라미터 업데이트 테스트"""
        engine = StrategyEngine()
        
        engine.update_hyperparameters({
            'momentum.rsi.period': 21
        })
        
        assert engine.hp_manager.get('momentum.rsi.period') == 21
    
    def test_backtest_signal(self, sample_data):
        """백테스트 신호 테스트"""
        engine = StrategyEngine()
        result = engine.backtest_signal(sample_data)
        
        assert 'signal' in result.columns
        assert 'signal_type' in result.columns
        assert 'position' in result.columns
        
        # position은 -1, 0, 1 중 하나
        assert set(result['position'].unique()).issubset({-1, 0, 1})
    
    def test_get_strategy_summary(self):
        """전략 요약 테스트"""
        engine = StrategyEngine()
        summary = engine.get_strategy_summary()
        
        assert 'combination_method' in summary
        assert 'category_weights' in summary
        assert 'thresholds' in summary
        assert 'risk_management' in summary
        assert 'enabled_indicators' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
