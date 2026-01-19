"""
Signal Generator
신호 생성 엔진

모든 기술적 지표와 패턴에서 신호를 수집하고 통합합니다.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import yaml

from ..indicators import TrendIndicators, MomentumIndicators, VolatilityIndicators, VolumeIndicators
from ..patterns import CandlestickPatterns, ChartPatterns
from ..patterns.chart_patterns import SupportResistance, FibonacciLevels


class SignalGenerator:
    """
    신호 생성 엔진
    
    모든 분석 모듈에서 신호를 수집하고 카테고리별로 정리합니다.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config_path: YAML 설정 파일 경로
            config: 직접 전달하는 설정 딕셔너리
        """
        self.config = self._load_config(config_path, config)
        self._initialize_modules()
    
    def _load_config(self, config_path: Optional[str], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """설정 로드"""
        if config:
            return config
        
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # 기본 설정
        return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'trend_indicators': {},
            'momentum_indicators': {},
            'volatility_indicators': {},
            'volume_indicators': {},
            'candlestick_patterns': {},
            'chart_patterns': {},
            'support_resistance': {},
            'fibonacci': {},
            'signal_combination': {
                'category_weights': {
                    'trend': 1.2,
                    'momentum': 1.0,
                    'volatility': 0.8,
                    'volume': 0.9,
                    'candlestick': 1.1,
                    'chart_pattern': 1.0,
                    'support_resistance': 1.0,
                    'fibonacci': 0.7
                }
            }
        }
    
    def _initialize_modules(self):
        """분석 모듈 초기화"""
        # 기술적 지표
        self.trend = TrendIndicators(self.config.get('trend_indicators'))
        self.momentum = MomentumIndicators(self.config.get('momentum_indicators'))
        self.volatility = VolatilityIndicators(self.config.get('volatility_indicators'))
        self.volume = VolumeIndicators(self.config.get('volume_indicators'))
        
        # 패턴 인식
        self.candlestick = CandlestickPatterns(self.config.get('candlestick_patterns'))
        self.chart_patterns = ChartPatterns(self.config.get('chart_patterns'))
        
        # 지지/저항 및 피보나치
        self.support_resistance = SupportResistance(self.config.get('support_resistance'))
        self.fibonacci = FibonacciLevels(self.config.get('fibonacci'))
        
        # 카테고리 가중치
        self.category_weights = self.config.get('signal_combination', {}).get('category_weights', {})
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술적 지표 계산
        
        Args:
            df: OHLCV 데이터프레임 (open, high, low, close, volume 컬럼 필요)
            
        Returns:
            모든 지표가 추가된 데이터프레임
        """
        result = df.copy()
        
        # 지표 계산
        result = self.trend.calculate_all(result)
        result = self.momentum.calculate_all(result)
        result = self.volatility.calculate_all(result)
        result = self.volume.calculate_all(result)
        
        # 패턴 감지
        result = self.candlestick.detect_all_patterns(result)
        result = self.chart_patterns.detect_all_patterns(result)
        
        return result
    
    def generate_signals_by_category(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        카테고리별 신호 생성
        
        Returns:
            {카테고리명: 신호 시리즈} 딕셔너리
        """
        # 먼저 지표 계산
        df_with_indicators = self.calculate_all_indicators(df)
        
        signals = {}
        
        # 추세 신호
        signals['trend'] = self.trend.get_combined_signal(df_with_indicators)
        
        # 모멘텀 신호
        signals['momentum'] = self.momentum.get_combined_signal(df_with_indicators)
        
        # 변동성 신호
        signals['volatility'] = self.volatility.get_combined_signal(df_with_indicators)
        
        # 거래량 신호
        signals['volume'] = self.volume.get_combined_signal(df_with_indicators)
        
        # 캔들스틱 패턴 신호
        signals['candlestick'] = self.candlestick.generate_signals(df_with_indicators)
        
        # 차트 패턴 신호
        signals['chart_pattern'] = self.chart_patterns.generate_signals(df_with_indicators)
        
        # 지지/저항 신호
        signals['support_resistance'] = self.support_resistance.generate_signals(df_with_indicators)
        
        # 피보나치 신호
        signals['fibonacci'] = self.fibonacci.generate_signals(df_with_indicators)
        
        return signals
    
    def generate_detailed_signals(self, df: pd.DataFrame) -> Dict[str, Dict[str, pd.Series]]:
        """
        상세 신호 생성 (각 지표별)
        
        Returns:
            {카테고리: {지표명: 신호 시리즈}} 중첩 딕셔너리
        """
        df_with_indicators = self.calculate_all_indicators(df)
        
        detailed = {}
        
        # 추세 지표별 신호
        detailed['trend'] = self.trend.generate_all_signals(df_with_indicators)
        
        # 모멘텀 지표별 신호
        detailed['momentum'] = self.momentum.generate_all_signals(df_with_indicators)
        
        # 변동성 지표별 신호
        detailed['volatility'] = self.volatility.generate_all_signals(df_with_indicators)
        
        # 거래량 지표별 신호
        detailed['volume'] = self.volume.generate_all_signals(df_with_indicators)
        
        return detailed
    
    def get_signal_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        신호 요약 테이블 생성
        
        Returns:
            각 시점별 신호 요약 데이터프레임
        """
        signals = self.generate_signals_by_category(df)
        
        summary = pd.DataFrame(index=df.index)
        
        for category, signal in signals.items():
            summary[f'signal_{category}'] = signal
            
            # 카테고리 가중치 적용
            weight = self.category_weights.get(category, 1.0)
            summary[f'weighted_{category}'] = signal * weight
        
        # 가중 평균 계산
        weighted_cols = [col for col in summary.columns if col.startswith('weighted_')]
        total_weight = sum(self.category_weights.get(cat, 1.0) for cat in signals.keys())
        
        summary['combined_signal'] = summary[weighted_cols].sum(axis=1) / total_weight
        
        # 신호 분류
        summary['signal_type'] = pd.cut(
            summary['combined_signal'],
            bins=[-np.inf, -0.7, -0.3, -0.1, 0.1, 0.3, 0.7, np.inf],
            labels=['strong_sell', 'sell', 'weak_sell', 'neutral', 'weak_buy', 'buy', 'strong_buy']
        )
        
        return summary
    
    def get_current_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        현재 시점의 신호 반환
        
        Returns:
            현재 신호 정보 딕셔너리
        """
        summary = self.get_signal_summary(df)
        latest = summary.iloc[-1]
        
        signals_by_category = {
            cat: float(latest[f'signal_{cat}']) 
            for cat in self.generate_signals_by_category(df).keys()
        }
        
        return {
            'timestamp': df.index[-1],
            'price': float(df['close'].iloc[-1]),
            'combined_signal': float(latest['combined_signal']),
            'signal_type': str(latest['signal_type']),
            'signals_by_category': signals_by_category,
            'recommendation': self._get_recommendation(float(latest['combined_signal']))
        }
    
    def _get_recommendation(self, signal: float) -> str:
        """신호값에 따른 추천 반환"""
        if signal >= 0.7:
            return "강력 매수 - 여러 지표가 상승을 강하게 시사"
        elif signal >= 0.3:
            return "매수 - 상승 신호 우세"
        elif signal >= 0.1:
            return "약한 매수 - 상승 가능성 있음, 추가 확인 필요"
        elif signal >= -0.1:
            return "중립 - 방향성 불분명, 관망 권장"
        elif signal >= -0.3:
            return "약한 매도 - 하락 가능성 있음, 주의 필요"
        elif signal >= -0.7:
            return "매도 - 하락 신호 우세"
        else:
            return "강력 매도 - 여러 지표가 하락을 강하게 시사"
    
    def get_confirming_indicators(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        방향이 일치하는 지표 목록 반환
        
        Returns:
            {'bullish': [...], 'bearish': [...], 'neutral': [...]}
        """
        detailed = self.generate_detailed_signals(df)
        
        bullish = []
        bearish = []
        neutral = []
        
        for category, indicators in detailed.items():
            for indicator_name, signal_series in indicators.items():
                latest_signal = signal_series.iloc[-1] if len(signal_series) > 0 else 0
                
                if latest_signal > 0.1:
                    bullish.append(f"{category}/{indicator_name}")
                elif latest_signal < -0.1:
                    bearish.append(f"{category}/{indicator_name}")
                else:
                    neutral.append(f"{category}/{indicator_name}")
        
        return {
            'bullish': bullish,
            'bearish': bearish,
            'neutral': neutral
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """설정 업데이트 및 모듈 재초기화"""
        self.config.update(new_config)
        self._initialize_modules()
