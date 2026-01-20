"""
Strategy Engine
전략 엔진

신호 생성, 포지션 관리, 리스크 관리를 통합하는 핵심 엔진입니다.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np

from ..signals import SignalGenerator, SignalCombiner
from .hyperparameters import HyperParameterManager


class PositionType(Enum):
    """포지션 타입"""
    NONE = "none"
    LONG = "long"
    SHORT = "short"


class SignalType(Enum):
    """신호 타입"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class Position:
    """포지션 정보"""
    type: PositionType = PositionType.NONE
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop: float = 0.0


@dataclass
class TradeSignal:
    """거래 신호"""
    timestamp: datetime
    signal_type: SignalType
    signal_value: float
    price: float
    confidence: float
    reasons: List[str] = field(default_factory=list)
    stop_loss: float = 0.0
    take_profit: float = 0.0


class StrategyEngine:
    """
    전략 엔진
    
    모든 분석과 거래 신호를 통합 관리합니다.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 설정 파일 경로
        """
        # 하이퍼파라미터 관리자
        self.hp_manager = HyperParameterManager(config_path)
        
        # 신호 생성기 및 조합기
        self.signal_generator = SignalGenerator(config=self.hp_manager.to_config_dict())
        self.signal_combiner = SignalCombiner(
            self.hp_manager.params.get('signal_combination', {})
        )
        
        # 현재 포지션
        self.position = Position()
        
        # 거래 이력
        self.trade_history: List[Dict[str, Any]] = []
        
        # 리스크 파라미터
        self.risk_params = self.hp_manager.params.get('risk_management', {})
    
    def _reinitialize(self):
        """하이퍼파라미터 변경 후 모듈 재초기화"""
        self.signal_generator = SignalGenerator(config=self.hp_manager.to_config_dict())
        self.signal_combiner = SignalCombiner(
            self.hp_manager.params.get('signal_combination', {})
        )
        self.risk_params = self.hp_manager.params.get('risk_management', {})
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        종합 분석 수행
        
        Args:
            df: OHLCV 데이터프레임
            
        Returns:
            분석 결과 딕셔너리
        """
        # 모든 지표 계산
        df_with_indicators = self.signal_generator.calculate_all_indicators(df)
        
        # 카테고리별 신호 생성
        signals_by_category = self.signal_generator.generate_signals_by_category(df)
        
        # 신호 조합
        combined_signal = self.signal_combiner.combine(signals_by_category)
        
        # 현재 시점 신호
        current_signal = combined_signal.iloc[-1] if len(combined_signal) > 0 else 0
        
        # 신호 분류
        signal_info = self.signal_combiner.get_signal_strength(current_signal)
        
        # 확인 지표
        confirming = self.signal_generator.get_confirming_indicators(df)
        
        # 지지/저항 레벨
        sr_levels = self.signal_generator.support_resistance.find_levels(df)
        
        # 피보나치 레벨
        fib_levels = self.signal_generator.fibonacci.calculate_levels(df)
        
        return {
            'timestamp': df.index[-1] if hasattr(df.index[-1], 'strftime') else str(df.index[-1]),
            'current_price': float(df['close'].iloc[-1]),
            'signal': {
                'value': float(current_signal),
                'type': signal_info['classification'],
                'strength': signal_info['strength'],
                'direction': signal_info['direction'],
                'confidence': signal_info['confidence']
            },
            'signals_by_category': {k: float(v.iloc[-1]) for k, v in signals_by_category.items()},
            'confirming_indicators': confirming,
            'levels': {
                'support': sr_levels['support'][:3],
                'resistance': sr_levels['resistance'][:3],
                'fibonacci': fib_levels
            },
            'recommendation': self._generate_recommendation(signal_info, confirming),
            'risk_parameters': self._calculate_risk_levels(df)
        }
    
    def generate_trade_signal(self, df: pd.DataFrame) -> TradeSignal:
        """
        거래 신호 생성
        
        Args:
            df: OHLCV 데이터프레임
            
        Returns:
            TradeSignal 객체
        """
        analysis = self.analyze(df)
        
        signal_value = analysis['signal']['value']
        signal_type = SignalType(analysis['signal']['type'])
        
        # 신호 이유 수집
        reasons = self._collect_signal_reasons(analysis)
        
        # 리스크 레벨 계산
        risk_levels = analysis['risk_parameters']
        
        return TradeSignal(
            timestamp=datetime.now(),
            signal_type=signal_type,
            signal_value=signal_value,
            price=analysis['current_price'],
            confidence=analysis['signal']['confidence'],
            reasons=reasons,
            stop_loss=risk_levels['stop_loss'],
            take_profit=risk_levels['take_profit']
        )
    
    def _generate_recommendation(self, signal_info: Dict[str, Any], 
                                  confirming: Dict[str, List[str]]) -> Dict[str, Any]:
        """거래 추천 생성"""
        classification = signal_info['classification']
        direction = signal_info['direction']
        confidence = signal_info['confidence']
        
        # 확인 지표 분석
        bullish_count = len(confirming['bullish'])
        bearish_count = len(confirming['bearish'])
        total_indicators = bullish_count + bearish_count + len(confirming['neutral'])
        
        # 일치도 계산
        if direction == 'bullish':
            agreement = bullish_count / total_indicators if total_indicators > 0 else 0
        elif direction == 'bearish':
            agreement = bearish_count / total_indicators if total_indicators > 0 else 0
        else:
            agreement = 0.5
        
        # 추천 행동
        if classification in ['strong_buy', 'buy']:
            action = 'LONG' if agreement > 0.5 else 'HOLD (확인 대기)'
            urgency = 'HIGH' if classification == 'strong_buy' else 'MEDIUM'
        elif classification in ['strong_sell', 'sell']:
            action = 'SHORT/CLOSE LONG' if agreement > 0.5 else 'HOLD (확인 대기)'
            urgency = 'HIGH' if classification == 'strong_sell' else 'MEDIUM'
        elif classification in ['weak_buy', 'weak_sell']:
            action = 'HOLD (약한 신호)'
            urgency = 'LOW'
        else:
            action = 'HOLD (관망)'
            urgency = 'NONE'
        
        return {
            'action': action,
            'urgency': urgency,
            'confidence': confidence,
            'indicator_agreement': agreement,
            'bullish_indicators': bullish_count,
            'bearish_indicators': bearish_count,
            'reasoning': self._get_reasoning(classification, agreement)
        }
    
    def _get_reasoning(self, classification: str, agreement: float) -> str:
        """추천 이유 생성"""
        reasons = {
            'strong_buy': f"강한 매수 신호. 지표 일치도: {agreement:.1%}",
            'buy': f"매수 신호. 지표 일치도: {agreement:.1%}",
            'weak_buy': "약한 매수 신호. 추가 확인 필요.",
            'neutral': "방향성 불명확. 관망 권장.",
            'weak_sell': "약한 매도 신호. 추가 확인 필요.",
            'sell': f"매도 신호. 지표 일치도: {agreement:.1%}",
            'strong_sell': f"강한 매도 신호. 지표 일치도: {agreement:.1%}"
        }
        return reasons.get(classification, "분석 불가")
    
    def _collect_signal_reasons(self, analysis: Dict[str, Any]) -> List[str]:
        """신호 발생 이유 수집"""
        reasons = []
        
        signals = analysis['signals_by_category']
        confirming = analysis['confirming_indicators']
        
        # 강한 신호를 주는 카테고리
        for category, value in signals.items():
            if abs(value) > 0.5:
                direction = "상승" if value > 0 else "하락"
                reasons.append(f"{category}: {direction} 신호 ({value:.2f})")
        
        # 확인 지표
        if len(confirming['bullish']) >= 3:
            reasons.append(f"다수 지표 상승 확인 ({len(confirming['bullish'])}개)")
        if len(confirming['bearish']) >= 3:
            reasons.append(f"다수 지표 하락 확인 ({len(confirming['bearish'])}개)")
        
        # 지지/저항 근접
        price = analysis['current_price']
        for support in analysis['levels']['support'][:2]:
            if abs(price - support) / price < 0.02:
                reasons.append(f"지지선 근접 ({support:.2f})")
        
        for resistance in analysis['levels']['resistance'][:2]:
            if abs(price - resistance) / price < 0.02:
                reasons.append(f"저항선 근접 ({resistance:.2f})")
        
        return reasons
    
    def _calculate_risk_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """리스크 레벨 계산"""
        current_price = df['close'].iloc[-1]
        
        # ATR 기반 손절/익절
        atr_indicator = self.signal_generator.volatility.indicators.get('atr')
        df_with_atr = atr_indicator.calculate(df) if atr_indicator else df
        
        atr = df_with_atr['atr'].iloc[-1] if 'atr' in df_with_atr.columns else current_price * 0.02
        
        atr_multiplier = self.risk_params.get('atr_stop_multiplier', 2.0)
        
        # 고정 비율 vs ATR 기반 선택
        stop_loss_pct = self.risk_params.get('stop_loss_percent', 0.02)
        take_profit_pct = self.risk_params.get('take_profit_percent', 0.04)
        
        stop_loss_fixed = current_price * (1 - stop_loss_pct)
        stop_loss_atr = current_price - (atr * atr_multiplier)
        
        take_profit_fixed = current_price * (1 + take_profit_pct)
        risk_reward = self.risk_params.get('risk_reward_ratio', 2.0)
        take_profit_rr = current_price + (current_price - stop_loss_atr) * risk_reward
        
        return {
            'stop_loss': max(stop_loss_fixed, stop_loss_atr),  # 더 보수적인 값
            'take_profit': min(take_profit_fixed, take_profit_rr),
            'atr': float(atr),
            'position_size': self._calculate_position_size(current_price, stop_loss_atr)
        }
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """포지션 크기 계산 (자본의 %)"""
        max_position = self.risk_params.get('max_position_size', 0.1)
        risk_per_trade = max_position * 0.5  # 단일 거래 최대 리스크
        
        price_risk = abs(entry_price - stop_loss) / entry_price
        
        if price_risk > 0:
            position_size = min(risk_per_trade / price_risk, max_position)
        else:
            position_size = max_position
        
        return position_size
    
    def update_hyperparameters(self, updates: Dict[str, Any]):
        """하이퍼파라미터 업데이트"""
        for path, value in updates.items():
            self.hp_manager.set(path, value)
        
        # 모듈 재초기화
        self.signal_generator = SignalGenerator(config=self.hp_manager.to_config_dict())
        self.signal_combiner = SignalCombiner(
            self.hp_manager.params.get('signal_combination', {})
        )
        self.risk_params = self.hp_manager.params.get('risk_management', {})
    
    def apply_preset(self, preset_name: str):
        """프리셋 적용"""
        self.hp_manager.apply_preset(preset_name)
        
        # 모듈 재초기화
        self.signal_generator = SignalGenerator(config=self.hp_manager.to_config_dict())
        self.signal_combiner = SignalCombiner(
            self.hp_manager.params.get('signal_combination', {})
        )
        self.risk_params = self.hp_manager.params.get('risk_management', {})
    
    def get_all_weights(self) -> Dict[str, float]:
        """모든 가중치 반환"""
        return self.hp_manager.get_all_weights()
    
    def set_weights(self, weights: Dict[str, float]):
        """가중치 설정"""
        self.hp_manager.set_all_weights(weights)
        self.signal_combiner.update_weights(
            self.hp_manager.params.get('signal_combination', {}).get('category_weights', {})
        )
    
    def backtest_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        백테스트용 신호 생성
        
        Returns:
            신호가 포함된 데이터프레임
        """
        # 전체 기간 신호 생성
        signals = self.signal_generator.generate_signals_by_category(df)
        combined = self.signal_combiner.combine(signals)
        
        result = df.copy()
        result['signal'] = combined
        result['signal_type'] = combined.apply(
            lambda x: self.signal_combiner.classify_signal(x)
        )
        
        # 신호 기반 포지션
        result['position'] = 0
        result.loc[result['signal'] > 0.3, 'position'] = 1  # Long
        result.loc[result['signal'] < -0.3, 'position'] = -1  # Short
        
        return result
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """전략 설정 요약"""
        return {
            'preset': self.hp_manager.history[-1].get('preset', 'custom') 
                     if self.hp_manager.history else 'default',
            'combination_method': self.signal_combiner.method.value,
            'category_weights': self.signal_combiner.category_weights,
            'thresholds': self.signal_combiner.thresholds,
            'risk_management': self.risk_params,
            'enabled_indicators': self._get_enabled_indicators()
        }
    
    def _get_enabled_indicators(self) -> Dict[str, List[str]]:
        """활성화된 지표 목록"""
        enabled = {}
        
        for category in ['trend', 'momentum', 'volatility', 'volume']:
            enabled[category] = []
            cat_config = self.hp_manager.params.get(category, {})
            for indicator, config in cat_config.items():
                if isinstance(config, dict) and config.get('enabled', True):
                    enabled[category].append(indicator)
        
        return enabled
