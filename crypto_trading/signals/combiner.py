"""
Signal Combiner
신호 조합기

여러 신호를 조합하여 최종 매매 신호를 생성합니다.
다양한 조합 방법과 가중치 조절 기능을 제공합니다.
"""

from typing import Dict, Any, Optional, List, Callable
import pandas as pd
import numpy as np
from enum import Enum


class CombinationMethod(Enum):
    """신호 조합 방법"""
    WEIGHTED_AVERAGE = "weighted_average"      # 가중 평균
    VOTING = "voting"                          # 투표 방식
    CONSENSUS = "consensus"                    # 합의 (대부분 일치)
    MAXIMUM = "maximum"                        # 최대값
    MINIMUM = "minimum"                        # 최소값
    ML_ENSEMBLE = "ml_ensemble"                # ML 앙상블 (미래 확장)


class SignalCombiner:
    """
    신호 조합기
    
    여러 분석 모듈의 신호를 다양한 방법으로 조합합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 조합 설정
        """
        self.config = config or self._default_config()
        
        self.method = CombinationMethod(
            self.config.get('method', 'weighted_average')
        )
        self.category_weights = self.config.get('category_weights', {})
        self.thresholds = self.config.get('thresholds', {})
        self.confirmation = self.config.get('confirmation', {})
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'method': 'weighted_average',
            'category_weights': {
                'trend': 1.2,
                'momentum': 1.0,
                'volatility': 0.8,
                'volume': 0.9,
                'candlestick': 1.1,
                'chart_pattern': 1.0,
                'support_resistance': 1.0,
                'fibonacci': 0.7
            },
            'thresholds': {
                'strong_buy': 0.7,
                'buy': 0.3,
                'neutral_high': 0.1,
                'neutral_low': -0.1,
                'sell': -0.3,
                'strong_sell': -0.7
            },
            'confirmation': {
                'min_confirming_indicators': 3,
                'trend_alignment_bonus': 0.2,
                'volume_confirmation_bonus': 0.15
            }
        }
    
    def combine(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """
        신호 조합 (설정된 방법 사용)
        
        Args:
            signals: {카테고리명: 신호 시리즈}
            
        Returns:
            조합된 신호 시리즈
        """
        if not signals:
            raise ValueError("No signals to combine")
        
        method_map = {
            CombinationMethod.WEIGHTED_AVERAGE: self._combine_weighted_average,
            CombinationMethod.VOTING: self._combine_voting,
            CombinationMethod.CONSENSUS: self._combine_consensus,
            CombinationMethod.MAXIMUM: self._combine_maximum,
            CombinationMethod.MINIMUM: self._combine_minimum,
            CombinationMethod.ML_ENSEMBLE: self._combine_ml_ensemble
        }
        
        combine_func = method_map.get(self.method, self._combine_weighted_average)
        combined = combine_func(signals)
        
        # 확인 신호 보너스 적용
        combined = self._apply_confirmation_bonus(signals, combined)
        
        return combined.clip(-1, 1)
    
    def _combine_weighted_average(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """가중 평균 조합"""
        weighted_sum = None
        total_weight = 0
        
        for category, signal in signals.items():
            weight = self.category_weights.get(category, 1.0)
            
            if weighted_sum is None:
                weighted_sum = signal * weight
            else:
                weighted_sum += signal * weight
            
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else weighted_sum
    
    def _combine_voting(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """
        투표 방식 조합
        - 각 신호가 0.1 이상이면 매수 표, -0.1 이하면 매도 표
        - 최종 신호는 (매수 표 - 매도 표) / 총 표
        """
        index = list(signals.values())[0].index
        buy_votes = pd.Series(0, index=index)
        sell_votes = pd.Series(0, index=index)
        total = len(signals)
        
        for category, signal in signals.items():
            weight = self.category_weights.get(category, 1.0)
            
            buy_votes += ((signal > 0.1) * weight)
            sell_votes += ((signal < -0.1) * weight)
        
        total_weight = sum(self.category_weights.get(cat, 1.0) for cat in signals.keys())
        
        return (buy_votes - sell_votes) / total_weight
    
    def _combine_consensus(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """
        합의 방식 조합
        - 대부분의 지표가 같은 방향을 가리킬 때만 신호
        """
        min_agreement = self.confirmation.get('min_confirming_indicators', 3)
        
        index = list(signals.values())[0].index
        result = pd.Series(0.0, index=index)
        
        for i in range(len(index)):
            bullish_count = sum(1 for s in signals.values() if s.iloc[i] > 0.1)
            bearish_count = sum(1 for s in signals.values() if s.iloc[i] < -0.1)
            
            if bullish_count >= min_agreement:
                # 합의된 매수 신호의 평균
                bullish_signals = [s.iloc[i] for s in signals.values() if s.iloc[i] > 0.1]
                result.iloc[i] = np.mean(bullish_signals)
            elif bearish_count >= min_agreement:
                bearish_signals = [s.iloc[i] for s in signals.values() if s.iloc[i] < -0.1]
                result.iloc[i] = np.mean(bearish_signals)
        
        return result
    
    def _combine_maximum(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """최대 절대값 조합 (가장 강한 신호)"""
        df = pd.DataFrame(signals)
        
        # 가중치 적용
        for col in df.columns:
            weight = self.category_weights.get(col, 1.0)
            df[col] *= weight
        
        # 절대값이 가장 큰 값 선택
        abs_df = df.abs()
        max_idx = abs_df.idxmax(axis=1)
        
        result = pd.Series(index=df.index, dtype=float)
        for i, idx in enumerate(df.index):
            result.iloc[i] = df.loc[idx, max_idx.iloc[i]]
        
        return result / max(self.category_weights.values())
    
    def _combine_minimum(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """최소 절대값 조합 (가장 보수적)"""
        df = pd.DataFrame(signals)
        
        # 가중치 적용
        for col in df.columns:
            weight = self.category_weights.get(col, 1.0)
            df[col] *= weight
        
        # 절대값이 가장 작은 값 선택
        abs_df = df.abs()
        min_idx = abs_df.idxmin(axis=1)
        
        result = pd.Series(index=df.index, dtype=float)
        for i, idx in enumerate(df.index):
            result.iloc[i] = df.loc[idx, min_idx.iloc[i]]
        
        return result / max(self.category_weights.values())
    
    def _combine_ml_ensemble(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """ML 앙상블 조합 (향후 확장용 - 현재는 가중 평균)"""
        # TODO: 실제 ML 모델 적용
        return self._combine_weighted_average(signals)
    
    def _apply_confirmation_bonus(self, signals: Dict[str, pd.Series], 
                                   combined: pd.Series) -> pd.Series:
        """확인 신호 보너스 적용"""
        result = combined.copy()
        
        trend_bonus = self.confirmation.get('trend_alignment_bonus', 0.2)
        volume_bonus = self.confirmation.get('volume_confirmation_bonus', 0.15)
        
        # 추세 정렬 보너스
        if 'trend' in signals:
            trend_signal = signals['trend']
            
            # 추세와 조합 신호가 같은 방향이면 보너스
            same_direction = (trend_signal * combined) > 0
            result[same_direction] += np.sign(combined[same_direction]) * trend_bonus
        
        # 거래량 확인 보너스
        if 'volume' in signals:
            volume_signal = signals['volume']
            
            # 거래량 신호가 방향을 확인해주면 보너스
            volume_confirms = (volume_signal * combined) > 0
            result[volume_confirms] += np.sign(combined[volume_confirms]) * volume_bonus
        
        return result
    
    def classify_signal(self, signal_value: float) -> str:
        """
        신호값을 분류
        
        Returns:
            신호 분류 문자열
        """
        thresholds = self.thresholds
        
        if signal_value >= thresholds.get('strong_buy', 0.7):
            return 'strong_buy'
        elif signal_value >= thresholds.get('buy', 0.3):
            return 'buy'
        elif signal_value >= thresholds.get('neutral_high', 0.1):
            return 'weak_buy'
        elif signal_value >= thresholds.get('neutral_low', -0.1):
            return 'neutral'
        elif signal_value >= thresholds.get('sell', -0.3):
            return 'weak_sell'
        elif signal_value >= thresholds.get('strong_sell', -0.7):
            return 'sell'
        else:
            return 'strong_sell'
    
    def get_signal_strength(self, signal_value: float) -> Dict[str, Any]:
        """
        신호 강도 분석
        
        Returns:
            신호 강도 정보 딕셔너리
        """
        classification = self.classify_signal(signal_value)
        
        strength_map = {
            'strong_buy': {'strength': 'very_strong', 'direction': 'bullish', 'confidence': 0.9},
            'buy': {'strength': 'strong', 'direction': 'bullish', 'confidence': 0.7},
            'weak_buy': {'strength': 'weak', 'direction': 'bullish', 'confidence': 0.5},
            'neutral': {'strength': 'none', 'direction': 'neutral', 'confidence': 0.3},
            'weak_sell': {'strength': 'weak', 'direction': 'bearish', 'confidence': 0.5},
            'sell': {'strength': 'strong', 'direction': 'bearish', 'confidence': 0.7},
            'strong_sell': {'strength': 'very_strong', 'direction': 'bearish', 'confidence': 0.9}
        }
        
        info = strength_map.get(classification, strength_map['neutral'])
        info['signal_value'] = signal_value
        info['classification'] = classification
        
        return info
    
    def update_weights(self, new_weights: Dict[str, float]):
        """카테고리 가중치 업데이트"""
        self.category_weights.update(new_weights)
    
    def set_method(self, method: str):
        """조합 방법 변경"""
        self.method = CombinationMethod(method)
    
    def get_weight_impact(self, signals: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        각 카테고리의 가중치 영향 분석
        
        Returns:
            카테고리별 기여도 데이터프레임
        """
        index = list(signals.values())[0].index
        impact = pd.DataFrame(index=index)
        
        total_weight = sum(self.category_weights.get(cat, 1.0) for cat in signals.keys())
        
        for category, signal in signals.items():
            weight = self.category_weights.get(category, 1.0)
            contribution = signal * weight / total_weight
            impact[f'{category}_contribution'] = contribution
            impact[f'{category}_raw'] = signal
            impact[f'{category}_weight'] = weight
        
        return impact
    
    def optimize_weights_for_direction(self, signals: Dict[str, pd.Series], 
                                        target_direction: str = 'bullish') -> Dict[str, float]:
        """
        특정 방향으로 신호를 강화하는 가중치 제안
        (단순 휴리스틱 - 실제 사용 시 주의)
        
        Args:
            signals: 신호 딕셔너리
            target_direction: 'bullish' 또는 'bearish'
            
        Returns:
            조정된 가중치 딕셔너리
        """
        adjusted_weights = self.category_weights.copy()
        
        multiplier = 1 if target_direction == 'bullish' else -1
        
        for category, signal in signals.items():
            avg_signal = signal.mean()
            
            # 타겟 방향과 일치하는 신호를 주는 카테고리의 가중치 증가
            if avg_signal * multiplier > 0:
                current_weight = adjusted_weights.get(category, 1.0)
                adjusted_weights[category] = min(current_weight * 1.2, 2.0)
            else:
                current_weight = adjusted_weights.get(category, 1.0)
                adjusted_weights[category] = max(current_weight * 0.8, 0.5)
        
        return adjusted_weights


class AdaptiveSignalCombiner(SignalCombiner):
    """
    적응형 신호 조합기
    
    시장 상황에 따라 가중치를 자동 조정합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.performance_history = {}
    
    def adapt_weights(self, signals: Dict[str, pd.Series], 
                      actual_returns: pd.Series, 
                      lookback: int = 50):
        """
        실제 수익률을 기반으로 가중치 적응
        
        Args:
            signals: 과거 신호
            actual_returns: 실제 수익률
            lookback: 분석 기간
        """
        if len(actual_returns) < lookback:
            return
        
        recent_returns = actual_returns.tail(lookback)
        
        for category, signal in signals.items():
            recent_signal = signal.tail(lookback)
            
            # 신호와 수익률의 상관관계 계산
            correlation = recent_signal.corr(recent_returns)
            
            if not np.isnan(correlation):
                # 상관관계가 높으면 가중치 증가, 낮으면 감소
                current_weight = self.category_weights.get(category, 1.0)
                
                if correlation > 0.3:
                    new_weight = min(current_weight * 1.1, 2.0)
                elif correlation < -0.1:
                    new_weight = max(current_weight * 0.9, 0.3)
                else:
                    new_weight = current_weight
                
                self.category_weights[category] = new_weight
                
                # 성능 기록
                if category not in self.performance_history:
                    self.performance_history[category] = []
                self.performance_history[category].append({
                    'correlation': correlation,
                    'weight': new_weight
                })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 반환"""
        report = {}
        
        for category, history in self.performance_history.items():
            if history:
                correlations = [h['correlation'] for h in history]
                report[category] = {
                    'current_weight': self.category_weights.get(category, 1.0),
                    'avg_correlation': np.mean(correlations),
                    'history_length': len(history)
                }
        
        return report
