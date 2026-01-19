"""
Chart Pattern Recognition
차트 패턴 인식 모듈

지원하는 패턴:
- Double Top/Bottom
- Head and Shoulders
- Triangles (Ascending, Descending, Symmetrical)
- Flags and Pennants
- Support/Resistance
- Fibonacci Levels
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


class ChartPatterns:
    """
    차트 패턴 인식 클래스
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.weight = self.config.get('weight', 1.3)
        self.patterns_config = self.config.get('patterns', {})
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'weight': 1.3,
            'patterns': {
                'double_top': {'enabled': True, 'signal': -1.0, 'lookback': 50, 'tolerance': 0.02},
                'double_bottom': {'enabled': True, 'signal': 1.0, 'lookback': 50, 'tolerance': 0.02},
                'head_and_shoulders': {'enabled': True, 'signal': -1.2, 'lookback': 60},
                'inverse_head_and_shoulders': {'enabled': True, 'signal': 1.2, 'lookback': 60},
                'ascending_triangle': {'enabled': True, 'signal': 0.8, 'lookback': 40},
                'descending_triangle': {'enabled': True, 'signal': -0.8, 'lookback': 40},
                'symmetrical_triangle': {'enabled': True, 'signal': 0.0, 'lookback': 40},
                'bull_flag': {'enabled': True, 'signal': 0.9, 'lookback': 30},
                'bear_flag': {'enabled': True, 'signal': -0.9, 'lookback': 30}
            }
        }
    
    def _find_local_extrema(self, series: pd.Series, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        로컬 고점/저점 찾기
        
        Args:
            series: 가격 시리즈
            order: 극값 판단을 위한 이웃 수
            
        Returns:
            (고점 인덱스, 저점 인덱스)
        """
        values = series.values
        
        # 고점 찾기
        local_max_idx = argrelextrema(values, np.greater, order=order)[0]
        
        # 저점 찾기
        local_min_idx = argrelextrema(values, np.less, order=order)[0]
        
        return local_max_idx, local_min_idx
    
    def _are_values_similar(self, v1: float, v2: float, tolerance: float = 0.02) -> bool:
        """두 값이 tolerance 범위 내에서 유사한지 확인"""
        return abs(v1 - v2) / max(v1, v2) < tolerance
    
    def detect_double_top(self, df: pd.DataFrame) -> pd.Series:
        """
        더블탑 패턴 감지
        - 두 개의 비슷한 높이의 고점
        - 중간에 저점 (neckline)
        - 두 번째 고점 이후 하락
        """
        result = pd.Series(False, index=df.index)
        config = self.patterns_config.get('double_top', {})
        
        if not config.get('enabled', True):
            return result
        
        lookback = config.get('lookback', 50)
        tolerance = config.get('tolerance', 0.02)
        
        high = df['high']
        close = df['close']
        
        local_max_idx, local_min_idx = self._find_local_extrema(high)
        
        for i in range(len(df)):
            if i < lookback:
                continue
            
            # 현재 윈도우 내의 고점들 찾기
            window_max_idx = local_max_idx[(local_max_idx >= i - lookback) & (local_max_idx < i)]
            
            if len(window_max_idx) >= 2:
                # 가장 최근 두 고점
                peak1_idx = window_max_idx[-2]
                peak2_idx = window_max_idx[-1]
                
                peak1 = high.iloc[peak1_idx]
                peak2 = high.iloc[peak2_idx]
                
                # 두 고점이 비슷한 높이인지
                if self._are_values_similar(peak1, peak2, tolerance):
                    # 중간에 저점이 있는지
                    between_min_idx = local_min_idx[(local_min_idx > peak1_idx) & (local_min_idx < peak2_idx)]
                    
                    if len(between_min_idx) > 0:
                        neckline = high.iloc[between_min_idx[-1]]
                        
                        # 현재 가격이 neckline 아래로 하락했는지
                        if close.iloc[i] < neckline:
                            result.iloc[i] = True
        
        return result
    
    def detect_double_bottom(self, df: pd.DataFrame) -> pd.Series:
        """
        더블바텀 패턴 감지
        """
        result = pd.Series(False, index=df.index)
        config = self.patterns_config.get('double_bottom', {})
        
        if not config.get('enabled', True):
            return result
        
        lookback = config.get('lookback', 50)
        tolerance = config.get('tolerance', 0.02)
        
        low = df['low']
        close = df['close']
        
        local_max_idx, local_min_idx = self._find_local_extrema(low)
        
        for i in range(len(df)):
            if i < lookback:
                continue
            
            window_min_idx = local_min_idx[(local_min_idx >= i - lookback) & (local_min_idx < i)]
            
            if len(window_min_idx) >= 2:
                trough1_idx = window_min_idx[-2]
                trough2_idx = window_min_idx[-1]
                
                trough1 = low.iloc[trough1_idx]
                trough2 = low.iloc[trough2_idx]
                
                if self._are_values_similar(trough1, trough2, tolerance):
                    between_max_idx = local_max_idx[(local_max_idx > trough1_idx) & (local_max_idx < trough2_idx)]
                    
                    if len(between_max_idx) > 0:
                        neckline = low.iloc[between_max_idx[-1]]
                        
                        if close.iloc[i] > neckline:
                            result.iloc[i] = True
        
        return result
    
    def detect_head_and_shoulders(self, df: pd.DataFrame) -> pd.Series:
        """
        헤드앤숄더 패턴 감지
        - 왼쪽 어깨, 머리 (더 높음), 오른쪽 어깨
        """
        result = pd.Series(False, index=df.index)
        config = self.patterns_config.get('head_and_shoulders', {})
        
        if not config.get('enabled', True):
            return result
        
        lookback = config.get('lookback', 60)
        
        high = df['high']
        close = df['close']
        
        local_max_idx, local_min_idx = self._find_local_extrema(high, order=7)
        
        for i in range(len(df)):
            if i < lookback:
                continue
            
            window_max_idx = local_max_idx[(local_max_idx >= i - lookback) & (local_max_idx < i)]
            
            if len(window_max_idx) >= 3:
                left_shoulder_idx = window_max_idx[-3]
                head_idx = window_max_idx[-2]
                right_shoulder_idx = window_max_idx[-1]
                
                left_shoulder = high.iloc[left_shoulder_idx]
                head = high.iloc[head_idx]
                right_shoulder = high.iloc[right_shoulder_idx]
                
                # 머리가 어깨보다 높고, 양 어깨가 비슷함
                if (head > left_shoulder and head > right_shoulder and
                    self._are_values_similar(left_shoulder, right_shoulder, 0.03)):
                    
                    # Neckline 계산 (어깨 사이의 저점들)
                    left_min_idx = local_min_idx[(local_min_idx > left_shoulder_idx) & (local_min_idx < head_idx)]
                    right_min_idx = local_min_idx[(local_min_idx > head_idx) & (local_min_idx < right_shoulder_idx)]
                    
                    if len(left_min_idx) > 0 and len(right_min_idx) > 0:
                        neckline = min(high.iloc[left_min_idx[-1]], high.iloc[right_min_idx[-1]])
                        
                        if close.iloc[i] < neckline:
                            result.iloc[i] = True
        
        return result
    
    def detect_inverse_head_and_shoulders(self, df: pd.DataFrame) -> pd.Series:
        """
        역 헤드앤숄더 패턴 감지
        """
        result = pd.Series(False, index=df.index)
        config = self.patterns_config.get('inverse_head_and_shoulders', {})
        
        if not config.get('enabled', True):
            return result
        
        lookback = config.get('lookback', 60)
        
        low = df['low']
        close = df['close']
        
        local_max_idx, local_min_idx = self._find_local_extrema(low, order=7)
        
        for i in range(len(df)):
            if i < lookback:
                continue
            
            window_min_idx = local_min_idx[(local_min_idx >= i - lookback) & (local_min_idx < i)]
            
            if len(window_min_idx) >= 3:
                left_shoulder_idx = window_min_idx[-3]
                head_idx = window_min_idx[-2]
                right_shoulder_idx = window_min_idx[-1]
                
                left_shoulder = low.iloc[left_shoulder_idx]
                head = low.iloc[head_idx]
                right_shoulder = low.iloc[right_shoulder_idx]
                
                # 머리가 어깨보다 낮고, 양 어깨가 비슷함
                if (head < left_shoulder and head < right_shoulder and
                    self._are_values_similar(left_shoulder, right_shoulder, 0.03)):
                    
                    left_max_idx = local_max_idx[(local_max_idx > left_shoulder_idx) & (local_max_idx < head_idx)]
                    right_max_idx = local_max_idx[(local_max_idx > head_idx) & (local_max_idx < right_shoulder_idx)]
                    
                    if len(left_max_idx) > 0 and len(right_max_idx) > 0:
                        neckline = max(low.iloc[left_max_idx[-1]], low.iloc[right_max_idx[-1]])
                        
                        if close.iloc[i] > neckline:
                            result.iloc[i] = True
        
        return result
    
    def detect_ascending_triangle(self, df: pd.DataFrame) -> pd.Series:
        """
        상승 삼각형 패턴 감지
        - 수평 저항선
        - 상승하는 지지선
        """
        result = pd.Series(False, index=df.index)
        config = self.patterns_config.get('ascending_triangle', {})
        
        if not config.get('enabled', True):
            return result
        
        lookback = config.get('lookback', 40)
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        for i in range(len(df)):
            if i < lookback:
                continue
            
            window_high = high.iloc[i - lookback:i]
            window_low = low.iloc[i - lookback:i]
            
            # 저항선 (최고점 부근이 수평)
            resistance = window_high.max()
            resistance_touches = (window_high >= resistance * 0.98).sum()
            
            # 지지선 (저점이 상승)
            low_slope = np.polyfit(range(len(window_low)), window_low.values, 1)[0]
            
            # 상승 삼각형: 수평 저항 + 상승 지지
            if resistance_touches >= 2 and low_slope > 0:
                # 돌파 확인
                if close.iloc[i] > resistance:
                    result.iloc[i] = True
        
        return result
    
    def detect_descending_triangle(self, df: pd.DataFrame) -> pd.Series:
        """
        하강 삼각형 패턴 감지
        """
        result = pd.Series(False, index=df.index)
        config = self.patterns_config.get('descending_triangle', {})
        
        if not config.get('enabled', True):
            return result
        
        lookback = config.get('lookback', 40)
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        for i in range(len(df)):
            if i < lookback:
                continue
            
            window_high = high.iloc[i - lookback:i]
            window_low = low.iloc[i - lookback:i]
            
            # 지지선 (수평)
            support = window_low.min()
            support_touches = (window_low <= support * 1.02).sum()
            
            # 저항선 (하락)
            high_slope = np.polyfit(range(len(window_high)), window_high.values, 1)[0]
            
            # 하강 삼각형: 수평 지지 + 하락 저항
            if support_touches >= 2 and high_slope < 0:
                if close.iloc[i] < support:
                    result.iloc[i] = True
        
        return result
    
    def detect_symmetrical_triangle(self, df: pd.DataFrame) -> pd.Series:
        """
        대칭 삼각형 패턴 감지
        """
        result = pd.Series(False, index=df.index)
        config = self.patterns_config.get('symmetrical_triangle', {})
        
        if not config.get('enabled', True):
            return result
        
        lookback = config.get('lookback', 40)
        
        high = df['high']
        low = df['low']
        
        for i in range(len(df)):
            if i < lookback:
                continue
            
            window_high = high.iloc[i - lookback:i]
            window_low = low.iloc[i - lookback:i]
            
            # 고점 하락, 저점 상승
            high_slope = np.polyfit(range(len(window_high)), window_high.values, 1)[0]
            low_slope = np.polyfit(range(len(window_low)), window_low.values, 1)[0]
            
            # 대칭 삼각형: 하락하는 고점 + 상승하는 저점
            if high_slope < 0 and low_slope > 0:
                # 범위 축소 확인
                range_start = window_high.iloc[0] - window_low.iloc[0]
                range_end = window_high.iloc[-1] - window_low.iloc[-1]
                
                if range_end < range_start * 0.5:  # 범위가 50% 이상 축소
                    result.iloc[i] = True
        
        return result
    
    def detect_bull_flag(self, df: pd.DataFrame) -> pd.Series:
        """
        상승 깃발 패턴 감지
        - 급격한 상승 (깃대)
        - 하락 또는 횡보 조정 (깃발)
        """
        result = pd.Series(False, index=df.index)
        config = self.patterns_config.get('bull_flag', {})
        
        if not config.get('enabled', True):
            return result
        
        lookback = config.get('lookback', 30)
        
        close = df['close']
        
        for i in range(len(df)):
            if i < lookback:
                continue
            
            # 깃대: 초반 급등 (lookback의 절반)
            pole_end = i - lookback // 2
            pole_start = i - lookback
            
            pole_gain = (close.iloc[pole_end] - close.iloc[pole_start]) / close.iloc[pole_start]
            
            # 깃발: 후반 조정
            flag_start = pole_end
            flag_end = i - 1
            
            if flag_end > flag_start:
                flag_change = (close.iloc[flag_end] - close.iloc[flag_start]) / close.iloc[flag_start]
                
                # 급등 후 소폭 하락 또는 횡보
                if pole_gain > 0.1 and -0.05 < flag_change < 0.02:
                    # 돌파 확인
                    if close.iloc[i] > close.iloc[pole_end]:
                        result.iloc[i] = True
        
        return result
    
    def detect_bear_flag(self, df: pd.DataFrame) -> pd.Series:
        """
        하락 깃발 패턴 감지
        """
        result = pd.Series(False, index=df.index)
        config = self.patterns_config.get('bear_flag', {})
        
        if not config.get('enabled', True):
            return result
        
        lookback = config.get('lookback', 30)
        
        close = df['close']
        
        for i in range(len(df)):
            if i < lookback:
                continue
            
            pole_end = i - lookback // 2
            pole_start = i - lookback
            
            pole_loss = (close.iloc[pole_end] - close.iloc[pole_start]) / close.iloc[pole_start]
            
            flag_start = pole_end
            flag_end = i - 1
            
            if flag_end > flag_start:
                flag_change = (close.iloc[flag_end] - close.iloc[flag_start]) / close.iloc[flag_start]
                
                # 급락 후 소폭 반등 또는 횡보
                if pole_loss < -0.1 and -0.02 < flag_change < 0.05:
                    if close.iloc[i] < close.iloc[pole_end]:
                        result.iloc[i] = True
        
        return result
    
    def detect_all_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 차트 패턴 감지"""
        result = df.copy()
        
        result['cp_double_top'] = self.detect_double_top(df)
        result['cp_double_bottom'] = self.detect_double_bottom(df)
        result['cp_head_and_shoulders'] = self.detect_head_and_shoulders(df)
        result['cp_inverse_head_and_shoulders'] = self.detect_inverse_head_and_shoulders(df)
        result['cp_ascending_triangle'] = self.detect_ascending_triangle(df)
        result['cp_descending_triangle'] = self.detect_descending_triangle(df)
        result['cp_symmetrical_triangle'] = self.detect_symmetrical_triangle(df)
        result['cp_bull_flag'] = self.detect_bull_flag(df)
        result['cp_bear_flag'] = self.detect_bear_flag(df)
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """차트 패턴 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        pattern_methods = {
            'double_top': self.detect_double_top,
            'double_bottom': self.detect_double_bottom,
            'head_and_shoulders': self.detect_head_and_shoulders,
            'inverse_head_and_shoulders': self.detect_inverse_head_and_shoulders,
            'ascending_triangle': self.detect_ascending_triangle,
            'descending_triangle': self.detect_descending_triangle,
            'symmetrical_triangle': self.detect_symmetrical_triangle,
            'bull_flag': self.detect_bull_flag,
            'bear_flag': self.detect_bear_flag
        }
        
        for pattern_name, detect_method in pattern_methods.items():
            pattern_config = self.patterns_config.get(pattern_name, {})
            if pattern_config.get('enabled', True):
                detected = detect_method(df)
                signal_value = pattern_config.get('signal', 0)
                signals[detected] += signal_value * self.weight
        
        return signals.clip(-1, 1)


class SupportResistance:
    """
    지지/저항 레벨 감지 클래스
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'enabled': True,
            'weight': 1.1,
            'params': {
                'lookback_period': 100,
                'min_touches': 2,
                'tolerance': 0.01
            },
            'signals': {
                'support_bounce': 0.8,
                'resistance_rejection': -0.8,
                'support_break': -1.0,
                'resistance_break': 1.0
            }
        }
        self.weight = self.config.get('weight', 1.1)
        self.params = self.config.get('params', {})
        self.signals_config = self.config.get('signals', {})
    
    def find_levels(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        지지/저항 레벨 찾기
        
        Returns:
            {'support': [...], 'resistance': [...]}
        """
        lookback = self.params.get('lookback_period', 100)
        min_touches = self.params.get('min_touches', 2)
        tolerance = self.params.get('tolerance', 0.01)
        
        high = df['high'].tail(lookback)
        low = df['low'].tail(lookback)
        close = df['close'].tail(lookback)
        
        # 피벗 포인트 기반 레벨 찾기
        levels = []
        
        # 고점/저점 찾기
        local_max_idx = argrelextrema(high.values, np.greater, order=5)[0]
        local_min_idx = argrelextrema(low.values, np.less, order=5)[0]
        
        # 고점들 (저항 후보)
        for idx in local_max_idx:
            levels.append(high.iloc[idx])
        
        # 저점들 (지지 후보)
        for idx in local_min_idx:
            levels.append(low.iloc[idx])
        
        # 레벨 클러스터링
        support_levels = []
        resistance_levels = []
        current_price = close.iloc[-1]
        
        for level in levels:
            # 터치 횟수 계산
            touches = ((high >= level * (1 - tolerance)) & 
                      (high <= level * (1 + tolerance))).sum()
            touches += ((low >= level * (1 - tolerance)) & 
                       (low <= level * (1 + tolerance))).sum()
            
            if touches >= min_touches:
                if level < current_price:
                    support_levels.append(level)
                else:
                    resistance_levels.append(level)
        
        return {
            'support': sorted(set(support_levels), reverse=True)[:5],  # 가장 가까운 5개
            'resistance': sorted(set(resistance_levels))[:5]
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """지지/저항 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        if not self.config.get('enabled', True):
            return signals
        
        levels = self.find_levels(df)
        tolerance = self.params.get('tolerance', 0.01)
        
        close = df['close']
        
        for i in range(len(df)):
            current_price = close.iloc[i]
            
            # 지지 레벨 체크
            for support in levels['support']:
                # 지지선 근처
                if abs(current_price - support) / support < tolerance:
                    # 반등 (이전보다 상승)
                    if i > 0 and close.iloc[i] > close.iloc[i - 1]:
                        signals.iloc[i] += self.signals_config.get('support_bounce', 0.8)
                    # 이탈 (이전보다 하락)
                    elif i > 0 and close.iloc[i] < close.iloc[i - 1]:
                        signals.iloc[i] += self.signals_config.get('support_break', -1.0)
            
            # 저항 레벨 체크
            for resistance in levels['resistance']:
                if abs(current_price - resistance) / resistance < tolerance:
                    if i > 0 and close.iloc[i] < close.iloc[i - 1]:
                        signals.iloc[i] += self.signals_config.get('resistance_rejection', -0.8)
                    elif i > 0 and close.iloc[i] > close.iloc[i - 1]:
                        signals.iloc[i] += self.signals_config.get('resistance_break', 1.0)
        
        return signals.clip(-1, 1) * self.weight


class FibonacciLevels:
    """
    피보나치 레벨 계산 클래스
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'enabled': True,
            'weight': 0.9,
            'params': {
                'levels': [0.236, 0.382, 0.5, 0.618, 0.786],
                'extension_levels': [1.0, 1.272, 1.618, 2.0, 2.618],
                'tolerance': 0.005
            },
            'signals': {
                'retracement_support': 0.6,
                'retracement_resistance': -0.6,
                'golden_ratio_618': 0.8
            }
        }
        self.weight = self.config.get('weight', 0.9)
        self.params = self.config.get('params', {})
        self.signals_config = self.config.get('signals', {})
    
    def calculate_levels(self, df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
        """
        피보나치 되돌림 레벨 계산
        
        Returns:
            {레벨명: 가격} 딕셔너리
        """
        high = df['high'].tail(lookback).max()
        low = df['low'].tail(lookback).min()
        
        diff = high - low
        
        levels = {}
        for fib_level in self.params.get('levels', [0.236, 0.382, 0.5, 0.618, 0.786]):
            levels[f'fib_{fib_level}'] = high - (diff * fib_level)
        
        levels['swing_high'] = high
        levels['swing_low'] = low
        
        return levels
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """피보나치 레벨 기반 신호 생성"""
        signals = pd.Series(0.0, index=df.index)
        
        if not self.config.get('enabled', True):
            return signals
        
        tolerance = self.params.get('tolerance', 0.005)
        levels = self.calculate_levels(df)
        
        close = df['close']
        
        for i in range(len(df)):
            current_price = close.iloc[i]
            
            for level_name, level_price in levels.items():
                if abs(current_price - level_price) / level_price < tolerance:
                    # 0.618 황금비율 특별 가중
                    if '0.618' in level_name:
                        if i > 0 and close.iloc[i] > close.iloc[i - 1]:
                            signals.iloc[i] += self.signals_config.get('golden_ratio_618', 0.8)
                        elif i > 0:
                            signals.iloc[i] -= self.signals_config.get('golden_ratio_618', 0.8)
                    else:
                        # 일반 피보나치 레벨
                        if i > 0 and close.iloc[i] > close.iloc[i - 1]:
                            signals.iloc[i] += self.signals_config.get('retracement_support', 0.6)
                        elif i > 0:
                            signals.iloc[i] += self.signals_config.get('retracement_resistance', -0.6)
        
        return signals.clip(-1, 1) * self.weight
