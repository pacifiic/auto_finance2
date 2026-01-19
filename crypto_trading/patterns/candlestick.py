"""
Candlestick Pattern Recognition
캔들스틱 패턴 인식 모듈

지원하는 패턴:
- 반전 패턴: Hammer, Inverted Hammer, Engulfing, Morning/Evening Star 등
- 지속 패턴: Doji, Spinning Top, Three Methods 등
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


class CandlestickPatterns:
    """
    캔들스틱 패턴 인식 클래스
    
    모든 패턴을 자동으로 감지하고 신호를 생성합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.reversal_config = self.config.get('reversal', {})
        self.continuation_config = self.config.get('continuation', {})
        
        self.reversal_weight = self.reversal_config.get('weight', 1.4)
        self.continuation_weight = self.continuation_config.get('weight', 0.9)
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'reversal': {
                'weight': 1.4,
                'patterns': {
                    'hammer': {'enabled': True, 'signal': 0.8},
                    'inverted_hammer': {'enabled': True, 'signal': 0.7},
                    'bullish_engulfing': {'enabled': True, 'signal': 1.0},
                    'morning_star': {'enabled': True, 'signal': 1.2},
                    'piercing_line': {'enabled': True, 'signal': 0.8},
                    'three_white_soldiers': {'enabled': True, 'signal': 1.3},
                    'hanging_man': {'enabled': True, 'signal': -0.8},
                    'shooting_star': {'enabled': True, 'signal': -0.9},
                    'bearish_engulfing': {'enabled': True, 'signal': -1.0},
                    'evening_star': {'enabled': True, 'signal': -1.2},
                    'dark_cloud_cover': {'enabled': True, 'signal': -0.8},
                    'three_black_crows': {'enabled': True, 'signal': -1.3}
                }
            },
            'continuation': {
                'weight': 0.9,
                'patterns': {
                    'doji': {'enabled': True, 'signal': 0.0},
                    'spinning_top': {'enabled': True, 'signal': 0.0},
                    'rising_three_methods': {'enabled': True, 'signal': 0.7},
                    'falling_three_methods': {'enabled': True, 'signal': -0.7}
                }
            }
        }
    
    def _get_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """캔들 특성 계산"""
        result = df.copy()
        
        o = df['open']
        h = df['high']
        l = df['low']
        c = df['close']
        
        # 캔들 몸통 (Body)
        result['body'] = abs(c - o)
        result['body_pct'] = result['body'] / o * 100
        
        # 캔들 방향
        result['is_bullish'] = c > o
        result['is_bearish'] = c < o
        
        # 캔들 범위 (Range)
        result['range'] = h - l
        
        # 위꼬리 (Upper Shadow)
        result['upper_shadow'] = h - np.maximum(o, c)
        
        # 아래꼬리 (Lower Shadow)
        result['lower_shadow'] = np.minimum(o, c) - l
        
        # 꼬리 비율
        result['upper_shadow_pct'] = result['upper_shadow'] / (result['range'] + 1e-10)
        result['lower_shadow_pct'] = result['lower_shadow'] / (result['range'] + 1e-10)
        result['body_pct_range'] = result['body'] / (result['range'] + 1e-10)
        
        return result
    
    def _detect_trend(self, df: pd.DataFrame, lookback: int = 5) -> pd.Series:
        """
        추세 감지
        Returns: 1 (상승), -1 (하락), 0 (횡보)
        """
        close = df['close']
        sma = close.rolling(window=lookback).mean()
        
        trend = pd.Series(0, index=df.index)
        trend[close > sma * 1.01] = 1   # 상승 추세
        trend[close < sma * 0.99] = -1  # 하락 추세
        
        return trend
    
    # ===============================
    # 상승 반전 패턴 (Bullish Reversal)
    # ===============================
    
    def detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        """
        해머 패턴 감지
        - 하락 추세 후
        - 작은 몸통
        - 긴 아래꼬리 (몸통의 2배 이상)
        - 위꼬리 거의 없음
        """
        features = self._get_candle_features(df)
        trend = self._detect_trend(df)
        
        hammer = (
            (trend == -1) &  # 하락 추세
            (features['lower_shadow'] >= features['body'] * 2) &  # 긴 아래꼬리
            (features['upper_shadow'] <= features['body'] * 0.3) &  # 짧은 위꼬리
            (features['body_pct_range'] < 0.4)  # 작은 몸통
        )
        
        result = hammer.shift(1)
        result = result.fillna(False).astype(bool)
        return result  # 다음 캔들에서 확인
    
    def detect_inverted_hammer(self, df: pd.DataFrame) -> pd.Series:
        """
        역해머 패턴 감지
        - 하락 추세 후
        - 작은 몸통
        - 긴 위꼬리 (몸통의 2배 이상)
        - 아래꼬리 거의 없음
        """
        features = self._get_candle_features(df)
        trend = self._detect_trend(df)
        
        inverted_hammer = (
            (trend == -1) &
            (features['upper_shadow'] >= features['body'] * 2) &
            (features['lower_shadow'] <= features['body'] * 0.3) &
            (features['body_pct_range'] < 0.4)
        )
        
        result = inverted_hammer.shift(1)
        result = result.fillna(False).astype(bool)
        return result
    
    def detect_bullish_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """
        상승 장악형 (Bullish Engulfing) 감지
        - 하락 추세 후
        - 이전: 음봉
        - 현재: 양봉이 이전 음봉을 완전히 감싸는 패턴
        """
        features = self._get_candle_features(df)
        trend = self._detect_trend(df)
        
        prev_bearish = features['is_bearish'].shift(1)
        curr_bullish = features['is_bullish']
        
        # 현재 양봉이 이전 음봉을 감쌈
        engulfing = (
            (trend.shift(1) == -1) &
            prev_bearish &
            curr_bullish &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        )
        
        return engulfing.fillna(False)
    
    def detect_morning_star(self, df: pd.DataFrame) -> pd.Series:
        """
        샛별형 (Morning Star) 감지
        - 3봉 패턴
        - 첫째 날: 큰 음봉
        - 둘째 날: 작은 몸통 (색깔 무관)
        - 셋째 날: 큰 양봉 (첫째 날 중간 이상까지 상승)
        """
        features = self._get_candle_features(df)
        trend = self._detect_trend(df)
        
        # 첫째 날 큰 음봉
        day1_big_bearish = features['is_bearish'].shift(2) & (features['body_pct_range'].shift(2) > 0.6)
        
        # 둘째 날 작은 몸통
        day2_small_body = features['body_pct_range'].shift(1) < 0.3
        
        # 셋째 날 큰 양봉
        day3_big_bullish = features['is_bullish'] & (features['body_pct_range'] > 0.5)
        
        # 셋째 날 종가가 첫째 날 몸통 중간 이상
        day1_midpoint = (df['open'].shift(2) + df['close'].shift(2)) / 2
        day3_closes_above_mid = df['close'] > day1_midpoint
        
        morning_star = (
            (trend.shift(2) == -1) &
            day1_big_bearish &
            day2_small_body &
            day3_big_bullish &
            day3_closes_above_mid
        )
        
        return morning_star.fillna(False)
    
    def detect_piercing_line(self, df: pd.DataFrame) -> pd.Series:
        """
        관통형 (Piercing Line) 감지
        - 하락 추세
        - 이전: 음봉
        - 현재: 양봉이 이전 음봉 중간 이상으로 상승
        """
        features = self._get_candle_features(df)
        trend = self._detect_trend(df)
        
        prev_bearish = features['is_bearish'].shift(1)
        curr_bullish = features['is_bullish']
        
        prev_midpoint = (df['open'].shift(1) + df['close'].shift(1)) / 2
        
        piercing = (
            (trend.shift(1) == -1) &
            prev_bearish &
            curr_bullish &
            (df['open'] < df['close'].shift(1)) &  # 갭다운 시작
            (df['close'] > prev_midpoint) &  # 중간 이상까지 상승
            (df['close'] < df['open'].shift(1))  # 완전히 감싸지는 않음
        )
        
        return piercing.fillna(False)
    
    def detect_three_white_soldiers(self, df: pd.DataFrame) -> pd.Series:
        """
        적삼병 (Three White Soldiers) 감지
        - 연속 3개의 양봉
        - 각 양봉의 종가가 이전보다 높음
        - 몸통이 크고, 꼬리가 짧음
        """
        features = self._get_candle_features(df)
        
        # 3일 연속 양봉
        three_bullish = (
            features['is_bullish'] &
            features['is_bullish'].shift(1) &
            features['is_bullish'].shift(2)
        )
        
        # 점진적 상승
        rising_closes = (
            (df['close'] > df['close'].shift(1)) &
            (df['close'].shift(1) > df['close'].shift(2))
        )
        
        # 큰 몸통
        big_bodies = (
            (features['body_pct_range'] > 0.5) &
            (features['body_pct_range'].shift(1) > 0.5) &
            (features['body_pct_range'].shift(2) > 0.5)
        )
        
        soldiers = three_bullish & rising_closes & big_bodies
        
        return soldiers.fillna(False)
    
    # ===============================
    # 하락 반전 패턴 (Bearish Reversal)
    # ===============================
    
    def detect_hanging_man(self, df: pd.DataFrame) -> pd.Series:
        """
        교수형 (Hanging Man) 감지
        - 상승 추세 후
        - 해머와 같은 모양이지만 상승 추세에서 발생
        """
        features = self._get_candle_features(df)
        trend = self._detect_trend(df)
        
        hanging_man = (
            (trend == 1) &  # 상승 추세
            (features['lower_shadow'] >= features['body'] * 2) &
            (features['upper_shadow'] <= features['body'] * 0.3) &
            (features['body_pct_range'] < 0.4)
        )
        
        result = hanging_man.shift(1)
        result = result.fillna(False).astype(bool)
        return result
    
    def detect_shooting_star(self, df: pd.DataFrame) -> pd.Series:
        """
        유성형 (Shooting Star) 감지
        - 상승 추세 후
        - 작은 몸통
        - 긴 위꼬리
        - 아래꼬리 거의 없음
        """
        features = self._get_candle_features(df)
        trend = self._detect_trend(df)
        
        shooting_star = (
            (trend == 1) &
            (features['upper_shadow'] >= features['body'] * 2) &
            (features['lower_shadow'] <= features['body'] * 0.3) &
            (features['body_pct_range'] < 0.4)
        )
        
        result = shooting_star.shift(1)
        result = result.fillna(False).astype(bool)
        return result
    
    def detect_bearish_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """
        하락 장악형 (Bearish Engulfing) 감지
        """
        features = self._get_candle_features(df)
        trend = self._detect_trend(df)
        
        prev_bullish = features['is_bullish'].shift(1)
        curr_bearish = features['is_bearish']
        
        engulfing = (
            (trend.shift(1) == 1) &
            prev_bullish &
            curr_bearish &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        )
        
        return engulfing.fillna(False)
    
    def detect_evening_star(self, df: pd.DataFrame) -> pd.Series:
        """
        저녁별형 (Evening Star) 감지
        """
        features = self._get_candle_features(df)
        trend = self._detect_trend(df)
        
        day1_big_bullish = features['is_bullish'].shift(2) & (features['body_pct_range'].shift(2) > 0.6)
        day2_small_body = features['body_pct_range'].shift(1) < 0.3
        day3_big_bearish = features['is_bearish'] & (features['body_pct_range'] > 0.5)
        
        day1_midpoint = (df['open'].shift(2) + df['close'].shift(2)) / 2
        day3_closes_below_mid = df['close'] < day1_midpoint
        
        evening_star = (
            (trend.shift(2) == 1) &
            day1_big_bullish &
            day2_small_body &
            day3_big_bearish &
            day3_closes_below_mid
        )
        
        return evening_star.fillna(False)
    
    def detect_dark_cloud_cover(self, df: pd.DataFrame) -> pd.Series:
        """
        먹구름형 (Dark Cloud Cover) 감지
        """
        features = self._get_candle_features(df)
        trend = self._detect_trend(df)
        
        prev_bullish = features['is_bullish'].shift(1)
        curr_bearish = features['is_bearish']
        
        prev_midpoint = (df['open'].shift(1) + df['close'].shift(1)) / 2
        
        dark_cloud = (
            (trend.shift(1) == 1) &
            prev_bullish &
            curr_bearish &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < prev_midpoint) &
            (df['close'] > df['open'].shift(1))
        )
        
        return dark_cloud.fillna(False)
    
    def detect_three_black_crows(self, df: pd.DataFrame) -> pd.Series:
        """
        흑삼병 (Three Black Crows) 감지
        """
        features = self._get_candle_features(df)
        
        three_bearish = (
            features['is_bearish'] &
            features['is_bearish'].shift(1) &
            features['is_bearish'].shift(2)
        )
        
        falling_closes = (
            (df['close'] < df['close'].shift(1)) &
            (df['close'].shift(1) < df['close'].shift(2))
        )
        
        big_bodies = (
            (features['body_pct_range'] > 0.5) &
            (features['body_pct_range'].shift(1) > 0.5) &
            (features['body_pct_range'].shift(2) > 0.5)
        )
        
        crows = three_bearish & falling_closes & big_bodies
        
        return crows.fillna(False)
    
    # ===============================
    # 지속 패턴 (Continuation)
    # ===============================
    
    def detect_doji(self, df: pd.DataFrame) -> pd.Series:
        """
        도지 (Doji) 감지
        - 시가와 종가가 거의 같음
        """
        features = self._get_candle_features(df)
        
        doji = features['body_pct_range'] < 0.1
        
        return doji.fillna(False)
    
    def detect_spinning_top(self, df: pd.DataFrame) -> pd.Series:
        """
        팽이형 (Spinning Top) 감지
        - 작은 몸통
        - 양쪽에 비슷한 꼬리
        """
        features = self._get_candle_features(df)
        
        spinning_top = (
            (features['body_pct_range'] < 0.3) &
            (features['body_pct_range'] > 0.1) &  # 도지 제외
            (abs(features['upper_shadow'] - features['lower_shadow']) < features['range'] * 0.2)
        )
        
        return spinning_top.fillna(False)
    
    def detect_all_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 캔들스틱 패턴 감지
        
        Returns:
            각 패턴 컬럼이 추가된 데이터프레임
        """
        result = df.copy()
        
        # 반전 패턴
        result['cs_hammer'] = self.detect_hammer(df)
        result['cs_inverted_hammer'] = self.detect_inverted_hammer(df)
        result['cs_bullish_engulfing'] = self.detect_bullish_engulfing(df)
        result['cs_morning_star'] = self.detect_morning_star(df)
        result['cs_piercing_line'] = self.detect_piercing_line(df)
        result['cs_three_white_soldiers'] = self.detect_three_white_soldiers(df)
        
        result['cs_hanging_man'] = self.detect_hanging_man(df)
        result['cs_shooting_star'] = self.detect_shooting_star(df)
        result['cs_bearish_engulfing'] = self.detect_bearish_engulfing(df)
        result['cs_evening_star'] = self.detect_evening_star(df)
        result['cs_dark_cloud_cover'] = self.detect_dark_cloud_cover(df)
        result['cs_three_black_crows'] = self.detect_three_black_crows(df)
        
        # 지속 패턴
        result['cs_doji'] = self.detect_doji(df)
        result['cs_spinning_top'] = self.detect_spinning_top(df)
        
        return result
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        캔들스틱 패턴 기반 신호 생성
        """
        signals = pd.Series(0.0, index=df.index)
        
        reversal_patterns = self.reversal_config.get('patterns', {})
        continuation_patterns = self.continuation_config.get('patterns', {})
        
        # 반전 패턴 신호
        pattern_methods = {
            'hammer': self.detect_hammer,
            'inverted_hammer': self.detect_inverted_hammer,
            'bullish_engulfing': self.detect_bullish_engulfing,
            'morning_star': self.detect_morning_star,
            'piercing_line': self.detect_piercing_line,
            'three_white_soldiers': self.detect_three_white_soldiers,
            'hanging_man': self.detect_hanging_man,
            'shooting_star': self.detect_shooting_star,
            'bearish_engulfing': self.detect_bearish_engulfing,
            'evening_star': self.detect_evening_star,
            'dark_cloud_cover': self.detect_dark_cloud_cover,
            'three_black_crows': self.detect_three_black_crows
        }
        
        for pattern_name, detect_method in pattern_methods.items():
            pattern_config = reversal_patterns.get(pattern_name, {})
            if pattern_config.get('enabled', True):
                detected = detect_method(df)
                signal_value = pattern_config.get('signal', 0)
                signals[detected] += signal_value * self.reversal_weight
        
        # 지속 패턴
        cont_pattern_methods = {
            'doji': self.detect_doji,
            'spinning_top': self.detect_spinning_top
        }
        
        for pattern_name, detect_method in cont_pattern_methods.items():
            pattern_config = continuation_patterns.get(pattern_name, {})
            if pattern_config.get('enabled', True):
                detected = detect_method(df)
                signal_value = pattern_config.get('signal', 0)
                signals[detected] += signal_value * self.continuation_weight
        
        return signals.clip(-1, 1)
    
    def get_detected_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        감지된 패턴 목록 반환
        
        Returns:
            감지된 패턴 정보 리스트
        """
        patterns_df = self.detect_all_patterns(df)
        detected = []
        
        pattern_columns = [col for col in patterns_df.columns if col.startswith('cs_')]
        
        for idx in patterns_df.index:
            for col in pattern_columns:
                if patterns_df.loc[idx, col]:
                    pattern_name = col.replace('cs_', '')
                    detected.append({
                        'index': idx,
                        'pattern': pattern_name,
                        'price': df.loc[idx, 'close'],
                        'signal_type': 'bullish' if 'bullish' in pattern_name or pattern_name in 
                                      ['hammer', 'inverted_hammer', 'morning_star', 'piercing_line', 'three_white_soldiers']
                                      else 'bearish' if pattern_name in 
                                      ['hanging_man', 'shooting_star', 'bearish_engulfing', 'evening_star', 'dark_cloud_cover', 'three_black_crows']
                                      else 'neutral'
                    })
        
        return detected
