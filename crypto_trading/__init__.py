"""
Crypto Trading Automation Library
코인 매매 자동화 라이브러리

이 패키지는 기술적 분석 기반의 코인 매매 자동화 시스템을 제공합니다.

주요 모듈:
- indicators: 기술적 지표 계산 (추세, 모멘텀, 변동성, 거래량)
- patterns: 캔들스틱 및 차트 패턴 인식
- signals: 매매 신호 생성 및 조합
- strategy: 전략 엔진 및 하이퍼파라미터 관리
"""

__version__ = "0.1.0"
__author__ = "Auto Finance Team"

from .indicators import TrendIndicators, MomentumIndicators, VolatilityIndicators, VolumeIndicators
from .patterns import CandlestickPatterns, ChartPatterns
from .signals import SignalGenerator, SignalCombiner
from .strategy import StrategyEngine, HyperParameterManager
