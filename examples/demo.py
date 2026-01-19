#!/usr/bin/env python3
"""
Crypto Trading Automation Demo
코인 매매 자동화 데모

이 스크립트는 라이브러리의 주요 기능을 시연합니다.
"""

import sys
import os

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading.strategy import StrategyEngine, HyperParameterManager
from crypto_trading.signals import SignalGenerator, SignalCombiner
from crypto_trading.utils import generate_sample_data


def demo_basic_analysis():
    """기본 분석 데모"""
    print("=" * 60)
    print("1. 기본 분석 데모")
    print("=" * 60)
    
    # 샘플 데이터 생성
    print("\n샘플 데이터 생성 (상승 추세)...")
    df = generate_sample_data(
        n_samples=300,
        start_price=50000,
        volatility=0.02,
        trend=0.0005,  # 약간의 상승 추세
        seed=42
    )
    
    print(f"데이터 기간: {df.index[0]} ~ {df.index[-1]}")
    print(f"시작 가격: ${df['close'].iloc[0]:,.2f}")
    print(f"현재 가격: ${df['close'].iloc[-1]:,.2f}")
    print(f"변화율: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.2f}%")
    
    # 전략 엔진 초기화
    engine = StrategyEngine()
    
    # 분석 수행
    print("\n종합 분석 중...")
    analysis = engine.analyze(df)
    
    print(f"\n현재 가격: ${analysis['current_price']:,.2f}")
    print(f"신호 값: {analysis['signal']['value']:.3f}")
    print(f"신호 타입: {analysis['signal']['type']}")
    print(f"신호 강도: {analysis['signal']['strength']}")
    print(f"신호 방향: {analysis['signal']['direction']}")
    print(f"신뢰도: {analysis['signal']['confidence']:.1%}")
    
    print("\n카테고리별 신호:")
    for category, value in analysis['signals_by_category'].items():
        direction = "↑" if value > 0 else "↓" if value < 0 else "→"
        print(f"  {category:20s}: {value:+.3f} {direction}")
    
    print("\n확인 지표:")
    print(f"  상승 지표: {len(analysis['confirming_indicators']['bullish'])}개")
    print(f"  하락 지표: {len(analysis['confirming_indicators']['bearish'])}개")
    print(f"  중립 지표: {len(analysis['confirming_indicators']['neutral'])}개")
    
    print("\n주요 레벨:")
    print(f"  지지선: {[f'${s:,.0f}' for s in analysis['levels']['support'][:3]]}")
    print(f"  저항선: {[f'${r:,.0f}' for r in analysis['levels']['resistance'][:3]]}")
    
    print("\n추천:")
    rec = analysis['recommendation']
    print(f"  행동: {rec['action']}")
    print(f"  긴급도: {rec['urgency']}")
    print(f"  이유: {rec['reasoning']}")
    
    print("\n리스크 파라미터:")
    risk = analysis['risk_parameters']
    print(f"  손절가: ${risk['stop_loss']:,.2f}")
    print(f"  익절가: ${risk['take_profit']:,.2f}")
    print(f"  ATR: ${risk['atr']:,.2f}")
    print(f"  권장 포지션 크기: {risk['position_size']:.1%}")
    
    return df, engine


def demo_presets():
    """프리셋 데모"""
    print("\n" + "=" * 60)
    print("2. 프리셋 데모")
    print("=" * 60)
    
    hp = HyperParameterManager()
    
    print("\n사용 가능한 프리셋:")
    for preset in hp.list_presets():
        print(f"  - {preset['name']}: {preset['description']}")
    
    # 공격적 프리셋 적용
    print("\n'aggressive' 프리셋 적용...")
    hp.apply_preset('aggressive')
    
    print("\n변경된 주요 파라미터:")
    print(f"  RSI 기간: {hp.get('momentum.rsi.period')}")
    print(f"  RSI 과매수: {hp.get('momentum.rsi.overbought')}")
    print(f"  SMA 단기: {hp.get('trend.sma.short_period')}")
    print(f"  손절 비율: {hp.get('risk_management.stop_loss_percent')}")
    
    return hp


def demo_weight_adjustment():
    """가중치 조절 데모"""
    print("\n" + "=" * 60)
    print("3. 가중치 조절 데모")
    print("=" * 60)
    
    df = generate_sample_data(n_samples=200, seed=42)
    
    # 신호 생성기 초기화
    generator = SignalGenerator()
    signals = generator.generate_signals_by_category(df)
    
    print("\n기본 가중치로 조합:")
    combiner = SignalCombiner()
    combined = combiner.combine(signals)
    print(f"  기본 조합 신호: {combined.iloc[-1]:.3f}")
    print(f"  기본 카테고리 가중치: {combiner.category_weights}")
    
    # 추세 강조 가중치
    print("\n추세 지표 강조 (trend: 2.0):")
    combiner.update_weights({'trend': 2.0, 'momentum': 0.8})
    combined_trend = combiner.combine(signals)
    print(f"  수정 조합 신호: {combined_trend.iloc[-1]:.3f}")
    
    # 모멘텀 강조 가중치
    print("\n모멘텀 지표 강조 (momentum: 2.0):")
    combiner.update_weights({'trend': 0.8, 'momentum': 2.0})
    combined_momentum = combiner.combine(signals)
    print(f"  수정 조합 신호: {combined_momentum.iloc[-1]:.3f}")
    
    # 영향 분석
    print("\n각 카테고리의 기여도 분석:")
    impact = combiner.get_weight_impact(signals)
    for cat in signals.keys():
        contrib = impact[f'{cat}_contribution'].iloc[-1]
        raw = impact[f'{cat}_raw'].iloc[-1]
        print(f"  {cat:20s}: raw={raw:+.3f}, contribution={contrib:+.3f}")


def demo_different_methods():
    """다양한 조합 방법 데모"""
    print("\n" + "=" * 60)
    print("4. 다양한 조합 방법 데모")
    print("=" * 60)
    
    df = generate_sample_data(n_samples=200, seed=42)
    
    generator = SignalGenerator()
    signals = generator.generate_signals_by_category(df)
    
    methods = ['weighted_average', 'voting', 'consensus', 'maximum', 'minimum']
    
    print("\n동일 데이터에 대한 다양한 조합 방법 비교:")
    for method in methods:
        combiner = SignalCombiner({'method': method})
        combined = combiner.combine(signals)
        current = combined.iloc[-1]
        classification = combiner.classify_signal(current)
        print(f"  {method:20s}: {current:+.3f} ({classification})")


def demo_backtest_signals():
    """백테스트용 신호 생성 데모"""
    print("\n" + "=" * 60)
    print("5. 백테스트용 신호 생성 데모")
    print("=" * 60)
    
    # 반전 데이터 생성 (상승 → 하락)
    print("\n반전 패턴 데이터 생성 (상승 → 하락)...")
    from crypto_trading.utils.data import generate_trending_data
    df = generate_trending_data(
        n_samples=300,
        start_price=50000,
        trend_type='reversal',
        trend_strength=0.001,
        volatility=0.015
    )
    
    engine = StrategyEngine()
    
    # 백테스트 신호 생성
    print("백테스트용 신호 생성 중...")
    result = engine.backtest_signal(df)
    
    # 신호 통계
    buy_signals = (result['position'] == 1).sum()
    sell_signals = (result['position'] == -1).sum()
    neutral_signals = (result['position'] == 0).sum()
    
    print(f"\n신호 통계 (총 {len(result)}개 캔들):")
    print(f"  매수 신호: {buy_signals}회 ({buy_signals/len(result)*100:.1f}%)")
    print(f"  매도 신호: {sell_signals}회 ({sell_signals/len(result)*100:.1f}%)")
    print(f"  중립: {neutral_signals}회 ({neutral_signals/len(result)*100:.1f}%)")
    
    # 신호 타입 분포
    print("\n신호 타입 분포:")
    print(result['signal_type'].value_counts())
    
    # 평균 신호 강도
    print(f"\n평균 신호 강도: {result['signal'].mean():.3f}")
    print(f"신호 표준편차: {result['signal'].std():.3f}")
    print(f"최대 신호: {result['signal'].max():.3f}")
    print(f"최소 신호: {result['signal'].min():.3f}")


def demo_trade_signal():
    """거래 신호 생성 데모"""
    print("\n" + "=" * 60)
    print("6. 거래 신호 생성 데모")
    print("=" * 60)
    
    df = generate_sample_data(n_samples=200, seed=42)
    
    engine = StrategyEngine()
    
    # 거래 신호 생성
    signal = engine.generate_trade_signal(df)
    
    print(f"\n거래 신호:")
    print(f"  타임스탬프: {signal.timestamp}")
    print(f"  신호 타입: {signal.signal_type.value}")
    print(f"  신호 값: {signal.signal_value:.3f}")
    print(f"  현재 가격: ${signal.price:,.2f}")
    print(f"  신뢰도: {signal.confidence:.1%}")
    print(f"  손절가: ${signal.stop_loss:,.2f}")
    print(f"  익절가: ${signal.take_profit:,.2f}")
    
    print("\n신호 발생 이유:")
    for reason in signal.reasons:
        print(f"  - {reason}")


def demo_parameter_tuning():
    """파라미터 튜닝 데모"""
    print("\n" + "=" * 60)
    print("7. 파라미터 튜닝 데모")
    print("=" * 60)
    
    engine = StrategyEngine()
    
    print("\n현재 RSI 설정:")
    rsi_period = engine.hp_manager.get('momentum.rsi.period')
    rsi_ob = engine.hp_manager.get('momentum.rsi.overbought')
    rsi_os = engine.hp_manager.get('momentum.rsi.oversold')
    print(f"  기간: {rsi_period}")
    print(f"  과매수: {rsi_ob}")
    print(f"  과매도: {rsi_os}")
    
    # 파라미터 수정
    print("\nRSI 파라미터 수정 (더 민감하게)...")
    engine.update_hyperparameters({
        'momentum.rsi.period': 7,
        'momentum.rsi.overbought': 75,
        'momentum.rsi.oversold': 25
    })
    
    print("\n수정된 RSI 설정:")
    print(f"  기간: {engine.hp_manager.get('momentum.rsi.period')}")
    print(f"  과매수: {engine.hp_manager.get('momentum.rsi.overbought')}")
    print(f"  과매도: {engine.hp_manager.get('momentum.rsi.oversold')}")
    
    # 변경 이력
    print("\n파라미터 변경 이력:")
    for change in engine.hp_manager.history[-3:]:
        print(f"  {change}")


def demo_strategy_summary():
    """전략 요약 데모"""
    print("\n" + "=" * 60)
    print("8. 전략 설정 요약")
    print("=" * 60)
    
    engine = StrategyEngine()
    engine.apply_preset('swing')
    
    summary = engine.get_strategy_summary()
    
    print(f"\n프리셋: {summary['preset']}")
    print(f"조합 방법: {summary['combination_method']}")
    
    print("\n카테고리 가중치:")
    for cat, weight in summary['category_weights'].items():
        print(f"  {cat:20s}: {weight:.2f}")
    
    print("\n임계값:")
    for threshold, value in summary['thresholds'].items():
        print(f"  {threshold:15s}: {value:.2f}")
    
    print("\n리스크 관리:")
    for param, value in summary['risk_management'].items():
        print(f"  {param:20s}: {value}")
    
    print("\n활성화된 지표:")
    for category, indicators in summary['enabled_indicators'].items():
        if indicators:
            print(f"  {category}: {', '.join(indicators)}")


def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("    코인 매매 자동화 라이브러리 데모")
    print("=" * 60)
    
    # 1. 기본 분석
    demo_basic_analysis()
    
    # 2. 프리셋
    demo_presets()
    
    # 3. 가중치 조절
    demo_weight_adjustment()
    
    # 4. 다양한 조합 방법
    demo_different_methods()
    
    # 5. 백테스트 신호
    demo_backtest_signals()
    
    # 6. 거래 신호
    demo_trade_signal()
    
    # 7. 파라미터 튜닝
    demo_parameter_tuning()
    
    # 8. 전략 요약
    demo_strategy_summary()
    
    print("\n" + "=" * 60)
    print("    데모 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
