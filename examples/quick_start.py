#!/usr/bin/env python3
"""
Quick Start Example
빠른 시작 예제

최소한의 코드로 라이브러리 사용법을 보여줍니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading.strategy import StrategyEngine
from crypto_trading.utils import generate_sample_data


def main():
    # 1. 샘플 데이터 생성 (실제로는 거래소 API에서 데이터를 가져옴)
    df = generate_sample_data(n_samples=300, seed=42)
    print(f"데이터 로드 완료: {len(df)}개 캔들")
    
    # 2. 전략 엔진 초기화
    engine = StrategyEngine()
    
    # 3. 분석 수행
    analysis = engine.analyze(df)
    
    # 4. 결과 출력
    print(f"\n현재 가격: ${analysis['current_price']:,.2f}")
    print(f"신호: {analysis['signal']['type']} ({analysis['signal']['value']:.3f})")
    print(f"추천: {analysis['recommendation']['action']}")
    print(f"손절가: ${analysis['risk_parameters']['stop_loss']:,.2f}")
    print(f"익절가: ${analysis['risk_parameters']['take_profit']:,.2f}")
    
    # 5. 프리셋 변경 (선택사항)
    engine.apply_preset('aggressive')
    analysis2 = engine.analyze(df)
    print(f"\n[공격적 전략] 신호: {analysis2['signal']['type']} ({analysis2['signal']['value']:.3f})")
    
    # 6. 가중치 커스텀 (선택사항)
    engine.set_weights({
        'category.trend': 2.0,
        'category.momentum': 1.5
    })
    analysis3 = engine.analyze(df)
    print(f"[커스텀 가중치] 신호: {analysis3['signal']['type']} ({analysis3['signal']['value']:.3f})")


if __name__ == '__main__':
    main()
