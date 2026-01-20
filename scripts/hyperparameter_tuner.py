"""
Hyperparameter Tuner
하이퍼파라미터 튜너

Train/Validation/Test 분할로 하이퍼파라미터를 최적화합니다.
"""

import sys
import os
import json
import itertools
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading.strategy import StrategyEngine, HyperParameterManager
from scripts.backtester import Backtester, BacktestResult
from scripts.data_loader import load_ohlcv, get_available_symbols


# 데이터 분할 기간
TRAIN_START = '2024-01-01'
TRAIN_END = '2024-12-31'
VALID_START = '2025-01-01'
VALID_END = '2025-09-30'
TEST_START = '2025-10-01'
TEST_END = '2026-01-19'

# 결과 저장 경로
RESULTS_DIR = Path(__file__).parent.parent / 'data' / 'tuning_results'


def split_data(symbol: str, timeframe: str = '4h') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train/Validation/Test 데이터 분할
    """
    df = load_ohlcv(symbol, timeframe)
    
    train = df[TRAIN_START:TRAIN_END]
    valid = df[VALID_START:VALID_END]
    test = df[TEST_START:TEST_END]
    
    return train, valid, test


def get_search_space() -> Dict[str, List]:
    """
    탐색할 하이퍼파라미터 공간 정의
    
    전체 공간을 탐색하면 시간이 오래 걸리므로
    핵심 파라미터만 선별하여 탐색합니다.
    """
    return {
        # 핵심 지표 파라미터
        'momentum.rsi.period': [7, 14, 21],
        'momentum.rsi.overbought': [70, 75, 80],
        'momentum.rsi.oversold': [20, 25, 30],
        
        # 추세 파라미터
        'trend.ema.short_period': [5, 9, 12],
        'trend.macd.fast_period': [8, 12, 16],
        
        # 볼린저 밴드
        'volatility.bollinger.period': [15, 20, 25],
        
        # 가중치 (좁은 범위로 제한)
        'signal_combination.category_weights.trend': [0.8, 1.0, 1.2],
        'signal_combination.category_weights.momentum': [0.8, 1.0, 1.2],
        
        # 임계값
        'signal_combination.thresholds.buy': [0.15, 0.2, 0.25],
        'signal_combination.thresholds.sell': [-0.25, -0.2, -0.15],
    }


def grid_search(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    search_space: Dict[str, List],
    max_combinations: int = 100,
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Grid Search를 통한 하이퍼파라미터 탐색
    
    조합 수가 많을 경우 랜덤 샘플링으로 제한합니다.
    """
    # 모든 조합 생성
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    all_combinations = list(itertools.product(*param_values))
    
    print(f"총 조합 수: {len(all_combinations)}")
    
    # 조합 수가 많으면 랜덤 샘플링
    if len(all_combinations) > max_combinations:
        np.random.seed(42)
        indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
        combinations = [all_combinations[i] for i in indices]
        print(f"랜덤 샘플링: {max_combinations}개 조합 선택")
    else:
        combinations = all_combinations
    
    backtester = Backtester()
    results = []
    best_score = float('-inf')
    best_params = None
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        try:
            # 엔진 생성 및 파라미터 설정
            engine = StrategyEngine()
            for path, value in params.items():
                engine.hp_manager.set(path, value, record_history=False)
            engine._reinitialize()
            
            # Training 백테스트
            train_result, _ = backtester.run(train_data, engine)
            
            # Validation 백테스트
            valid_result, _ = backtester.run(valid_data, engine)
            
            # 스코어 = Validation Sharpe (오버피팅 방지)
            score = valid_result.sharpe_ratio
            
            # 오버피팅 패널티: Train이 Validation보다 너무 좋으면 패널티
            overfit_ratio = train_result.sharpe_ratio / (valid_result.sharpe_ratio + 0.001)
            if overfit_ratio > 2.0:
                score *= 0.7  # 30% 패널티
            
            results.append({
                'params': params,
                'train_return': train_result.total_return,
                'train_sharpe': train_result.sharpe_ratio,
                'valid_return': valid_result.total_return,
                'valid_sharpe': valid_result.sharpe_ratio,
                'train_trades': train_result.total_trades,
                'valid_trades': valid_result.total_trades,
                'score': score,
            })
            
            if score > best_score:
                best_score = score
                best_params = params
            
            # 진행 상황 출력
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[{i+1}/{len(combinations)}] Best Score: {best_score:.3f}")
                
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue
    
    return best_params, results


def evaluate_on_test(
    test_data: pd.DataFrame,
    params: Dict[str, Any],
) -> BacktestResult:
    """
    최적 파라미터로 Test 세트 평가
    """
    engine = StrategyEngine()
    for path, value in params.items():
        engine.hp_manager.set(path, value, record_history=False)
    engine._reinitialize()
    
    backtester = Backtester()
    result, trades_df = backtester.run(test_data, engine)
    
    return result, trades_df


def save_results(
    best_params: Dict[str, Any],
    all_results: List[Dict],
    train_result: BacktestResult,
    valid_result: BacktestResult,
    test_result: BacktestResult,
    symbol: str,
):
    """결과 저장"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 최적 파라미터 저장
    best_params_file = RESULTS_DIR / f'best_params_{symbol.replace("/", "_")}_{timestamp}.json'
    with open(best_params_file, 'w') as f:
        json.dump({
            'symbol': symbol,
            'timestamp': timestamp,
            'best_params': best_params,
            'train_metrics': train_result.to_dict(),
            'valid_metrics': valid_result.to_dict(),
            'test_metrics': test_result.to_dict(),
        }, f, indent=2)
    
    # 전체 탐색 결과 저장
    results_df = pd.DataFrame([
        {**r['params'], 
         'train_return': r['train_return'],
         'train_sharpe': r['train_sharpe'],
         'valid_return': r['valid_return'],
         'valid_sharpe': r['valid_sharpe'],
         'score': r['score']}
        for r in all_results
    ])
    results_file = RESULTS_DIR / f'search_results_{symbol.replace("/", "_")}_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    
    print(f"\n결과 저장됨:")
    print(f"  - {best_params_file}")
    print(f"  - {results_file}")


def run_tuning(symbol: str = 'BTC/USDT', timeframe: str = '4h'):
    """
    전체 튜닝 파이프라인 실행
    """
    print("=" * 70)
    print(f"  하이퍼파라미터 튜닝: {symbol} ({timeframe})")
    print("=" * 70)
    
    # 1. 데이터 분할
    print("\n[1] 데이터 분할")
    print("-" * 50)
    train_data, valid_data, test_data = split_data(symbol, timeframe)
    print(f"  Train:      {TRAIN_START} ~ {TRAIN_END} ({len(train_data)} 캔들)")
    print(f"  Validation: {VALID_START} ~ {VALID_END} ({len(valid_data)} 캔들)")
    print(f"  Test:       {TEST_START} ~ {TEST_END} ({len(test_data)} 캔들)")
    
    # 2. 기본 성능 측정
    print("\n[2] 기본 파라미터 성능")
    print("-" * 50)
    engine = StrategyEngine()
    backtester = Backtester()
    
    baseline_train, _ = backtester.run(train_data, engine)
    baseline_valid, _ = backtester.run(valid_data, engine)
    baseline_test, _ = backtester.run(test_data, engine)
    
    print(f"  {'세트':12} {'수익률':>10} {'Sharpe':>10} {'MDD':>10} {'거래수':>8}")
    print(f"  {'Train':12} {baseline_train.total_return:>10.2%} {baseline_train.sharpe_ratio:>10.2f} {baseline_train.max_drawdown:>10.2%} {baseline_train.total_trades:>8}")
    print(f"  {'Validation':12} {baseline_valid.total_return:>10.2%} {baseline_valid.sharpe_ratio:>10.2f} {baseline_valid.max_drawdown:>10.2%} {baseline_valid.total_trades:>8}")
    print(f"  {'Test':12} {baseline_test.total_return:>10.2%} {baseline_test.sharpe_ratio:>10.2f} {baseline_test.max_drawdown:>10.2%} {baseline_test.total_trades:>8}")
    
    # 3. Grid Search
    print("\n[3] 하이퍼파라미터 탐색")
    print("-" * 50)
    search_space = get_search_space()
    best_params, all_results = grid_search(
        train_data, 
        valid_data, 
        search_space,
        max_combinations=100
    )
    
    print(f"\n최적 파라미터:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    # 4. 최적 파라미터로 전체 평가
    print("\n[4] 최적 파라미터 성능")
    print("-" * 50)
    
    engine = StrategyEngine()
    for path, value in best_params.items():
        engine.hp_manager.set(path, value, record_history=False)
    engine._reinitialize()
    
    opt_train, _ = backtester.run(train_data, engine)
    opt_valid, _ = backtester.run(valid_data, engine)
    opt_test, trades_df = backtester.run(test_data, engine)
    
    print(f"  {'세트':12} {'수익률':>10} {'Sharpe':>10} {'MDD':>10} {'승률':>8} {'거래수':>8}")
    print(f"  {'Train':12} {opt_train.total_return:>10.2%} {opt_train.sharpe_ratio:>10.2f} {opt_train.max_drawdown:>10.2%} {opt_train.win_rate:>8.1%} {opt_train.total_trades:>8}")
    print(f"  {'Validation':12} {opt_valid.total_return:>10.2%} {opt_valid.sharpe_ratio:>10.2f} {opt_valid.max_drawdown:>10.2%} {opt_valid.win_rate:>8.1%} {opt_valid.total_trades:>8}")
    print(f"  {'Test':12} {opt_test.total_return:>10.2%} {opt_test.sharpe_ratio:>10.2f} {opt_test.max_drawdown:>10.2%} {opt_test.win_rate:>8.1%} {opt_test.total_trades:>8}")
    
    # 5. 개선율 계산
    print("\n[5] 개선율 (vs 기본)")
    print("-" * 50)
    train_improvement = (opt_train.sharpe_ratio - baseline_train.sharpe_ratio)
    valid_improvement = (opt_valid.sharpe_ratio - baseline_valid.sharpe_ratio)
    test_improvement = (opt_test.sharpe_ratio - baseline_test.sharpe_ratio)
    
    print(f"  Train Sharpe:      {baseline_train.sharpe_ratio:.2f} → {opt_train.sharpe_ratio:.2f} ({train_improvement:+.2f})")
    print(f"  Validation Sharpe: {baseline_valid.sharpe_ratio:.2f} → {opt_valid.sharpe_ratio:.2f} ({valid_improvement:+.2f})")
    print(f"  Test Sharpe:       {baseline_test.sharpe_ratio:.2f} → {opt_test.sharpe_ratio:.2f} ({test_improvement:+.2f})")
    
    # 6. 오버피팅 검증
    print("\n[6] 오버피팅 검증")
    print("-" * 50)
    train_valid_gap = abs(opt_train.sharpe_ratio - opt_valid.sharpe_ratio)
    valid_test_gap = abs(opt_valid.sharpe_ratio - opt_test.sharpe_ratio)
    
    print(f"  Train-Validation Gap: {train_valid_gap:.2f}")
    print(f"  Validation-Test Gap:  {valid_test_gap:.2f}")
    
    if train_valid_gap < 0.5 and valid_test_gap < 0.5:
        print("  ✅ 오버피팅 위험 낮음")
    elif train_valid_gap < 1.0 and valid_test_gap < 1.0:
        print("  ⚠️ 오버피팅 위험 중간")
    else:
        print("  ❌ 오버피팅 위험 높음")
    
    # 7. 결과 저장
    save_results(best_params, all_results, opt_train, opt_valid, opt_test, symbol)
    
    print("\n" + "=" * 70)
    print("  튜닝 완료!")
    print("=" * 70)
    
    return best_params, opt_test


def run_multi_symbol_tuning(symbols: List[str] = None, timeframe: str = '4h'):
    """
    여러 코인에 대해 튜닝 실행
    """
    if symbols is None:
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    results = {}
    for symbol in symbols:
        try:
            best_params, test_result = run_tuning(symbol, timeframe)
            results[symbol] = {
                'params': best_params,
                'test_result': test_result.to_dict(),
            }
        except Exception as e:
            print(f"Error tuning {symbol}: {e}")
            continue
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='하이퍼파라미터 튜닝')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='심볼')
    parser.add_argument('--timeframe', type=str, default='4h', help='타임프레임')
    parser.add_argument('--multi', action='store_true', help='여러 코인 튜닝')
    
    args = parser.parse_args()
    
    if args.multi:
        run_multi_symbol_tuning()
    else:
        run_tuning(args.symbol, args.timeframe)
