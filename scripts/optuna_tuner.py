"""
Hyperparameter Tuner with Optuna
Optuna 기반 하이퍼파라미터 최적화

베이지안 최적화(TPE)를 사용하여 효율적으로 최적 파라미터를 탐색합니다.
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# Optuna import
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("Optuna가 설치되어 있지 않습니다. 설치 중...")
    os.system("pip install optuna -q")
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading.strategy import StrategyEngine
from scripts.backtester import Backtester, BacktestResult
from scripts.data_loader import load_ohlcv, get_available_symbols

# 로깅 설정
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 데이터 분할 기간
TRAIN_START = '2024-01-01'
TRAIN_END = '2024-12-31'
VALID_START = '2025-01-01'
VALID_END = '2025-09-30'
TEST_START = '2025-10-01'
TEST_END = '2026-01-19'

# 결과 저장 경로
RESULTS_DIR = Path(__file__).parent.parent / 'data' / 'tuning_results'


def create_objective(train_data: pd.DataFrame, valid_data: pd.DataFrame):
    """
    Optuna Objective 함수 생성
    """
    backtester = Backtester()
    
    def objective(trial: optuna.Trial) -> float:
        # 하이퍼파라미터 샘플링
        params = {
            # 모멘텀 지표
            'momentum.rsi.period': trial.suggest_int('rsi_period', 7, 28),
            'momentum.rsi.overbought': trial.suggest_int('rsi_overbought', 65, 85),
            'momentum.rsi.oversold': trial.suggest_int('rsi_oversold', 15, 35),
            
            # 추세 지표
            'trend.ema.short_period': trial.suggest_int('ema_short', 3, 20),
            'trend.macd.fast_period': trial.suggest_int('macd_fast', 8, 20),
            'trend.macd.slow_period': trial.suggest_int('macd_slow', 20, 40),
            
            # 볼린저 밴드
            'volatility.bollinger.period': trial.suggest_int('bb_period', 10, 30),
            'volatility.bollinger.std_dev': trial.suggest_float('bb_std', 1.5, 3.0),
            
            # 카테고리 가중치
            'signal_combination.category_weights.trend': trial.suggest_float('weight_trend', 0.5, 2.0),
            'signal_combination.category_weights.momentum': trial.suggest_float('weight_momentum', 0.5, 2.0),
            'signal_combination.category_weights.volatility': trial.suggest_float('weight_volatility', 0.3, 1.5),
            'signal_combination.category_weights.volume': trial.suggest_float('weight_volume', 0.3, 1.5),
            
            # 임계값
            'signal_combination.thresholds.buy': trial.suggest_float('threshold_buy', 0.1, 0.5),
            'signal_combination.thresholds.sell': trial.suggest_float('threshold_sell', -0.5, -0.1),
        }
        
        try:
            # 엔진 생성 및 파라미터 설정
            engine = StrategyEngine()
            for path, value in params.items():
                engine.hp_manager.set(path, value, record_history=False)
            engine._reinitialize()
            
            # Training 백테스트
            train_result, _ = backtester.run(train_data, engine)
            
            # Pruning: Train 성과가 너무 나쁘면 조기 종료
            if train_result.sharpe_ratio < -3:
                raise optuna.TrialPruned()
            
            # Validation 백테스트
            valid_result, _ = backtester.run(valid_data, engine)
            
            # 최적화 목표: Validation Sharpe (높을수록 좋음)
            score = valid_result.sharpe_ratio
            
            # 오버피팅 패널티
            overfit_penalty = max(0, (train_result.sharpe_ratio - valid_result.sharpe_ratio) * 0.3)
            score -= overfit_penalty
            
            # 거래가 너무 적으면 패널티
            if valid_result.total_trades < 10:
                score -= 1.0
            
            # 메트릭 저장
            trial.set_user_attr('train_return', train_result.total_return)
            trial.set_user_attr('train_sharpe', train_result.sharpe_ratio)
            trial.set_user_attr('valid_return', valid_result.total_return)
            trial.set_user_attr('valid_sharpe', valid_result.sharpe_ratio)
            trial.set_user_attr('train_trades', train_result.total_trades)
            trial.set_user_attr('valid_trades', valid_result.total_trades)
            
            return score
            
        except Exception as e:
            raise optuna.TrialPruned()
    
    return objective


def run_optuna_tuning(
    symbol: str = 'BTC/USDT',
    timeframe: str = '4h',
    n_trials: int = 50,
    timeout: int = None,
):
    """
    Optuna를 사용한 하이퍼파라미터 최적화
    """
    print("=" * 70)
    print(f"  Optuna 하이퍼파라미터 최적화: {symbol} ({timeframe})")
    print("=" * 70)
    
    # 1. 데이터 로드 및 분할
    print("\n[1] 데이터 분할")
    print("-" * 50)
    df = load_ohlcv(symbol, timeframe)
    train_data = df[TRAIN_START:TRAIN_END]
    valid_data = df[VALID_START:VALID_END]
    test_data = df[TEST_START:TEST_END]
    
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
    
    print(f"  {'세트':12} {'수익률':>10} {'Sharpe':>10}")
    print(f"  {'Train':12} {baseline_train.total_return:>10.2%} {baseline_train.sharpe_ratio:>10.2f}")
    print(f"  {'Validation':12} {baseline_valid.total_return:>10.2%} {baseline_valid.sharpe_ratio:>10.2f}")
    print(f"  {'Test':12} {baseline_test.total_return:>10.2%} {baseline_test.sharpe_ratio:>10.2f}")
    
    # 3. Optuna 최적화
    print(f"\n[3] Optuna 최적화 ({n_trials} trials)")
    print("-" * 50)
    
    # Study 생성
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=f'{symbol}_{timeframe}_optimization'
    )
    
    # Objective 생성 및 최적화 실행
    objective = create_objective(train_data, valid_data)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )
    
    print(f"  완료: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])} trials")
    
    # 4. 최적 파라미터 추출
    print(f"\n[4] 최적화 결과")
    print("-" * 50)
    print(f"  완료된 trials: {len(study.trials)}")
    print(f"  Best Score: {study.best_value:.3f}")
    
    # Optuna 파라미터명 → 실제 경로 매핑
    param_mapping = {
        'rsi_period': 'momentum.rsi.period',
        'rsi_overbought': 'momentum.rsi.overbought',
        'rsi_oversold': 'momentum.rsi.oversold',
        'ema_short': 'trend.ema.short_period',
        'macd_fast': 'trend.macd.fast_period',
        'macd_slow': 'trend.macd.slow_period',
        'bb_period': 'volatility.bollinger.period',
        'bb_std': 'volatility.bollinger.std_dev',
        'weight_trend': 'signal_combination.category_weights.trend',
        'weight_momentum': 'signal_combination.category_weights.momentum',
        'weight_volatility': 'signal_combination.category_weights.volatility',
        'weight_volume': 'signal_combination.category_weights.volume',
        'threshold_buy': 'signal_combination.thresholds.buy',
        'threshold_sell': 'signal_combination.thresholds.sell',
    }
    
    best_params = {
        param_mapping[k]: v 
        for k, v in study.best_params.items()
    }
    
    print("\n최적 파라미터:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    # 5. 최적 파라미터로 Test 세트 평가
    print(f"\n[5] 최적 파라미터 성능")
    print("-" * 50)
    
    engine = StrategyEngine()
    for path, value in best_params.items():
        engine.hp_manager.set(path, value, record_history=False)
    engine._reinitialize()
    
    opt_train, _ = backtester.run(train_data, engine)
    opt_valid, _ = backtester.run(valid_data, engine)
    opt_test, trades_df = backtester.run(test_data, engine)
    
    print(f"  {'세트':12} {'수익률':>10} {'Sharpe':>10} {'MDD':>10} {'승률':>8} {'거래':>6}")
    print(f"  {'Train':12} {opt_train.total_return:>10.2%} {opt_train.sharpe_ratio:>10.2f} {opt_train.max_drawdown:>10.2%} {opt_train.win_rate:>8.1%} {opt_train.total_trades:>6}")
    print(f"  {'Validation':12} {opt_valid.total_return:>10.2%} {opt_valid.sharpe_ratio:>10.2f} {opt_valid.max_drawdown:>10.2%} {opt_valid.win_rate:>8.1%} {opt_valid.total_trades:>6}")
    print(f"  {'Test':12} {opt_test.total_return:>10.2%} {opt_test.sharpe_ratio:>10.2f} {opt_test.max_drawdown:>10.2%} {opt_test.win_rate:>8.1%} {opt_test.total_trades:>6}")
    
    # 6. 개선율
    print(f"\n[6] 개선율 (vs 기본)")
    print("-" * 50)
    print(f"  Train:      {baseline_train.sharpe_ratio:.2f} → {opt_train.sharpe_ratio:.2f} ({opt_train.sharpe_ratio - baseline_train.sharpe_ratio:+.2f})")
    print(f"  Validation: {baseline_valid.sharpe_ratio:.2f} → {opt_valid.sharpe_ratio:.2f} ({opt_valid.sharpe_ratio - baseline_valid.sharpe_ratio:+.2f})")
    print(f"  Test:       {baseline_test.sharpe_ratio:.2f} → {opt_test.sharpe_ratio:.2f} ({opt_test.sharpe_ratio - baseline_test.sharpe_ratio:+.2f})")
    
    # 7. 결과 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    result_file = RESULTS_DIR / f'optuna_best_{symbol.replace("/", "_")}_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump({
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': timestamp,
            'n_trials': n_trials,
            'best_score': study.best_value,
            'best_params': best_params,
            'baseline': {
                'train_sharpe': baseline_train.sharpe_ratio,
                'valid_sharpe': baseline_valid.sharpe_ratio,
                'test_sharpe': baseline_test.sharpe_ratio,
            },
            'optimized': {
                'train': opt_train.to_dict(),
                'valid': opt_valid.to_dict(),
                'test': opt_test.to_dict(),
            }
        }, f, indent=2)
    
    print(f"\n결과 저장: {result_file}")
    
    print("\n" + "=" * 70)
    print("  최적화 완료!")
    print("=" * 70)
    
    return study, best_params, opt_test


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Optuna 하이퍼파라미터 최적화')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='심볼')
    parser.add_argument('--timeframe', type=str, default='4h', help='타임프레임')
    parser.add_argument('--trials', type=int, default=50, help='최적화 시도 횟수')
    parser.add_argument('--timeout', type=int, default=None, help='최대 실행 시간(초)')
    
    args = parser.parse_args()
    
    run_optuna_tuning(
        symbol=args.symbol,
        timeframe=args.timeframe,
        n_trials=args.trials,
        timeout=args.timeout,
    )
