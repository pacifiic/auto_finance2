"""
RL2 Pair Trading - Training & Evaluation (Paper-Based)
논문 기반 RL2 페어 트레이딩 학습 및 평가

A2C 에이전트 + Zone 기반 보상 + 연속 액션 공간
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_trading.pair_trading_env import PairTradingEnvRL2
from rl_trading.a2c_agent import A2CAgent
from scripts.data_loader import load_ohlcv

# 경로 설정
MODELS_DIR = Path(__file__).parent / 'models'
RESULTS_DIR = Path(__file__).parent.parent / 'data' / 'rl_results'


def train(
    asset1: str = 'BTC/USDT',
    asset2: str = 'ETH/USDT',
    timeframe: str = '4h',
    train_start: str = '2024-01-01',
    train_end: str = '2024-12-31',
    n_episodes: int = 1000,
    n_steps: int = 20,  # A2C 업데이트 간격
    verbose: bool = True,
):
    """
    A2C 에이전트 학습 (논문 기반)
    """
    print("=" * 70)
    print("  RL2 Pair Trading - A2C Training (Paper-Based)")
    print("=" * 70)
    
    # 데이터 로드
    print(f"\n[1] 데이터 로드")
    df1 = load_ohlcv(asset1, timeframe, start_date=train_start, end_date=train_end)
    df2 = load_ohlcv(asset2, timeframe, start_date=train_start, end_date=train_end)
    print(f"  {asset1}: {len(df1)} rows")
    print(f"  {asset2}: {len(df2)} rows")
    
    # 환경 생성 (논문 설정)
    print(f"\n[2] RL2 환경 생성 (Paper Settings)")
    env = PairTradingEnvRL2(
        df1, df2,
        window_size=20,
        open_threshold=2.0,     # 논문: 2 std
        close_threshold=0.5,    # 논문: 0.5 std
        transaction_cost=0.0002,  # 0.02%
        action_reward_weight=0.5,
    )
    print(f"  State size: {env.observation_space.shape[0]}")
    print(f"  Action space: Continuous [-1, 1]")
    print(f"  Open/Close Threshold: {env.open_threshold}/{env.close_threshold}")
    print(f"  Transaction cost: {env.transaction_cost*100:.2f}%")
    
    # A2C 에이전트 생성
    print(f"\n[3] A2C 에이전트 생성")
    agent = A2CAgent(
        state_size=env.observation_space.shape[0],
        hidden_size=128,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        value_coef=0.5,
        entropy_coef=0.01,
        n_steps=n_steps,
    )
    print(f"  Device: {agent.device}")
    print(f"  N-steps: {n_steps}")
    
    # 학습
    print(f"\n[4] 학습 시작 ({n_episodes} episodes)")
    print("-" * 60)
    
    rewards_history = []
    capital_history = []
    best_capital = 0
    best_episode = 0
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        
        while True:
            # 액션 선택
            action, value = agent.select_action(state, deterministic=False)
            
            # 환경 스텝
            next_state, reward, done, _, info = env.step(np.array([action]))
            
            # 경험 저장
            agent.store(state, action, reward, value, done)
            
            total_reward += reward
            step_count += 1
            
            # N-step 학습
            if step_count % n_steps == 0 or done:
                agent.learn(next_state)
            
            state = next_state
            
            if done:
                break
        
        rewards_history.append(total_reward)
        capital_history.append(info['capital'])
        
        # 최고 성능 저장
        if info['capital'] > best_capital:
            best_capital = info['capital']
            best_episode = episode
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            agent.save(str(MODELS_DIR / 'best_a2c_model.pt'))
        
        # 진행 상황 출력
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_capital = np.mean(capital_history[-100:])
            win_rate = info['win_rate'] * 100
            print(f"  Episode {episode+1:4d} | "
                  f"Reward: {avg_reward:>7.1f} | "
                  f"Capital: ${avg_capital:>9,.0f} | "
                  f"Trades: {info['total_trades']:3d} | "
                  f"WinRate: {win_rate:5.1f}%")
    
    print("-" * 60)
    print(f"  Best capital: ${best_capital:,.0f} (Episode {best_episode})")
    print(f"  Model saved: {MODELS_DIR / 'best_a2c_model.pt'}")
    
    return agent, rewards_history, capital_history


def evaluate(
    agent: A2CAgent,
    asset1: str = 'BTC/USDT',
    asset2: str = 'ETH/USDT',
    timeframe: str = '4h',
    test_start: str = '2025-01-01',
    test_end: str = '2025-09-30',
):
    """학습된 에이전트 평가"""
    print(f"\n[5] 테스트 세트 평가")
    print("-" * 60)
    
    # 데이터 로드
    df1 = load_ohlcv(asset1, timeframe, start_date=test_start, end_date=test_end)
    df2 = load_ohlcv(asset2, timeframe, start_date=test_start, end_date=test_end)
    print(f"  Test period: {test_start} ~ {test_end}")
    print(f"  Data size: {len(df1)} rows")
    
    # 환경 생성
    env = PairTradingEnvRL2(
        df1, df2,
        window_size=20,
        open_threshold=2.0,
        close_threshold=0.5,
        transaction_cost=0.0002,
    )
    
    # 평가 실행 (deterministic)
    state, _ = env.reset()
    total_reward = 0
    positions = []
    
    while True:
        action, _ = agent.select_action(state, deterministic=True)
        next_state, reward, done, _, info = env.step(np.array([action]))
        total_reward += reward
        positions.append(info['position'])
        state = next_state
        if done:
            break
    
    # 결과
    initial = env.initial_capital
    final = info['capital']
    ret = (final - initial) / initial * 100
    
    print(f"\n  📊 Results:")
    print(f"    Initial capital: ${initial:,.0f}")
    print(f"    Final capital: ${final:,.0f}")
    print(f"    Return: {ret:+.2f}%")
    print(f"    Total trades: {info['total_trades']}")
    print(f"    Win rate: {info['win_rate']*100:.1f}%")
    
    # 포지션 분석
    positions = np.array(positions)
    long_ratio = (positions > 0.1).mean() * 100
    short_ratio = (positions < -0.1).mean() * 100
    neutral_ratio = (np.abs(positions) <= 0.1).mean() * 100
    
    print(f"\n  📈 Position Distribution:")
    print(f"    Long: {long_ratio:.1f}%")
    print(f"    Short: {short_ratio:.1f}%")
    print(f"    Neutral: {neutral_ratio:.1f}%")
    
    # Buy & Hold 비교
    price_change_1 = (df1['close'].iloc[-1] / df1['close'].iloc[0] - 1) * 100
    price_change_2 = (df2['close'].iloc[-1] / df2['close'].iloc[0] - 1) * 100
    
    print(f"\n  📌 Comparison (Buy & Hold):")
    print(f"    {asset1}: {price_change_1:+.2f}%")
    print(f"    {asset2}: {price_change_2:+.2f}%")
    
    outperform = ret > max(price_change_1, price_change_2)
    if outperform:
        print(f"\n  ✅ Outperformed Buy & Hold!")
    else:
        print(f"\n  ⚠️ Underperformed Buy & Hold")
    
    return {
        'return': ret,
        'final_capital': final,
        'total_trades': info['total_trades'],
        'win_rate': info['win_rate'],
        'bh_asset1': price_change_1,
        'bh_asset2': price_change_2,
        'outperformed': outperform,
    }


def run_full_pipeline():
    """전체 파이프라인 실행"""
    # 학습
    agent, rewards, capitals = train(
        asset1='BTC/USDT',
        asset2='ETH/USDT',
        timeframe='4h',
        train_start='2024-01-01',
        train_end='2024-12-31',
        n_episodes=1000,
        n_steps=20,
    )
    
    # 평가
    results = evaluate(
        agent,
        asset1='BTC/USDT',
        asset2='ETH/USDT',
        timeframe='4h',
        test_start='2025-01-01',
        test_end='2025-09-30',
    )
    
    # 결과 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = RESULTS_DIR / f'rl2_a2c_{timestamp}.json'
    
    with open(result_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'method': 'RL2 A2C (Paper-Based)',
            'train_period': '2024-01-01 ~ 2024-12-31',
            'test_period': '2025-01-01 ~ 2025-09-30',
            'results': results,
        }, f, indent=2)
    
    print(f"\n결과 저장: {result_file}")
    
    print("\n" + "=" * 70)
    print("  Training & Evaluation Complete!")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RL2 Pair Trading (Paper-Based)')
    parser.add_argument('--episodes', type=int, default=1000, help='학습 에피소드 수')
    parser.add_argument('--asset1', type=str, default='BTC/USDT', help='첫 번째 자산')
    parser.add_argument('--asset2', type=str, default='ETH/USDT', help='두 번째 자산')
    
    args = parser.parse_args()
    
    run_full_pipeline()
