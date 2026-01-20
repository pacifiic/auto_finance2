"""
RL2 Pair Trading Training (Sentiment & Macro)
심리/거시 데이터 포함 학습 스크립트
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_loader import load_ohlcv
from rl_trading.pair_trading_env import PairTradingEnvRL2
from rl_trading.a2c_agent import A2CAgent

MODELS_DIR = Path(__file__).parent / 'models'
RESULTS_DIR = Path(__file__).parent.parent / 'data' / 'rl_results'
SENTIMENT_DIR = Path(__file__).parent.parent / 'data' / 'sentiment'
MACRO_DIR = Path(__file__).parent.parent / 'data' / 'macro'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_sentiment_macro():
    """심리 및 거시 데이터 로드"""
    print("\n[Loading Sentiment & Macro Data]")
    
    # Fear & Greed
    fng_path = SENTIMENT_DIR / 'fear_greed_index.csv'
    if fng_path.exists():
        fng = pd.read_csv(fng_path, parse_dates=['timestamp'], index_col='timestamp')
        print(f"  F&G Index: {len(fng)} rows")
    else:
        print("  ⚠️ F&G Index not found")
        fng = None
        
    # Interest Rates
    rate_path = MACRO_DIR / 'us_treasury_10y.csv'
    if rate_path.exists():
        rate = pd.read_csv(rate_path, parse_dates=['date'])
        rate = rate.set_index('date').rename(columns={'value': 'value_10y'})
    else:
        rate = None
        
    # Yield Spread
    spread_path = MACRO_DIR / 'us_yield_curve_spread.csv'
    if spread_path.exists():
        spread = pd.read_csv(spread_path, parse_dates=['date'])
        spread = spread.set_index('date')[['spread']]
    else:
        spread = None
        
    # Macro 병합
    if rate is not None and spread is not None:
        macro = rate.join(spread, how='outer').fillna(method='ffill')
        print(f"  Macro Data: {len(macro)} rows")
    else:
        print("  ⚠️ Macro Data incomplete")
        macro = None
        
    return fng, macro


def train(n_episodes=500):
    print("=" * 70)
    print("  RL2 Training with Sentiment & Macro")
    print("=" * 70)
    
    # 1. 데이터 로드 (2020~2024 Train)
    btc = load_ohlcv('BTC/USDT', '4h', start_date='2020-01-01', end_date='2024-12-31')
    eth = load_ohlcv('ETH/USDT', '4h', start_date='2020-01-01', end_date='2024-12-31')
    fng, macro = load_sentiment_macro()
    
    # 2. 환경 생성
    env = PairTradingEnvRL2(
        btc, eth, 
        sentiment_df=fng, 
        macro_df=macro,
        window_size=20,
        open_threshold=1.0,  # 더 낮춰서 적극적 거래 유도
        close_threshold=0.3,
        transaction_cost=0.0002,
    )
    
    print(f"\nEnvironment Initialized:")
    print(f"  State dim: {env.observation_space.shape[0]} (Includes Sentiment/Macro)")
    print(f"  Action dim: {env.action_space.shape[0]}")
    
    # 3. 에이전트 생성
    agent = A2CAgent(
        state_size=env.observation_space.shape[0],
        hidden_size=128,
        learning_rate=0.0003,
        gamma=0.99,
        n_steps=20,
    )
    
    # 4. 학습
    print(f"\nStarting Training ({n_episodes} eps)...")
    best_capital = 0
    scores = []
    
    for ep in range(n_episodes):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            action, value = agent.select_action(state)
            next_state, reward, done, _, info = env.step(np.array([action]))
            
            agent.store(state, action, reward, value, done)
            if len(agent.states) >= agent.n_steps or done:
                agent.learn(next_state)
                
            score += reward
            state = next_state
        
        scores.append(score)
        
        if info['capital'] > best_capital:
            best_capital = info['capital']
            agent.save(str(MODELS_DIR / 'best_sentiment_model.pt'))
            
        if (ep + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"  Ep {ep+1:3d} | Score: {avg_score:>7.1f} | Cap: ${info['capital']:>8.0f} | Trades: {info['total_trades']}")
            
    print(f"\nTraining Complete. Best Capital: ${best_capital:,.0f}")
    return agent


def evaluate(agent):
    print("\n[Evaluation 2025-2026]")
    
    # Test Data (2025~)
    btc = load_ohlcv('BTC/USDT', '4h', start_date='2025-01-01', end_date='2026-01-20')
    eth = load_ohlcv('ETH/USDT', '4h', start_date='2025-01-01', end_date='2026-01-20')
    fng, macro = load_sentiment_macro()
    
    env = PairTradingEnvRL2(
        btc, eth, 
        sentiment_df=fng, 
        macro_df=macro,
        window_size=20,
        open_threshold=1.0,
        close_threshold=0.3,
        transaction_cost=0.0002,
    )
    
    state, _ = env.reset()
    done = False
    
    while not done:
        action, _ = agent.select_action(state, deterministic=True)
        state, _, done, _, info = env.step(np.array([action]))
        
    ret = (info['capital'] - 10000) / 10000 * 100
    print(f"  Final Capital: ${info['capital']:,.0f}")
    print(f"  Return: {ret:+.2f}%")
    print(f"  Trades: {info['total_trades']}")
    
    # Save results
    res = {
        'timestamp': datetime.now().isoformat(),
        'model': 'RL2_Sentiment',
        'return': ret,
        'trades': info['total_trades'],
        'capital': info['capital']
    }
    with open(RESULTS_DIR / 'sentiment_result.json', 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    agent = train(n_episodes=300)
    evaluate(agent)
