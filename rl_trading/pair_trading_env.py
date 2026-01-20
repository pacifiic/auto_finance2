"""
RL2 Pair Trading Environment (Sentiment Augmented)
심리/거시 지표가 통합된 RL2 환경

추가된 상태 변수:
1. Fear & Greed Index (0~100 -> 0~1 정규화)
2. US 10Y Interest Rate (정규화)
3. Yield Curve Spread (10Y-2Y, 경기 선행 지표)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces


class PairTradingEnvRL2(gym.Env):
    """
    RL2 페어 트레이딩 환경 (심리/거시 지표 포함)
    
    State (8차원):
        0. Position: 현재 포지션 [-1, 1]
        1. Spread (Z-score): 정규화된 스프레드
        2. Spread MA: 스프레드 이동평균
        3. Zone: 현재 존 [-1, 1]
        4. Unrealized PnL: 미실현 손익
        5. Fear & Greed Index: 시장 심리 [0, 1]
        6. US Interest Rate: 미국 금리 (10Y)
        7. Yield Spread: 장단기 금리차 (10Y-2Y)
        
    Actions:
        - Continuous [-1, 1]: 목표 포지션 비율
    """
    
    metadata = {'render_modes': ['human']}
    
    ZONE_LONG = -2
    ZONE_NEUTRAL_LONG = -1
    ZONE_CLOSE = 0
    ZONE_NEUTRAL_SHORT = 1
    ZONE_SHORT = 2
    
    def __init__(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        # sentiment_df: Optional[pd.DataFrame] = None,  # Fear & Greed (Commented out)
        # macro_df: Optional[pd.DataFrame] = None,      # Interest Rates (Commented out)
        window_size: int = 20,
        open_threshold: float = 1.0, 
        close_threshold: float = 0.3,
        transaction_cost: float = 0.0002,
        initial_capital: float = 10000.0,
        action_reward_weight: float = 0.5,
    ):
        super().__init__()
        
        self.window_size = window_size
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        self.action_reward_weight = action_reward_weight
        
        # 데이터 정렬 및 병합
        self.df = self._merge_data(df1, df2, None, None) # sentiment_df, macro_df -> None
        
        # 기본 데이터 계산
        self.prices1 = self.df['close1'].values
        self.prices2 = self.df['close2'].values
        self.spread = np.log(self.prices1) - np.log(self.prices2)
        
        # Z-score 계산
        spread_s = pd.Series(self.spread)
        self.spread_mean = spread_s.rolling(window_size).mean().values
        self.spread_std = spread_s.rolling(window_size).std().values
        self.zscore = (self.spread - self.spread_mean) / (self.spread_std + 1e-8)
        
        # 추가 특성값 정규화 (Min-Max) - Commented out
        self.fng = np.zeros(len(self.df))
        self.rate_10y = np.zeros(len(self.df))
        self.yield_spread = np.zeros(len(self.df))
        
        # if 'fng' in self.df.columns:
        #     self.fng = self.df['fng'].values / 100.0
        # else:
        #     self.fng = np.zeros(len(self.df))
            
        # if 'rate_10y' in self.df.columns:
        #     r = self.df['rate_10y'].values
        #     self.rate_10y = (r - 0.5) / 5.0
        # else:
        #     self.rate_10y = np.zeros(len(self.df))
            
        # if 'yield_spread' in self.df.columns:
        #     s = self.df['yield_spread'].values
        #     self.yield_spread = s
        # else:
        #     self.yield_spread = np.zeros(len(self.df))
        
        # 인덱스 설정
        self.start_idx = window_size
        self.max_steps = len(self.df) - self.start_idx - 1
        
        # Space 정의 (Reverted to 5 dim)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32) # Reverted
        
        self.reset()
    
    def _merge_data(self, df1, df2, sentiment, macro):
        """OHLCV, Sentiment, Macro 데이터를 시간 기준으로 병합 (ffill)"""
        # 1. 자산 데이터 병합
        common_idx = df1.index.intersection(df2.index)
        merged = pd.DataFrame(index=common_idx)
        merged['close1'] = df1.loc[common_idx, 'close']
        merged['close2'] = df2.loc[common_idx, 'close']
        
        # Sentiment & Macro Merging - Commented out
        """
        # 2. Sentiment 병합
        if sentiment is not None:
            sentiment = sentiment.reindex(merged.index, method='ffill')
            merged['fng'] = sentiment['value']
        
        # 3. Macro 병합
        if macro is not None:
            if 'date' in macro.columns:
                macro = macro.set_index('date')
            if not isinstance(macro.index, pd.DatetimeIndex):
                macro.index = pd.to_datetime(macro.index)
            macro = macro.reindex(merged.index, method='ffill')
            merged['rate_10y'] = macro['value_10y']
            merged['yield_spread'] = macro['yield_spread'] if 'yield_spread' in macro.columns else macro['spread']
        """
            
        merged = merged.fillna(0)
        return merged
    
    def _get_zone(self, zscore):
        if zscore > self.open_threshold: return self.ZONE_SHORT
        if zscore > self.close_threshold: return self.ZONE_NEUTRAL_SHORT
        if zscore > -self.close_threshold: return self.ZONE_CLOSE
        if zscore > -self.open_threshold: return self.ZONE_NEUTRAL_LONG
        return self.ZONE_LONG

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.start_idx
        self.position = 0.0
        self.entry_price_ratio = 1.0
        self.capital = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.total_trades = 0
        self.winning_trades = 0
        return self._get_observation(), {}

    def _get_observation(self):
        idx = self.current_step
        z = self.zscore[idx]
        zone = self._get_zone(z)
        
        # PnL
        if abs(self.position) > 0.01:
            curr_ratio = self.prices1[idx] / self.prices2[idx]
            pnl = self.position * (curr_ratio / self.entry_price_ratio - 1)
        else:
            pnl = 0.0
            
        obs = np.array([
            self.position,
            z,
            np.mean(self.zscore[max(0, idx-5):idx+1]),
            zone / 2.0,
            pnl,
            # self.fng[idx],           # 심리 (Commented out)
            # self.rate_10y[idx],      # 금리 (Commented out)
            # self.yield_spread[idx],  # 스프레드 (Commented out)
        ], dtype=np.float32)
        
        # NaN/Inf 안전 처리
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs

    def step(self, action):
        target_pos = float(np.clip(action[0], -1, 1))
        idx = self.current_step
        
        current_ratio = self.prices1[idx] / self.prices2[idx]
        
        # 1. PnL Reward
        reward = 0.0
        if abs(self.position) > 0.01:
            price_change = current_ratio / self.entry_price_ratio - 1
            realized = self.position * price_change * abs(self.position - target_pos)
            reward += realized * 100
            
        # 2. Zone Alignment Reward
        zone = self._get_zone(self.zscore[idx])
        desired = 0.0
        if zone == self.ZONE_SHORT: desired = -1.0
        elif zone == self.ZONE_LONG: desired = 1.0
        
        # *심리 기반 보정 (Commented out)*
        # if self.fng[idx] < 0.2 and zone == self.ZONE_LONG:
        #     reward += 0.1
            
        align = 1.0 - abs(target_pos - desired)/2.0
        reward += align * self.action_reward_weight
        
        # 3. Transaction Cost
        change = abs(target_pos - self.position)
        if change > 0.01:
            cost = change * self.transaction_cost * self.portfolio_value
            self.capital -= cost
            reward -= change * 0.5
            
            if abs(self.position) > 0.01 and abs(target_pos) < 0.01:
                self.total_trades += 1
                
        # Update
        if abs(target_pos) > 0.01 and abs(self.position) < 0.01:
            self.entry_price_ratio = current_ratio
            
        self.position = target_pos
        
        if abs(self.position) > 0.01:
            price_change = current_ratio / self.entry_price_ratio - 1
            self.portfolio_value = self.capital * (1 + self.position * price_change)
        else:
            self.portfolio_value = self.capital
            
        self.current_step += 1
        terminated = self.current_step >= self.start_idx + self.max_steps
        
        obs = self._get_observation()
        info = {
            'capital': self.capital,
            'total_trades': self.total_trades,
            'win_rate': 0.0
        }
        
        return obs, reward, terminated, False, info
