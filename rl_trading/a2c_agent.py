"""
A2C Agent for Pair Trading (Paper-Based)
A2C (Advantage Actor-Critic) 에이전트

논문에서 A2C가 최고 성과를 보임 (31.53% 수익)
연속 액션 공간을 위한 Actor-Critic 구조
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, List


class ActorCritic(nn.Module):
    """
    Actor-Critic 네트워크
    
    Actor: 연속 액션 (평균, 분산) 출력
    Critic: 상태 가치 V(s) 출력
    """
    
    def __init__(self, state_size: int, hidden_size: int = 128):
        super(ActorCritic, self).__init__()
        
        # 공유 레이어
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Actor 헤드 (평균과 로그 분산)
        self.actor_mean = nn.Linear(hidden_size, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))
        
        # Critic 헤드
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        shared_out = self.shared(x)
        
        # Actor: 평균 출력 (tanh로 [-1, 1] 범위)
        mean = torch.tanh(self.actor_mean(shared_out))
        std = torch.exp(self.actor_log_std).expand_as(mean)
        
        # Critic: 상태 가치
        value = self.critic(shared_out)
        
        return mean, std, value
    
    def get_action(self, state, deterministic=False):
        """액션 샘플링"""
        mean, std, value = self.forward(state)
        
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, -1, 1)
        
        return action, value
    
    def evaluate(self, states, actions):
        """정책 평가 (학습용)"""
        mean, std, values = self.forward(states)
        
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy


class A2CAgent:
    """
    A2C (Advantage Actor-Critic) 에이전트
    
    - 연속 액션 공간 지원
    - GAE (Generalized Advantage Estimation)
    - Entropy 보너스로 탐색 장려
    """
    
    def __init__(
        self,
        state_size: int,
        hidden_size: int = 128,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 5,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = ActorCritic(state_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # 경험 버퍼
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[float, float]:
        """액션 선택"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, value = self.network.get_action(state_tensor, deterministic)
        
        return action.cpu().numpy()[0], value.cpu().item()
    
    def store(self, state, action, reward, value, done):
        """경험 저장"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float) -> Tuple[List, List]:
        """GAE (Generalized Advantage Estimation) 계산"""
        advantages = []
        returns = []
        
        gae = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        return advantages, returns
    
    def learn(self, next_state: np.ndarray) -> float:
        """학습"""
        if len(self.states) < self.n_steps:
            return 0.0
        
        # 다음 상태 가치
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.network.get_action(next_state_tensor, deterministic=True)
        
        # GAE 계산
        advantages, returns = self.compute_gae(next_value.item())
        
        # 텐서 변환
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
        
        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 정책 평가
        log_probs, values, entropy = self.network.evaluate(states, actions)
        
        # 손실 계산
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropy.mean()
        
        total_loss = (
            actor_loss 
            + self.value_coef * critic_loss 
            + self.entropy_coef * entropy_loss
        )
        
        # 역전파
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # 버퍼 클리어
        self.clear_buffer()
        
        return total_loss.item()
    
    def clear_buffer(self):
        """버퍼 클리어"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def save(self, path: str):
        """모델 저장"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


if __name__ == '__main__':
    # 테스트
    agent = A2CAgent(state_size=5)
    
    # 더미 상태
    state = np.random.randn(5).astype(np.float32)
    action, value = agent.select_action(state)
    
    print(f"A2C Agent Test:")
    print(f"  State: {state}")
    print(f"  Action: {action}")
    print(f"  Value: {value:.3f}")
