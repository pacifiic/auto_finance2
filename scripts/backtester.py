"""
Backtester
백테스트 엔진

전략의 과거 성능을 평가합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading.strategy import StrategyEngine


@dataclass
class BacktestResult:
    """백테스트 결과"""
    total_return: float  # 총 수익률
    sharpe_ratio: float  # 샤프 비율
    max_drawdown: float  # 최대 낙폭
    win_rate: float  # 승률
    profit_factor: float  # 이익/손실 비율
    total_trades: int  # 총 거래 수
    avg_trade_return: float  # 평균 거래 수익률
    volatility: float  # 수익률 변동성
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'avg_trade_return': self.avg_trade_return,
            'volatility': self.volatility,
        }


class Backtester:
    """
    백테스트 엔진
    
    전략 엔진의 신호를 기반으로 가상 거래를 수행하고
    성능 지표를 계산합니다.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1% 수수료
        slippage: float = 0.0005,  # 0.05% 슬리피지
        tax_rate: float = 0.22,  # 22% 양도소득세 (한국 기준: 20% + 지방세 2%)
        confidence_threshold: float = 0.0,  # 확신도 임계값 (0~1, 높을수록 확실한 신호만)
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.tax_rate = tax_rate  # 수익 발생 시 적용
        self.confidence_threshold = confidence_threshold  # 신호 강도가 이 값 이상일 때만 매매
    
    def run(
        self,
        df: pd.DataFrame,
        engine: StrategyEngine,
        risk_per_trade: float = 0.02,  # 거래당 2% 리스크
    ) -> Tuple[BacktestResult, pd.DataFrame]:
        """
        백테스트 실행
        
        Args:
            df: OHLCV 데이터프레임
            engine: 전략 엔진
            risk_per_trade: 거래당 리스크 비율
            
        Returns:
            (BacktestResult, 상세 거래 기록 DataFrame)
        """
        # 신호 생성
        signals_df = engine.backtest_signal(df)
        
        # 초기화
        capital = self.initial_capital
        position = 0  # 0: 없음, 1: 롱
        entry_price = 0.0
        trades = []
        equity_curve = [capital]
        
        for i in range(1, len(signals_df)):
            current_price = signals_df['close'].iloc[i]
            signal = signals_df['position'].iloc[i]
            signal_strength = abs(signals_df['signal'].iloc[i])  # 신호 강도 (확신도)
            prev_signal = signals_df['position'].iloc[i-1]
            
            # 확신도 필터: 신호 강도가 임계값 이상일 때만 매매
            meets_confidence = signal_strength >= self.confidence_threshold
            
            # 포지션 진입 (신호가 1이고 확신도 충분할 때)
            if signal == 1 and position == 0 and meets_confidence:
                # 슬리피지 적용
                entry_price = current_price * (1 + self.slippage)
                # 포지션 크기 계산 (자본의 일정 비율)
                position_value = capital * 0.95  # 95% 투자
                # 수수료 포함한 총 비용
                total_cost = position_value * (1 + self.commission)
                # 자본에서 차감
                capital -= total_cost
                # 코인 수량 계산
                position = position_value / entry_price
                
            # 포지션 청산 (신호가 -1이고 확신도 충분할 때, 또는 강제 청산)
            elif (signal <= 0 and position > 0 and meets_confidence) or (signal == -1 and position > 0):
                # 슬리피지 적용
                exit_price = current_price * (1 - self.slippage)
                # 수익 계산
                trade_return = (exit_price - entry_price) / entry_price
                trade_pnl = position * (exit_price - entry_price)
                # 청산 수령액 (수수료 차감)
                exit_value = position * exit_price * (1 - self.commission)
                
                # 세금 계산 (수익이 발생한 경우에만)
                if trade_pnl > 0:
                    tax = trade_pnl * self.tax_rate
                    exit_value -= tax
                
                # 자본에 추가
                capital += exit_value
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'pnl': trade_pnl,
                    'timestamp': signals_df.index[i],
                })
                
                position = 0
                entry_price = 0.0
            
            # 현재 자산 가치 계산
            if position > 0:
                current_equity = capital + position * current_price
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
        
        # 마지막 포지션 청산
        if position > 0:
            final_price = signals_df['close'].iloc[-1] * (1 - self.slippage)
            trade_return = (final_price - entry_price) / entry_price
            trade_pnl = position * (final_price - entry_price)
            exit_value = position * final_price * (1 - self.commission)
            
            # 세금 계산 (수익이 발생한 경우에만)
            if trade_pnl > 0:
                tax = trade_pnl * self.tax_rate
                exit_value -= tax
            
            capital += exit_value
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': final_price,
                'return': trade_return,
                'pnl': trade_pnl,
                'timestamp': signals_df.index[-1],
            })
        
        # 성능 지표 계산
        result = self._calculate_metrics(trades, equity_curve)
        
        # 거래 기록 DataFrame
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        return result, trades_df
    
    def _calculate_metrics(
        self, 
        trades: List[Dict], 
        equity_curve: List[float]
    ) -> BacktestResult:
        """성능 지표 계산"""
        
        # 총 수익률
        final_equity = equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # 수익률 시계열
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # 변동성 (연율화 - 4시간봉 기준)
        periods_per_year = 365 * 6  # 4시간봉 기준
        
        # 안전한 변동성 계산
        if len(returns) > 0 and returns.std() > 0:
            volatility = returns.std() * np.sqrt(periods_per_year)
        else:
            volatility = 0.0
        
        # 안전한 Sharpe Ratio 계산 (단순 방식)
        # 평균 수익률을 연율화하고 변동성으로 나눔
        if len(returns) > 0 and volatility > 0:
            mean_return = returns.mean()
            annualized_mean_return = mean_return * periods_per_year
            sharpe_ratio = annualized_mean_return / volatility
            # 합리적인 범위로 클리핑 (-5 ~ 5)
            sharpe_ratio = float(np.clip(sharpe_ratio, -5, 5))
        else:
            sharpe_ratio = 0.0
        
        # 최대 낙폭
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # 거래 통계
        if trades:
            trade_returns = [t['return'] for t in trades]
            winning_trades = [t for t in trades if t['return'] > 0]
            losing_trades = [t for t in trades if t['return'] <= 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            total_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
            total_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
            
            avg_trade_return = np.mean(trade_returns)
            total_trades = len(trades)
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_return = 0.0
            total_trades = 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_return=avg_trade_return,
            volatility=volatility,
        )
    
    def run_quick(
        self,
        df: pd.DataFrame,
        engine: StrategyEngine,
    ) -> float:
        """
        빠른 백테스트 (튜닝용)
        
        전체 지표 대신 Sharpe Ratio만 반환
        """
        result, _ = self.run(df, engine)
        
        # 복합 스코어: Sharpe + 수익률 보정
        score = result.sharpe_ratio
        
        # 수익률이 음수면 패널티
        if result.total_return < 0:
            score -= abs(result.total_return)
        
        # 거래가 너무 적으면 패널티
        if result.total_trades < 5:
            score *= 0.5
        
        return score


if __name__ == '__main__':
    # 테스트
    from data_loader import load_ohlcv
    
    print("Backtester 테스트")
    print("=" * 60)
    
    # 데이터 로드
    df = load_ohlcv('BTC/USDT', '4h', start_date='2024-01-01', end_date='2024-06-30')
    print(f"데이터: {len(df)} 캔들")
    
    # 백테스트 실행
    engine = StrategyEngine()
    backtester = Backtester()
    
    result, trades_df = backtester.run(df, engine)
    
    print(f"\n📊 백테스트 결과:")
    print(f"  총 수익률: {result.total_return:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  최대 낙폭: {result.max_drawdown:.2%}")
    print(f"  승률: {result.win_rate:.1%}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  총 거래 수: {result.total_trades}")
    print(f"  평균 거래 수익률: {result.avg_trade_return:.2%}")
