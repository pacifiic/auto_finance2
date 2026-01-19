"""
Hyperparameter Manager
하이퍼파라미터 관리 모듈

모든 패턴과 지표의 파라미터를 중앙에서 관리하고,
프리셋 및 최적화 기능을 제공합니다.
"""

from typing import Dict, Any, Optional, List, Tuple
import yaml
import json
import copy
from pathlib import Path


class HyperParameterManager:
    """
    하이퍼파라미터 관리자
    
    모든 기술적 분석 파라미터를 중앙에서 관리합니다.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: YAML 설정 파일 경로
        """
        self.config_path = config_path
        self.params = self._load_or_create_default()
        self.presets = self._define_presets()
        self.history = []  # 파라미터 변경 이력
    
    def _load_or_create_default(self) -> Dict[str, Any]:
        """설정 로드 또는 기본값 생성"""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return self._default_params()
    
    def _default_params(self) -> Dict[str, Any]:
        """기본 하이퍼파라미터"""
        return {
            # ============================================
            # 추세 지표 파라미터
            # ============================================
            'trend': {
                'sma': {
                    'short_period': 20,
                    'medium_period': 50,
                    'long_period': 200,
                    'weight': 1.0,
                    'enabled': True
                },
                'ema': {
                    'short_period': 12,
                    'medium_period': 26,
                    'long_period': 50,
                    'weight': 1.2,
                    'enabled': True
                },
                'macd': {
                    'fast_period': 12,
                    'slow_period': 26,
                    'signal_period': 9,
                    'weight': 1.5,
                    'enabled': True
                },
                'adx': {
                    'period': 14,
                    'strong_trend_threshold': 25,
                    'weak_trend_threshold': 20,
                    'weight': 0.8,
                    'enabled': True
                }
            },
            
            # ============================================
            # 모멘텀 지표 파라미터
            # ============================================
            'momentum': {
                'rsi': {
                    'period': 14,
                    'overbought': 70,
                    'oversold': 30,
                    'extreme_overbought': 80,
                    'extreme_oversold': 20,
                    'weight': 1.3,
                    'enabled': True
                },
                'stochastic': {
                    'k_period': 14,
                    'd_period': 3,
                    'smooth_k': 3,
                    'overbought': 80,
                    'oversold': 20,
                    'weight': 1.0,
                    'enabled': True
                },
                'williams_r': {
                    'period': 14,
                    'overbought': -20,
                    'oversold': -80,
                    'weight': 0.7,
                    'enabled': True
                },
                'cci': {
                    'period': 20,
                    'overbought': 100,
                    'oversold': -100,
                    'weight': 0.8,
                    'enabled': True
                }
            },
            
            # ============================================
            # 변동성 지표 파라미터
            # ============================================
            'volatility': {
                'bollinger': {
                    'period': 20,
                    'std_dev': 2.0,
                    'squeeze_threshold': 0.05,
                    'weight': 1.2,
                    'enabled': True
                },
                'atr': {
                    'period': 14,
                    'multiplier': 2.0,
                    'weight': 0.6,
                    'enabled': True
                },
                'keltner': {
                    'ema_period': 20,
                    'atr_period': 10,
                    'multiplier': 2.0,
                    'weight': 0.7,
                    'enabled': True
                }
            },
            
            # ============================================
            # 거래량 지표 파라미터
            # ============================================
            'volume': {
                'obv': {
                    'ma_period': 20,
                    'weight': 0.9,
                    'enabled': True
                },
                'volume_ma': {
                    'short_period': 10,
                    'long_period': 20,
                    'spike_threshold': 2.0,
                    'weight': 0.8,
                    'enabled': True
                },
                'mfi': {
                    'period': 14,
                    'overbought': 80,
                    'oversold': 20,
                    'weight': 0.9,
                    'enabled': True
                },
                'vwap': {
                    'period': 20,
                    'weight': 0.7,
                    'enabled': True
                }
            },
            
            # ============================================
            # 패턴 파라미터
            # ============================================
            'patterns': {
                'candlestick': {
                    'reversal_weight': 1.4,
                    'continuation_weight': 0.9,
                    'enabled': True
                },
                'chart': {
                    'double_top_lookback': 50,
                    'double_bottom_lookback': 50,
                    'head_shoulders_lookback': 60,
                    'triangle_lookback': 40,
                    'flag_lookback': 30,
                    'tolerance': 0.02,
                    'weight': 1.3,
                    'enabled': True
                }
            },
            
            # ============================================
            # 지지/저항 및 피보나치
            # ============================================
            'levels': {
                'support_resistance': {
                    'lookback_period': 100,
                    'min_touches': 2,
                    'tolerance': 0.01,
                    'weight': 1.1,
                    'enabled': True
                },
                'fibonacci': {
                    'levels': [0.236, 0.382, 0.5, 0.618, 0.786],
                    'tolerance': 0.005,
                    'weight': 0.9,
                    'enabled': True
                }
            },
            
            # ============================================
            # 신호 조합 파라미터
            # ============================================
            'signal_combination': {
                'method': 'weighted_average',
                'category_weights': {
                    'trend': 1.2,
                    'momentum': 1.0,
                    'volatility': 0.8,
                    'volume': 0.9,
                    'candlestick': 1.1,
                    'chart_pattern': 1.0,
                    'support_resistance': 1.0,
                    'fibonacci': 0.7
                },
                'thresholds': {
                    'strong_buy': 0.7,
                    'buy': 0.3,
                    'neutral_high': 0.1,
                    'neutral_low': -0.1,
                    'sell': -0.3,
                    'strong_sell': -0.7
                },
                'min_confirming_indicators': 3,
                'trend_alignment_bonus': 0.2,
                'volume_confirmation_bonus': 0.15
            },
            
            # ============================================
            # 리스크 관리 파라미터
            # ============================================
            'risk_management': {
                'max_position_size': 0.1,
                'stop_loss_percent': 0.02,
                'take_profit_percent': 0.04,
                'risk_reward_ratio': 2.0,
                'atr_stop_multiplier': 2.0,
                'trailing_stop': True,
                'trailing_percent': 0.01
            }
        }
    
    def _define_presets(self) -> Dict[str, Dict[str, Any]]:
        """미리 정의된 프리셋들"""
        return {
            # 공격적 트레이딩 (단기)
            'aggressive': {
                'description': '짧은 기간, 빈번한 거래, 높은 위험',
                'params': {
                    'trend': {
                        'sma': {'short_period': 10, 'medium_period': 20, 'long_period': 50},
                        'ema': {'short_period': 5, 'medium_period': 13, 'long_period': 26}
                    },
                    'momentum': {
                        'rsi': {'period': 7, 'overbought': 75, 'oversold': 25},
                        'stochastic': {'k_period': 7, 'd_period': 3}
                    },
                    'signal_combination': {
                        'thresholds': {
                            'buy': 0.2,
                            'sell': -0.2
                        }
                    },
                    'risk_management': {
                        'max_position_size': 0.2,
                        'stop_loss_percent': 0.03,
                        'take_profit_percent': 0.06
                    }
                }
            },
            
            # 보수적 트레이딩 (장기)
            'conservative': {
                'description': '긴 기간, 확실한 신호만, 낮은 위험',
                'params': {
                    'trend': {
                        'sma': {'short_period': 50, 'medium_period': 100, 'long_period': 200},
                        'ema': {'short_period': 20, 'medium_period': 50, 'long_period': 100}
                    },
                    'momentum': {
                        'rsi': {'period': 21, 'overbought': 80, 'oversold': 20},
                        'stochastic': {'k_period': 21, 'd_period': 5}
                    },
                    'signal_combination': {
                        'thresholds': {
                            'buy': 0.5,
                            'sell': -0.5
                        },
                        'min_confirming_indicators': 5
                    },
                    'risk_management': {
                        'max_position_size': 0.05,
                        'stop_loss_percent': 0.015,
                        'take_profit_percent': 0.03
                    }
                }
            },
            
            # 스윙 트레이딩
            'swing': {
                'description': '중기 트레이딩, 며칠~몇 주 보유',
                'params': {
                    'trend': {
                        'sma': {'short_period': 20, 'medium_period': 50, 'long_period': 100},
                        'ema': {'short_period': 12, 'medium_period': 26, 'long_period': 50}
                    },
                    'momentum': {
                        'rsi': {'period': 14},
                        'stochastic': {'k_period': 14}
                    },
                    'signal_combination': {
                        'category_weights': {
                            'trend': 1.5,
                            'momentum': 1.2,
                            'chart_pattern': 1.3
                        }
                    }
                }
            },
            
            # 스캘핑
            'scalping': {
                'description': '초단기 거래, 작은 이익 빈번히',
                'params': {
                    'trend': {
                        'sma': {'short_period': 5, 'medium_period': 10, 'long_period': 20},
                        'ema': {'short_period': 3, 'medium_period': 8, 'long_period': 13}
                    },
                    'momentum': {
                        'rsi': {'period': 5, 'overbought': 70, 'oversold': 30},
                        'stochastic': {'k_period': 5, 'd_period': 2}
                    },
                    'volatility': {
                        'bollinger': {'period': 10, 'std_dev': 1.5}
                    },
                    'signal_combination': {
                        'thresholds': {
                            'buy': 0.15,
                            'sell': -0.15
                        }
                    },
                    'risk_management': {
                        'stop_loss_percent': 0.005,
                        'take_profit_percent': 0.01
                    }
                }
            },
            
            # 추세 추종
            'trend_following': {
                'description': '추세 지표 중심, 추세 방향 매매',
                'params': {
                    'signal_combination': {
                        'category_weights': {
                            'trend': 2.0,
                            'momentum': 1.5,
                            'volatility': 0.5,
                            'volume': 1.0,
                            'candlestick': 0.7,
                            'chart_pattern': 0.8
                        }
                    }
                }
            },
            
            # 평균 회귀
            'mean_reversion': {
                'description': '과매수/과매도 기반, 평균으로 회귀 예상',
                'params': {
                    'signal_combination': {
                        'category_weights': {
                            'trend': 0.5,
                            'momentum': 2.0,
                            'volatility': 1.5,
                            'volume': 1.0,
                            'support_resistance': 1.5
                        }
                    },
                    'momentum': {
                        'rsi': {'overbought': 75, 'oversold': 25, 'weight': 1.5}
                    }
                }
            }
        }
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        경로로 파라미터 값 가져오기
        
        Args:
            path: 점(.)으로 구분된 경로 (예: 'trend.rsi.period')
            default: 기본값
            
        Returns:
            파라미터 값
        """
        keys = path.split('.')
        value = self.params
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any, record_history: bool = True):
        """
        경로로 파라미터 값 설정
        
        Args:
            path: 점(.)으로 구분된 경로
            value: 설정할 값
            record_history: 변경 이력 기록 여부
        """
        keys = path.split('.')
        params = self.params
        
        # 이전 값 저장
        old_value = self.get(path)
        
        # 마지막 키 전까지 탐색
        for key in keys[:-1]:
            if key not in params:
                params[key] = {}
            params = params[key]
        
        # 값 설정
        params[keys[-1]] = value
        
        # 이력 기록
        if record_history:
            self.history.append({
                'path': path,
                'old_value': old_value,
                'new_value': value
            })
    
    def apply_preset(self, preset_name: str):
        """
        프리셋 적용
        
        Args:
            preset_name: 프리셋 이름
        """
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(self.presets.keys())}")
        
        preset = self.presets[preset_name]
        preset_params = preset['params']
        
        # 깊은 병합
        self._deep_merge(self.params, preset_params)
        
        self.history.append({
            'action': 'apply_preset',
            'preset': preset_name
        })
    
    def _deep_merge(self, base: Dict, update: Dict):
        """깊은 딕셔너리 병합"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get_preset_info(self, preset_name: str) -> Dict[str, Any]:
        """프리셋 정보 반환"""
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        return self.presets[preset_name]
    
    def list_presets(self) -> List[Dict[str, str]]:
        """모든 프리셋 목록 반환"""
        return [
            {'name': name, 'description': preset['description']}
            for name, preset in self.presets.items()
        ]
    
    def get_all_weights(self) -> Dict[str, float]:
        """모든 가중치 반환"""
        weights = {}
        
        for category in ['trend', 'momentum', 'volatility', 'volume']:
            if category in self.params:
                for indicator, config in self.params[category].items():
                    if isinstance(config, dict) and 'weight' in config:
                        weights[f'{category}.{indicator}'] = config['weight']
        
        # 카테고리 가중치
        cat_weights = self.get('signal_combination.category_weights', {})
        for cat, weight in cat_weights.items():
            weights[f'category.{cat}'] = weight
        
        return weights
    
    def set_all_weights(self, weights: Dict[str, float]):
        """모든 가중치 일괄 설정"""
        for path, weight in weights.items():
            if path.startswith('category.'):
                cat = path.replace('category.', '')
                self.set(f'signal_combination.category_weights.{cat}', weight)
            else:
                self.set(f'{path}.weight', weight)
    
    def reset_to_default(self):
        """기본값으로 리셋"""
        self.params = self._default_params()
        self.history.append({'action': 'reset_to_default'})
    
    def save(self, path: Optional[str] = None):
        """설정 저장"""
        save_path = path or self.config_path
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.params, f, default_flow_style=False, allow_unicode=True)
    
    def load(self, path: str):
        """설정 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            self.params = yaml.safe_load(f)
    
    def export_json(self) -> str:
        """JSON으로 내보내기"""
        return json.dumps(self.params, indent=2, ensure_ascii=False)
    
    def import_json(self, json_str: str):
        """JSON에서 가져오기"""
        self.params = json.loads(json_str)
    
    def get_parameter_ranges(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        파라미터 최적화를 위한 범위 정의
        
        Returns:
            {경로: (최소값, 최대값)} 딕셔너리
        """
        return {
            # 추세 지표
            'trend.sma.short_period': (5, 50),
            'trend.sma.medium_period': (20, 100),
            'trend.sma.long_period': (50, 300),
            'trend.ema.short_period': (3, 30),
            'trend.ema.medium_period': (10, 50),
            'trend.macd.fast_period': (8, 20),
            'trend.macd.slow_period': (20, 40),
            'trend.macd.signal_period': (5, 15),
            
            # 모멘텀 지표
            'momentum.rsi.period': (7, 28),
            'momentum.rsi.overbought': (65, 85),
            'momentum.rsi.oversold': (15, 35),
            'momentum.stochastic.k_period': (5, 21),
            
            # 변동성 지표
            'volatility.bollinger.period': (10, 30),
            'volatility.bollinger.std_dev': (1.5, 3.0),
            
            # 가중치
            'signal_combination.category_weights.trend': (0.5, 2.0),
            'signal_combination.category_weights.momentum': (0.5, 2.0),
            'signal_combination.category_weights.volatility': (0.3, 1.5),
            'signal_combination.category_weights.volume': (0.5, 1.5),
            
            # 임계값
            'signal_combination.thresholds.buy': (0.1, 0.5),
            'signal_combination.thresholds.sell': (-0.5, -0.1),
            
            # 리스크
            'risk_management.stop_loss_percent': (0.01, 0.05),
            'risk_management.take_profit_percent': (0.02, 0.1)
        }
    
    def to_config_dict(self) -> Dict[str, Any]:
        """전체 설정을 모듈에서 사용할 수 있는 형식으로 변환"""
        return {
            'trend_indicators': self.params.get('trend', {}),
            'momentum_indicators': self.params.get('momentum', {}),
            'volatility_indicators': self.params.get('volatility', {}),
            'volume_indicators': self.params.get('volume', {}),
            'candlestick_patterns': self.params.get('patterns', {}).get('candlestick', {}),
            'chart_patterns': self.params.get('patterns', {}).get('chart', {}),
            'support_resistance': self.params.get('levels', {}).get('support_resistance', {}),
            'fibonacci': self.params.get('levels', {}).get('fibonacci', {}),
            'signal_combination': self.params.get('signal_combination', {})
        }
    
    def __repr__(self) -> str:
        return f"HyperParameterManager(presets={list(self.presets.keys())}, history_length={len(self.history)})"
