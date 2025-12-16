"""
Reinforcement Learning Model - PPO/DQN agents for trading
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import gym
from gym import spaces

from .base_model import BaseModel
from config import config


logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    Custom Gym environment for trading.
    The agent learns to make buy/sell/hold decisions.
    """

    def __init__(
        self,
        df,
        feature_columns: List[str],
        initial_balance: float = 10000,
        commission: float = 0.001,
        max_position: float = 1.0,
        lookback: int = 50
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_position = max_position
        self.lookback = lookback

        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        # Observation space: features + position + balance
        n_features = len(feature_columns) * lookback + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.position = 0.0  # -1 to 1 (short to long)
        self.position_price = 0.0
        self.total_reward = 0
        self.trades = []

        return self._get_observation()

    def _get_observation(self):
        """Get current state observation"""
        # Get feature window
        start = self.current_step - self.lookback
        end = self.current_step

        features = self.df[self.feature_columns].iloc[start:end].values.flatten()

        # Add position and normalized balance
        state = np.concatenate([
            features,
            [self.position, self.balance / self.initial_balance]
        ])

        return state.astype(np.float32)

    def _get_price(self):
        """Get current price"""
        return self.df['close'].iloc[self.current_step]

    def step(self, action):
        """Execute action and return new state, reward, done"""
        current_price = self._get_price()
        prev_portfolio_value = self._get_portfolio_value(current_price)

        # Execute action
        if action == 1:  # Buy
            if self.position < self.max_position:
                # Calculate position size
                buy_amount = (self.max_position - self.position)
                cost = buy_amount * current_price * (1 + self.commission)

                if cost <= self.balance:
                    self.balance -= cost
                    self.position += buy_amount
                    self.position_price = current_price
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'buy',
                        'price': current_price,
                        'position': self.position
                    })

        elif action == 2:  # Sell
            if self.position > -self.max_position:
                # Calculate position size
                sell_amount = (self.position + self.max_position)

                if self.position > 0:
                    # Close long position
                    revenue = sell_amount * current_price * (1 - self.commission)
                    self.balance += revenue

                self.position -= sell_amount
                self.position_price = current_price
                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'price': current_price,
                    'position': self.position
                })

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # Calculate reward
        new_price = self._get_price() if not done else current_price
        new_portfolio_value = self._get_portfolio_value(new_price)

        # Reward is the change in portfolio value (normalized)
        reward = (new_portfolio_value - prev_portfolio_value) / self.initial_balance

        # Add penalty for excessive trading
        if action != 0:
            reward -= self.commission * 0.1

        self.total_reward += reward

        info = {
            'portfolio_value': new_portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'total_reward': self.total_reward
        }

        return self._get_observation(), reward, done, info

    def _get_portfolio_value(self, price):
        """Calculate total portfolio value"""
        position_value = self.position * price
        return self.balance + position_value

    def render(self, mode='human'):
        """Render environment state"""
        price = self._get_price()
        portfolio_value = self._get_portfolio_value(price)
        print(f"Step: {self.current_step}, Price: {price:.2f}, "
              f"Position: {self.position:.2f}, Balance: {self.balance:.2f}, "
              f"Portfolio: {portfolio_value:.2f}")


class RLModel(BaseModel):
    """Reinforcement Learning model using Stable Baselines3"""

    def __init__(
        self,
        name: str = "rl_agent",
        algorithm: str = "PPO"
    ):
        super().__init__(name)
        self.algorithm = algorithm
        self.env = None

    def build(
        self,
        input_shape: Tuple = None,
        algorithm: str = None,
        policy: str = "MlpPolicy",
        **kwargs
    ) -> None:
        """
        Build RL agent

        Args:
            input_shape: Not used, environment defines observation space
            algorithm: RL algorithm (PPO, A2C, DQN)
            policy: Policy network type
        """
        self.algorithm = algorithm or self.algorithm or config.model.rl_algorithm
        self.policy = policy
        self.model_kwargs = kwargs

        logger.info(f"Configured {self.algorithm} agent with {policy} policy")

    def create_environment(
        self,
        df,
        feature_columns: List[str],
        **env_kwargs
    ) -> TradingEnvironment:
        """Create trading environment"""
        self.env = TradingEnvironment(df, feature_columns, **env_kwargs)
        return self.env

    def train(
        self,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        df: Any = None,
        feature_columns: List[str] = None,
        total_timesteps: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the RL agent

        Args:
            df: DataFrame with OHLCV and features
            feature_columns: Feature columns to use
            total_timesteps: Total training steps
        """
        try:
            from stable_baselines3 import PPO, A2C, DQN
            from stable_baselines3.common.vec_env import DummyVecEnv
            from stable_baselines3.common.callbacks import EvalCallback

            if df is None or feature_columns is None:
                raise ValueError("df and feature_columns required for RL training")

            # Create environment
            if self.env is None:
                self.create_environment(df, feature_columns)

            # Wrap environment
            env = DummyVecEnv([lambda: self.env])

            # Select algorithm
            algorithms = {
                'PPO': PPO,
                'A2C': A2C,
                'DQN': DQN
            }

            if self.algorithm not in algorithms:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            AlgoClass = algorithms[self.algorithm]

            # Training parameters
            total_timesteps = total_timesteps or config.model.rl_episodes * len(df)

            # Create model
            self.model = AlgoClass(
                self.policy,
                env,
                verbose=1,
                learning_rate=config.model.learning_rate,
                gamma=config.model.rl_gamma,
                **self.model_kwargs
            )

            # Train
            logger.info(f"Training {self.algorithm} for {total_timesteps} timesteps...")
            self.model.learn(total_timesteps=total_timesteps)

            self.is_trained = True
            self.last_trained = datetime.now()

            # Evaluate
            eval_results = self._evaluate_agent(df, feature_columns)

            return {
                'algorithm': self.algorithm,
                'total_timesteps': total_timesteps,
                'evaluation': eval_results
            }

        except ImportError:
            logger.error("stable-baselines3 not installed. Run: pip install stable-baselines3")
            raise

    def _evaluate_agent(
        self,
        df,
        feature_columns: List[str],
        n_episodes: int = 5
    ) -> Dict[str, float]:
        """Evaluate trained agent"""
        rewards = []
        final_values = []

        for _ in range(n_episodes):
            env = TradingEnvironment(df, feature_columns)
            obs = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)
            final_values.append(info['portfolio_value'])

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_final_value': np.mean(final_values),
            'mean_return': (np.mean(final_values) - 10000) / 10000
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict actions for given states

        Args:
            X: State observations

        Returns:
            Predicted actions
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        actions = []
        for obs in X:
            action, _ = self.model.predict(obs, deterministic=True)
            actions.append(action)

        return np.array(actions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get action probabilities (for policy gradient methods)"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # For PPO/A2C, we can get action distribution
        if hasattr(self.model, 'policy'):
            import torch
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(X)
                distribution = self.model.policy.get_distribution(obs_tensor)
                probs = distribution.distribution.probs.numpy()
                return probs

        # Fallback: return one-hot based on deterministic action
        actions = self.predict(X)
        n_actions = self.env.action_space.n if self.env else 3
        proba = np.zeros((len(actions), n_actions))
        for i, a in enumerate(actions):
            proba[i, a] = 1.0
        return proba

    def get_signal(self, state: np.ndarray, threshold: float = 0.6) -> int:
        """
        Get trading signal from RL agent

        Args:
            state: Current state observation
            threshold: Not used for RL (deterministic policy)

        Returns:
            1 (buy), -1 (sell), or 0 (hold)
        """
        if not self.is_trained:
            return 0

        action, _ = self.model.predict(state, deterministic=True)

        # Map action to signal
        if action == 1:  # Buy
            return 1
        elif action == 2:  # Sell
            return -1
        return 0  # Hold

    def save(self, path: str = None) -> str:
        """Save RL model"""
        from pathlib import Path
        import json

        if path is None:
            path = Path(config.models_dir) / 'saved' / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(path / 'model')

        # Save metadata
        metadata = {
            'name': self.name,
            'algorithm': self.algorithm,
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'created_at': self.created_at.isoformat(),
            'last_trained': self.last_trained.isoformat() if self.last_trained else None
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"RL model saved to {path}")
        return str(path)

    def load(self, path: str) -> None:
        """Load RL model"""
        from pathlib import Path
        import json
        from stable_baselines3 import PPO, A2C, DQN

        path = Path(path)

        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.name = metadata['name']
        self.algorithm = metadata['algorithm']
        self.is_trained = metadata['is_trained']
        self.metrics = metadata['metrics']

        # Load model
        algorithms = {'PPO': PPO, 'A2C': A2C, 'DQN': DQN}
        AlgoClass = algorithms[self.algorithm]
        self.model = AlgoClass.load(path / 'model')

        logger.info(f"RL model loaded from {path}")
