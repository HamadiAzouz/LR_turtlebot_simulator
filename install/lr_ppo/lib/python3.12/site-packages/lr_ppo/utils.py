"""
Utility Functions for ROSI PPO Training
========================================

Common mathematical and helper functions used throughout
the lr_ppo package.
"""

import math
import numpy as np
from typing import Union, Tuple


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-pi, pi] range.
    
    Args:
        angle: Angle in radians
    
    Returns:
        Normalized angle in [-pi, pi]
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def euclidean_distance(
    p1: Union[np.ndarray, Tuple[float, float]],
    p2: Union[np.ndarray, Tuple[float, float]]
) -> float:
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
    
    Returns:
        Euclidean distance
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return float(np.linalg.norm(p1 - p2))


def angle_to_goal(
    position: Union[np.ndarray, Tuple[float, float]],
    goal: Union[np.ndarray, Tuple[float, float]],
    current_yaw: float
) -> float:
    """
    Calculate the relative angle from robot's heading to goal.
    
    Args:
        position: Robot's current position (x, y)
        goal: Goal position (x, y)
        current_yaw: Robot's current yaw angle (radians)
    
    Returns:
        Relative angle to goal in [-pi, pi]
    """
    dx = goal[0] - position[0]
    dy = goal[1] - position[1]
    goal_angle = math.atan2(dy, dx)
    return normalize_angle(goal_angle - current_yaw)


def moving_average(values: np.ndarray, window: int = 100) -> np.ndarray:
    """
    Calculate moving average of values.
    
    Args:
        values: Array of values
        window: Window size for averaging
    
    Returns:
        Array of moving averages
    """
    if len(values) < window:
        window = len(values)
    return np.convolve(values, np.ones(window) / window, mode='valid')


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate explained variance between predictions and targets.
    
    Used to monitor value function quality during training.
    
    Args:
        y_pred: Predicted values
        y_true: True values
    
    Returns:
        Explained variance ratio in [-inf, 1], where 1 is perfect prediction
    """
    var_y = np.var(y_true)
    if var_y == 0:
        return 0.0
    return float(1 - np.var(y_true - y_pred) / var_y)


def discount_cumsum(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """
    Calculate discounted cumulative sum of rewards (returns).
    
    Args:
        rewards: Array of rewards
        gamma: Discount factor
    
    Returns:
        Array of discounted returns
    """
    n = len(rewards)
    discounted = np.zeros(n)
    running_sum = 0.0
    
    for i in reversed(range(n)):
        running_sum = rewards[i] + gamma * running_sum
        discounted[i] = running_sum
    
    return discounted


def generalized_advantage_estimation(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    GAE provides a good balance between bias and variance in
    advantage estimation for policy gradient methods.
    
    Args:
        rewards: Array of rewards
        values: Array of state values
        next_values: Array of next state values
        dones: Array of done flags (1 if episode ended)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        Tuple of (advantages, returns)
    """
    n = len(rewards)
    advantages = np.zeros(n)
    last_gae = 0.0
    
    for t in reversed(range(n)):
        if dones[t]:
            next_value = 0.0
        else:
            next_value = next_values[t]
        
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    
    returns = advantages + values
    
    return advantages, returns


class RunningMeanStd:
    """
    Running mean and standard deviation calculator.
    
    Used for observation normalization during training.
    """
    
    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-8):
        """
        Initialize running statistics.
        
        Args:
            shape: Shape of the values to track
            epsilon: Small constant for numerical stability
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon
    
    def update(self, x: np.ndarray):
        """
        Update running statistics with new batch of values.
        
        Args:
            x: Batch of values (shape: [batch_size, ...])
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int
    ):
        """Update statistics from batch moments."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize values using running statistics.
        
        Args:
            x: Values to normalize
        
        Returns:
            Normalized values
        """
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """
        Denormalize values using running statistics.
        
        Args:
            x: Normalized values
        
        Returns:
            Original scale values
        """
        return x * np.sqrt(self.var + self.epsilon) + self.mean


class ReplayBuffer:
    """
    Simple replay buffer for PPO rollouts.
    
    Stores transitions from environment interactions for
    later use in policy updates.
    """
    
    def __init__(self, size: int = 2048):
        """
        Initialize replay buffer.
        
        Args:
            size: Maximum buffer size
        """
        self.size = size
        self.clear()
    
    def clear(self):
        """Clear all stored transitions."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ):
        """
        Add a transition to the buffer.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
            log_prob: Log probability of action
            value: Value estimate
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get_all(self) -> dict:
        """
        Get all stored transitions as numpy arrays.
        
        Returns:
            Dictionary of transition arrays
        """
        return {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'next_observations': np.array(self.next_observations),
            'dones': np.array(self.dones, dtype=np.float32),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),
        }
    
    def __len__(self) -> int:
        """Return number of stored transitions."""
        return len(self.observations)
