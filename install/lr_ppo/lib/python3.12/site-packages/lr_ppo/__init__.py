"""
lr_ppo - PPO Reinforcement Learning Package for ROSI the TurtleBot3
====================================================================

This package provides:
- Gymnasium environment for ROSI maze navigation
- PPO (Proximal Policy Optimization) implementation
- Training and evaluation utilities

ROSI learns to navigate mazes using:
- LiDAR sensor data (360° laser scan)
- Odometry (position and orientation)
- Goal-directed reward shaping
"""

from lr_ppo.environment import RosiMazeEnv
from lr_ppo.ppo_agent import PPOAgent
from lr_ppo.networks import ActorCritic
from lr_ppo.utils import (
    normalize_angle,
    euclidean_distance,
    angle_to_goal,
)

__version__ = "1.0.0"
__author__ = "ROSI Student"

__all__ = [
    "RosiMazeEnv",
    "PPOAgent",
    "ActorCritic",
    "normalize_angle",
    "euclidean_distance",
    "angle_to_goal",
]
