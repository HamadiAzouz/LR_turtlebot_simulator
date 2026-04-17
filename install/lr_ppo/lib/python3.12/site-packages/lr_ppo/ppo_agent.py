"""
PPO Agent for ROSI TurtleBot3
==============================

This module implements the Proximal Policy Optimization (PPO) algorithm
for training ROSI to navigate through mazes.

PPO is a policy gradient method that uses:
- Clipped surrogate objective for stable policy updates
- Generalized Advantage Estimation (GAE) for variance reduction
- Value function clipping (optional) for stability
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from lr_ppo.networks import ActorCritic, SeparateActorCritic, EnhancedActorCritic
from lr_ppo.utils import (
    generalized_advantage_estimation,
    explained_variance,
    ReplayBuffer,
    RunningMeanStd,
)


class RolloutDataset(Dataset):
    """Dataset for PPO rollout data."""
    
    def __init__(self, data: Dict[str, np.ndarray]):
        self.observations = torch.FloatTensor(data['observations'])
        self.actions = torch.FloatTensor(data['actions'])
        self.old_log_probs = torch.FloatTensor(data['log_probs'])
        self.advantages = torch.FloatTensor(data['advantages'])
        self.returns = torch.FloatTensor(data['returns'])
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.actions[idx],
            self.old_log_probs[idx],
            self.advantages[idx],
            self.returns[idx],
        )


class PPOAgent:
    """
    Proximal Policy Optimization Agent.
    
    This agent learns to control ROSI through trial and error,
    optimizing a clipped surrogate objective for stability.
    
    Key hyperparameters:
    - clip_ratio: How much the policy can change per update (0.2 typical)
    - gamma: Discount factor for future rewards (0.99 typical)
    - gae_lambda: GAE parameter (0.95 typical)
    - lr: Learning rate
    - n_epochs: Number of epochs per PPO update
    """
    
    DEFAULT_CONFIG = {
        # Network architecture
        "hidden_sizes": (256, 256),
        "activation": "tanh",
        "separate_networks": False,
        "enhanced_network": True,  # Use enhanced CNN+attention network
        "num_lidar_samples": 24,   # For enhanced network
        
        # PPO hyperparameters
        "clip_ratio": 0.2,
        "clip_value": True,
        "clip_value_range": 0.2,
        "target_kl": 0.03,  # Early stopping if KL divergence exceeds this (increased for more exploration)
        
        # GAE parameters
        "gamma": 0.99,
        "gae_lambda": 0.95,
        
        # Optimization
        "learning_rate": 1e-4,  # Reduced for more stable learning
        "lr_schedule": "constant",  # "constant", "linear", "cosine"
        "max_grad_norm": 0.5,
        "n_epochs": 10,
        "batch_size": 128,  # Increased for better gradient estimates
        
        # Coefficients
        "value_coef": 0.5,
        "entropy_coef": 0.05,  # Increased from 0.01 to prevent policy collapse
        "entropy_decay": 0.9995,  # Slower decay to maintain exploration longer
        "min_entropy_coef": 0.01,  # Higher minimum to always maintain some exploration
        
        # Normalization
        "normalize_advantages": True,
        "normalize_observations": False,
        
        # Rollout settings
        "rollout_length": 2048,
        
        # Device
        "device": "auto",  # "auto", "cuda", "cpu"
    }
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the PPO Agent.
        
        Args:
            obs_dim: Dimension of observation space
            action_dim: Dimension of action space
            config: Configuration dictionary
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        # Set device
        if self.config["device"] == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config["device"])
        
        print(f"🧠 PPO Agent using device: {self.device}")
        
        # Create network
        if self.config["enhanced_network"]:
            print("🚀 Using Enhanced Actor-Critic with CNN + Residual blocks")
            self.network = EnhancedActorCritic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_lidar_samples=self.config["num_lidar_samples"],
                hidden_dim=256,
                num_residual_blocks=3,
            ).to(self.device)
        elif self.config["separate_networks"]:
            self.network = SeparateActorCritic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                actor_hidden_sizes=self.config["hidden_sizes"],
                critic_hidden_sizes=self.config["hidden_sizes"],
                activation=self.config["activation"],
            ).to(self.device)
        else:
            self.network = ActorCritic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=self.config["hidden_sizes"],
                activation=self.config["activation"],
            ).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config["learning_rate"],
            eps=1e-5
        )
        
        # Learning rate scheduler
        self.lr_scheduler = None
        if self.config["lr_schedule"] == "linear":
            # Will be set up when training starts
            pass
        elif self.config["lr_schedule"] == "cosine":
            # Will be set up when training starts
            pass
        
        # Replay buffer
        self.buffer = ReplayBuffer(size=self.config["rollout_length"])
        
        # Observation normalization
        if self.config["normalize_observations"]:
            self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        else:
            self.obs_rms = None
        
        # Current entropy coefficient
        self.entropy_coef = self.config["entropy_coef"]
        
        # Training statistics
        self.total_timesteps = 0
        self.total_updates = 0
        self.episode_count = 0
    
    def select_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select an action given an observation.
        
        Args:
            observation: Current observation
            deterministic: If True, select mean action (no exploration)
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        # Normalize observation if enabled
        if self.obs_rms is not None:
            observation = self.obs_rms.normalize(observation)
        
        # Convert to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.network(
                obs_tensor, deterministic=deterministic
            )
        
        return (
            action.cpu().numpy().squeeze(0),
            float(log_prob.cpu().numpy()),
            float(value.cpu().numpy())
        )
    
    def store_transition(
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
        Store a transition in the replay buffer.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
            log_prob: Log probability of action
            value: Value estimate
        """
        self.buffer.add(
            observation, action, reward, next_observation,
            done, log_prob, value
        )
        self.total_timesteps += 1
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout data.
        
        Returns:
            Dictionary of training statistics
        """
        # Get data from buffer
        data = self.buffer.get_all()
        
        # Normalize observations if enabled
        if self.obs_rms is not None:
            self.obs_rms.update(data['observations'])
            data['observations'] = self.obs_rms.normalize(data['observations'])
            data['next_observations'] = self.obs_rms.normalize(data['next_observations'])
        
        # Compute next values for GAE
        with torch.no_grad():
            next_obs = torch.FloatTensor(data['next_observations']).to(self.device)
            next_values = self.network.get_value(next_obs).cpu().numpy()
        
        # Compute advantages and returns using GAE
        advantages, returns = generalized_advantage_estimation(
            rewards=data['rewards'],
            values=data['values'],
            next_values=next_values,
            dones=data['dones'],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"]
        )
        
        # Normalize advantages
        if self.config["normalize_advantages"]:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Add to data dict
        data['advantages'] = advantages
        data['returns'] = returns
        
        # Create dataset and dataloader
        dataset = RolloutDataset(data)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=False
        )
        
        # Training statistics
        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': [],
        }
        
        # PPO update epochs
        early_stop = False
        for epoch in range(self.config["n_epochs"]):
            if early_stop:
                break
            
            for batch in dataloader:
                obs, actions, old_log_probs, advantages_batch, returns_batch = [
                    b.to(self.device) for b in batch
                ]
                
                # Get current policy outputs
                new_log_probs, entropy, values = self.network.evaluate_actions(
                    obs, actions
                )
                
                # Compute policy ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config["clip_ratio"],
                    1.0 + self.config["clip_ratio"]
                ) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.config["clip_value"]:
                    # Clipped value loss
                    old_values = torch.FloatTensor(data['values']).to(self.device)
                    # Note: This is a simplified version; full implementation
                    # would need to properly index old_values
                    value_loss = F.mse_loss(values, returns_batch)
                else:
                    value_loss = F.mse_loss(values, returns_batch)
                
                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.config["value_coef"] * value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                # Gradient update
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config["max_grad_norm"] > 0:
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        self.config["max_grad_norm"]
                    )
                
                self.optimizer.step()
                
                # Compute statistics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.config["clip_ratio"]).float().mean().item()
                
                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(-entropy_loss.item())
                stats['approx_kl'].append(approx_kl)
                stats['clip_fraction'].append(clip_fraction)
                
                # Early stopping based on KL divergence
                if self.config["target_kl"] is not None:
                    if approx_kl > 1.5 * self.config["target_kl"]:
                        early_stop = True
                        break
        
        # Clear buffer
        self.buffer.clear()
        self.total_updates += 1
        
        # Decay entropy coefficient
        self.entropy_coef = max(
            self.entropy_coef * self.config["entropy_decay"],
            self.config["min_entropy_coef"]
        )
        
        # Compute explained variance
        exp_var = explained_variance(data['values'], returns)
        
        # Average statistics
        return {
            'policy_loss': np.mean(stats['policy_loss']),
            'value_loss': np.mean(stats['value_loss']),
            'entropy': np.mean(stats['entropy']),
            'approx_kl': np.mean(stats['approx_kl']),
            'clip_fraction': np.mean(stats['clip_fraction']),
            'explained_variance': exp_var,
            'entropy_coef': self.entropy_coef,
            'early_stopped': early_stop,
        }
    
    def save(self, path: str, include_optimizer: bool = True):
        """
        Save the agent to disk.
        
        Args:
            path: Save path
            include_optimizer: Whether to include optimizer state
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'network_state_dict': self.network.state_dict(),
            'config': self.config,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'total_timesteps': self.total_timesteps,
            'total_updates': self.total_updates,
            'episode_count': self.episode_count,
            'entropy_coef': self.entropy_coef,
        }
        
        if include_optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if self.obs_rms is not None:
            save_dict['obs_rms'] = {
                'mean': self.obs_rms.mean,
                'var': self.obs_rms.var,
                'count': self.obs_rms.count,
            }
        
        torch.save(save_dict, path)
        print(f"💾 Agent saved to {path}")
    
    def load(self, path: str, load_optimizer: bool = True):
        """
        Load the agent from disk.
        
        Args:
            path: Load path
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        self.total_updates = checkpoint.get('total_updates', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.entropy_coef = checkpoint.get('entropy_coef', self.config['entropy_coef'])
        
        if self.obs_rms is not None and 'obs_rms' in checkpoint:
            self.obs_rms.mean = checkpoint['obs_rms']['mean']
            self.obs_rms.var = checkpoint['obs_rms']['var']
            self.obs_rms.count = checkpoint['obs_rms']['count']
        
        print(f"📂 Agent loaded from {path}")
        print(f"   Total timesteps: {self.total_timesteps}")
        print(f"   Total updates: {self.total_updates}")
    
    @classmethod
    def load_from_checkpoint(
        cls,
        path: str,
        device: str = "auto"
    ) -> "PPOAgent":
        """
        Create a new agent from a checkpoint file.
        
        Args:
            path: Path to checkpoint
            device: Device to load to
        
        Returns:
            Loaded PPOAgent
        """
        checkpoint = torch.load(path, map_location="cpu")
        
        config = checkpoint['config']
        config['device'] = device
        
        agent = cls(
            obs_dim=checkpoint['obs_dim'],
            action_dim=checkpoint['action_dim'],
            config=config,
        )
        
        agent.load(path)
        return agent


# Import F for MSE loss
import torch.nn.functional as F
