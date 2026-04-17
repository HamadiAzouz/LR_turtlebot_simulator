"""
Neural Network Architectures for PPO
=====================================

This module defines the Actor-Critic neural network architectures
used for ROSI's PPO training.

The Actor outputs action distribution parameters (mean and std for
continuous actions), while the Critic estimates state values.

Enhanced architecture includes:
- 1D Convolutional layers for LiDAR processing
- Attention mechanism for spatial awareness
- Layer normalization for training stability
- Residual connections for better gradient flow
- Separate processing paths for different input modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Optional


class LidarEncoder(nn.Module):
    """
    1D Convolutional encoder for LiDAR data.
    Processes spatial patterns in laser scan data.
    """
    def __init__(self, num_lidar_samples: int = 24, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.norm = nn.LayerNorm(hidden_dim * 4)
        
    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        # lidar: [batch, num_samples]
        x = lidar.unsqueeze(1)  # [batch, 1, num_samples]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # [batch, hidden_dim, 4]
        x = x.flatten(1)  # [batch, hidden_dim * 4]
        x = self.norm(x)
        return x


class AttentionBlock(nn.Module):
    """
    Self-attention for spatial awareness of obstacles.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)


class ResidualBlock(nn.Module):
    """
    Residual block with layer normalization.
    """
    def __init__(self, dim: int, activation: type = nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.activation = activation()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.activation(self.fc1(x))
        x = self.norm2(x)
        x = self.fc2(x)
        return x + residual


def init_weights(module: nn.Module, gain: float = np.sqrt(2)):
    """
    Initialize network weights using orthogonal initialization.
    
    Orthogonal initialization helps with gradient flow in deep networks.
    
    Args:
        module: PyTorch module to initialize
        gain: Gain factor for initialization
    """
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    
    Architecture:
    - Shared feature extraction layers
    - Separate heads for policy (actor) and value (critic)
    
    The actor outputs parameters of a Gaussian distribution
    for continuous action selection.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: str = "tanh",
        log_std_init: float = -0.5,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        """
        Initialize the Actor-Critic network.
        
        Args:
            obs_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_sizes: Sizes of hidden layers
            activation: Activation function ("tanh" or "relu")
            log_std_init: Initial value for log standard deviation
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Select activation function
        if activation == "tanh":
            self.activation = nn.Tanh
        elif activation == "relu":
            self.activation = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build shared feature extractor
        self.shared_net = self._build_mlp(
            input_dim=obs_dim,
            hidden_sizes=hidden_sizes[:-1] if len(hidden_sizes) > 1 else hidden_sizes,
            activation=self.activation
        )
        
        shared_out_dim = hidden_sizes[-2] if len(hidden_sizes) > 1 else hidden_sizes[0]
        
        # Actor head (policy)
        self.actor_net = nn.Sequential(
            nn.Linear(shared_out_dim, hidden_sizes[-1]),
            self.activation(),
            nn.Linear(hidden_sizes[-1], action_dim)
        )
        
        # Learnable log standard deviation
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )
        
        # Critic head (value function)
        self.critic_net = nn.Sequential(
            nn.Linear(shared_out_dim, hidden_sizes[-1]),
            self.activation(),
            nn.Linear(hidden_sizes[-1], 1)
        )
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        # Use smaller gain for output layers
        init_weights(self.actor_net[-1], gain=0.01)
        init_weights(self.critic_net[-1], gain=1.0)
    
    def _build_mlp(
        self,
        input_dim: int,
        hidden_sizes: Tuple[int, ...],
        activation: type
    ) -> nn.Sequential:
        """Build a multi-layer perceptron."""
        layers = []
        prev_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_size),
                activation()
            ])
            prev_dim = hidden_size
        
        return nn.Sequential(*layers)
    
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            deterministic: If True, return mean action (no sampling)
        
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        # Get shared features
        features = self.shared_net(obs)
        
        # Get action distribution parameters
        action_mean = self.actor_net(features)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action_std = log_std.exp()
        
        # Create distribution
        dist = Normal(action_mean, action_std)
        
        # Sample action
        if deterministic:
            action = action_mean
        else:
            action = dist.rsample()  # Reparameterized sample
        
        # Compute log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Compute entropy
        entropy = dist.entropy().sum(dim=-1)
        
        # Get value estimate
        value = self.critic_net(features).squeeze(-1)
        
        # Squash action to [-1, 1]
        action = torch.tanh(action)
        
        return action, log_prob, entropy, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for given observations.
        
        Args:
            obs: Observation tensor
        
        Returns:
            Value estimates
        """
        features = self.shared_net(obs)
        return self.critic_net(features).squeeze(-1)
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action for given observation.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return mean action
        
        Returns:
            Tuple of (action, log_prob)
        """
        features = self.shared_net(obs)
        action_mean = self.actor_net(features)
        
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action_std = log_std.exp()
        
        dist = Normal(action_mean, action_std)
        
        if deterministic:
            action = action_mean
        else:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        action = torch.tanh(action)
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.
        
        Used during PPO update to compute policy ratio.
        
        Args:
            obs: Observation tensor
            actions: Action tensor (should be in [-1, 1] range, will be unquashed)
        
        Returns:
            Tuple of (log_prob, entropy, value)
        """
        features = self.shared_net(obs)
        action_mean = self.actor_net(features)
        
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action_std = log_std.exp()
        
        dist = Normal(action_mean, action_std)
        
        # Unquash actions for log prob computation
        # actions are in [-1, 1], need to undo tanh
        # Using inverse tanh: arctanh(x) = 0.5 * log((1+x)/(1-x))
        actions_unquashed = torch.clamp(actions, -0.999, 0.999)
        actions_unquashed = 0.5 * torch.log((1 + actions_unquashed) / (1 - actions_unquashed))
        
        log_prob = dist.log_prob(actions_unquashed).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic_net(features).squeeze(-1)
        
        return log_prob, entropy, value


class SeparateActorCritic(nn.Module):
    """
    Actor-Critic with completely separate networks.
    
    This variant doesn't share any layers between actor and critic,
    which can sometimes lead to better training stability.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_sizes: Tuple[int, ...] = (256, 256),
        critic_hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: str = "tanh",
        log_std_init: float = -0.5,
    ):
        """
        Initialize separate Actor-Critic networks.
        
        Args:
            obs_dim: Dimension of observation space
            action_dim: Dimension of action space
            actor_hidden_sizes: Hidden layer sizes for actor
            critic_hidden_sizes: Hidden layer sizes for critic
            activation: Activation function
            log_std_init: Initial log standard deviation
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        if activation == "tanh":
            act_fn = nn.Tanh
        else:
            act_fn = nn.ReLU
        
        # Build actor network
        actor_layers = []
        prev_dim = obs_dim
        for hidden_size in actor_hidden_sizes:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_size),
                act_fn()
            ])
            prev_dim = hidden_size
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Build critic network
        critic_layers = []
        prev_dim = obs_dim
        for hidden_size in critic_hidden_sizes:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_size),
                act_fn()
            ])
            prev_dim = hidden_size
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Learnable log std
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        # Initialize
        self.apply(lambda m: init_weights(m))
    
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        action_mean = self.actor(obs)
        action_std = self.log_std.exp()
        
        dist = Normal(action_mean, action_std)
        
        if deterministic:
            action = action_mean
        else:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        action = torch.tanh(action)
        
        return action, log_prob, entropy, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        return self.critic(obs).squeeze(-1)
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action."""
        action_mean = self.actor(obs)
        action_std = self.log_std.exp()
        
        dist = Normal(action_mean, action_std)
        
        if deterministic:
            action = action_mean
        else:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        action = torch.tanh(action)
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions."""
        action_mean = self.actor(obs)
        action_std = self.log_std.exp()
        
        dist = Normal(action_mean, action_std)
        
        actions_unquashed = torch.clamp(actions, -0.999, 0.999)
        actions_unquashed = 0.5 * torch.log((1 + actions_unquashed) / (1 - actions_unquashed))
        
        log_prob = dist.log_prob(actions_unquashed).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        
        return log_prob, entropy, value

class EnhancedActorCritic(nn.Module):
    """
    Enhanced Actor-Critic network with specialized architecture for maze navigation.
    
    Features:
    - 1D CNN for LiDAR spatial pattern recognition
    - Separate processing of position/velocity states
    - Deeper networks with residual connections
    - Layer normalization for stable training
    - State-dependent log_std for adaptive exploration
    
    Observation structure expected:
    - lidar: first 24 values (normalized laser scans)
    - position: next 2 values (x, y)
    - orientation: next 2 values (cos, sin of yaw)
    - velocities: next 2 values (linear, angular)
    - exploration_progress: last 1 value
    """
    
    def __init__(
        self,
        obs_dim: int = 31,
        action_dim: int = 2,
        num_lidar_samples: int = 24,
        hidden_dim: int = 256,
        num_residual_blocks: int = 3,
        log_std_init: float = 0.5,  # Higher initial exploration (was 0.0)
        log_std_min: float = -1.0,  # Allow less certainty (was -2.0)
        log_std_max: float = 1.0,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_lidar_samples = num_lidar_samples
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Dimensions of different observation components
        self.state_dim = obs_dim - num_lidar_samples  # position, orientation, velocities, exploration
        
        # LiDAR encoder (processes spatial patterns)
        self.lidar_encoder = LidarEncoder(num_lidar_samples, hidden_dim=64)
        lidar_out_dim = 64 * 4  # from adaptive pool
        
        # State encoder (position, orientation, velocities)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        # Combined feature dimension
        combined_dim = lidar_out_dim + 64
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # === ACTOR (Policy Network) ===
        self.actor_residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, nn.ReLU) for _ in range(num_residual_blocks)
        ])
        self.actor_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        
        # State-dependent log_std (learns when to explore more)
        self.log_std_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.log_std_bias = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        # === CRITIC (Value Network) ===
        self.critic_residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, nn.ReLU) for _ in range(num_residual_blocks)
        ])
        self.critic_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Custom weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Smaller init for output heads
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
    
    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation into feature vector."""
        # Split observation into lidar and state
        lidar = obs[:, :self.num_lidar_samples]
        state = obs[:, self.num_lidar_samples:]
        
        # Encode each modality
        lidar_features = self.lidar_encoder(lidar)
        state_features = self.state_encoder(state)
        
        # Fuse features
        combined = torch.cat([lidar_features, state_features], dim=-1)
        features = self.fusion(combined)
        
        return features
    
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        features = self._encode(obs)
        
        # Actor forward
        actor_features = features
        for block in self.actor_residual_blocks:
            actor_features = block(actor_features)
        action_mean = self.actor_head(actor_features)
        
        # State-dependent exploration
        log_std = self.log_std_net(actor_features) + self.log_std_bias
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        action_std = log_std.exp()
        
        # Create distribution and sample
        dist = Normal(action_mean, action_std)
        if deterministic:
            action = action_mean
        else:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # Critic forward
        critic_features = features
        for block in self.critic_residual_blocks:
            critic_features = block(critic_features)
        value = self.critic_head(critic_features).squeeze(-1)
        
        # Squash action to [-1, 1]
        action = torch.tanh(action)
        
        return action, log_prob, entropy, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        features = self._encode(obs)
        critic_features = features
        for block in self.critic_residual_blocks:
            critic_features = block(critic_features)
        return self.critic_head(critic_features).squeeze(-1)
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action for given observation."""
        features = self._encode(obs)
        
        actor_features = features
        for block in self.actor_residual_blocks:
            actor_features = block(actor_features)
        action_mean = self.actor_head(actor_features)
        
        log_std = self.log_std_net(actor_features) + self.log_std_bias
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        action_std = log_std.exp()
        
        dist = Normal(action_mean, action_std)
        if deterministic:
            action = action_mean
        else:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        action = torch.tanh(action)
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities and entropy for given actions."""
        features = self._encode(obs)
        
        # Actor
        actor_features = features
        for block in self.actor_residual_blocks:
            actor_features = block(actor_features)
        action_mean = self.actor_head(actor_features)
        
        log_std = self.log_std_net(actor_features) + self.log_std_bias
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        action_std = log_std.exp()
        
        dist = Normal(action_mean, action_std)
        
        # Unquash actions
        actions_unquashed = torch.clamp(actions, -0.999, 0.999)
        actions_unquashed = 0.5 * torch.log((1 + actions_unquashed) / (1 - actions_unquashed))
        
        log_prob = dist.log_prob(actions_unquashed).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        # Critic
        critic_features = features
        for block in self.critic_residual_blocks:
            critic_features = block(critic_features)
        value = self.critic_head(critic_features).squeeze(-1)
        
        return log_prob, entropy, value