#!/usr/bin/env python3
"""
ROSI PPO Training Script
=========================

Train ROSI the TurtleBot3 to navigate through mazes using
Proximal Policy Optimization (PPO).

Usage:
    ros2 run lr_ppo train_rosi.py --maze maze_1.world --episodes 1000

Prerequisites:
    1. Launch the simulation first:
       ros2 launch lr_turtlebot_sim turtlebot_in_maze.launch.py world:=maze_1.world
    
    2. Then run training:
       ros2 run lr_ppo train_rosi.py
"""

import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Check for PyTorch and install hints
try:
    import torch
except ImportError:
    print("❌ PyTorch not found! Install it with:")
    print("   pip install torch")
    sys.exit(1)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard not available. Install with: pip install tensorboard")

import rclpy
from rclpy.node import Node

from lr_ppo.environment import RosiMazeEnv
from lr_ppo.ppo_agent import PPOAgent


class TrainingLogger:
    """Logger for training metrics and progress."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True
    ):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        
        # Episode history
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_collisions = []
        
        # Training history
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        
        # Best metrics
        self.best_reward = -float('inf')
        self.best_success_rate = 0.0
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        success: bool,
        collision: bool,
        info: Dict[str, Any]
    ):
        """Log episode results."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_successes.append(float(success))
        self.episode_collisions.append(float(collision))
        
        if self.use_tensorboard:
            self.writer.add_scalar("episode/reward", reward, episode)
            self.writer.add_scalar("episode/length", length, episode)
            self.writer.add_scalar("episode/success", float(success), episode)
            self.writer.add_scalar("episode/collision", float(collision), episode)
            
            # Rolling averages
            window = min(100, len(self.episode_rewards))
            avg_reward = np.mean(self.episode_rewards[-window:])
            avg_success = np.mean(self.episode_successes[-window:])
            self.writer.add_scalar("episode/avg_reward_100", avg_reward, episode)
            self.writer.add_scalar("episode/success_rate_100", avg_success, episode)
    
    def log_update(
        self,
        update: int,
        stats: Dict[str, float],
        timesteps: int
    ):
        """Log PPO update statistics."""
        self.policy_losses.append(stats.get('policy_loss', 0))
        self.value_losses.append(stats.get('value_loss', 0))
        self.entropies.append(stats.get('entropy', 0))
        
        if self.use_tensorboard:
            self.writer.add_scalar("train/policy_loss", stats['policy_loss'], timesteps)
            self.writer.add_scalar("train/value_loss", stats['value_loss'], timesteps)
            self.writer.add_scalar("train/entropy", stats['entropy'], timesteps)
            self.writer.add_scalar("train/approx_kl", stats['approx_kl'], timesteps)
            self.writer.add_scalar("train/clip_fraction", stats['clip_fraction'], timesteps)
            self.writer.add_scalar("train/explained_variance", stats['explained_variance'], timesteps)
            self.writer.add_scalar("train/entropy_coef", stats['entropy_coef'], timesteps)
    
    def log_evaluation(
        self,
        episode: int,
        success_rate: float,
        avg_reward: float,
        avg_length: float
    ):
        """Log evaluation results."""
        if self.use_tensorboard:
            self.writer.add_scalar("eval/success_rate", success_rate, episode)
            self.writer.add_scalar("eval/avg_reward", avg_reward, episode)
            self.writer.add_scalar("eval/avg_length", avg_length, episode)
    
    def check_best(self, reward: float, success_rate: float) -> tuple:
        """Check if current metrics are best so far."""
        reward_best = reward > self.best_reward
        success_best = success_rate > self.best_success_rate
        
        if reward_best:
            self.best_reward = reward
        if success_best:
            self.best_success_rate = success_rate
        
        return reward_best, success_best
    
    def save_history(self):
        """Save training history to JSON."""
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_successes': self.episode_successes,
            'episode_collisions': self.episode_collisions,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropies': self.entropies,
            'best_reward': self.best_reward,
            'best_success_rate': self.best_success_rate,
        }
        
        with open(self.log_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    def close(self):
        """Close the logger."""
        if self.use_tensorboard:
            self.writer.close()
        self.save_history()


def create_experiment_name(args) -> str:
    """Create a unique experiment name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    maze = args.maze.replace('.world', '')
    return f"rosi_ppo_{maze}_{timestamp}"


def train_rosi(args):
    """
    Main training function for ROSI.
    
    Args:
        args: Command line arguments
    """
    print("=" * 60)
    print("🤖 ROSI PPO Training")
    print("=" * 60)
    
    # Create experiment name and directories
    experiment_name = create_experiment_name(args)
    model_dir = Path(args.model_dir) / experiment_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Experiment: {experiment_name}")
    print(f"📁 Model directory: {model_dir}")
    
    # Set initial position based on maze (different mazes have different valid spawn points)
    maze_initial_positions = {
        "maze_1": (-2.0, -0.5),
        "maze_2": (-2.0, -0.5),
        "maze_3": (-6.0, -4.0),  # Inside the first corridor
    }
    maze_key = args.maze.replace('.world', '')
    initial_pos = maze_initial_positions.get(maze_key, (-2.0, -0.5))
    
    # Environment configuration
    env_config = {
        "num_lidar_samples": args.lidar_samples,
        "max_episode_steps": args.max_steps,
        "goal_position": tuple(args.goal),
        "collision_threshold": args.collision_threshold,
        "goal_threshold": args.goal_threshold,
        "randomize_goal": args.randomize_goal,
        "reward_goal": 100.0,
        "reward_collision": -100.0,
        "reward_progress_scale": args.progress_scale,
        "reward_time_penalty": -0.1,
        # Set world name based on maze argument (maze_1 -> maze_world_1, etc.)
        "world_name": args.maze.replace('.world', '').replace('maze_', 'maze_world_'),
        # Set initial position based on maze
        "initial_position": initial_pos,
    }
    
    # Create environment
    print("\n🌍 Creating environment...")
    print(f"   World name: {env_config['world_name']}")
    print(f"   Initial position: {env_config['initial_position']}")
    env = RosiMazeEnv(config=env_config)
    
    # Agent configuration
    agent_config = {
        "hidden_sizes": tuple(args.hidden_sizes),
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_ratio": args.clip_ratio,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "rollout_length": args.rollout_length,
        "entropy_coef": args.entropy_coef,
        "normalize_observations": args.normalize_obs,
    }
    
    # Create agent
    print("🧠 Creating PPO agent...")
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config=agent_config,
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"📂 Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)
    
    # Create logger
    logger = TrainingLogger(
        log_dir=args.log_dir,
        experiment_name=experiment_name,
        use_tensorboard=not args.no_tensorboard
    )
    
    # Save configuration
    config = {
        'env_config': env_config,
        'agent_config': agent_config,
        'args': vars(args),
    }
    with open(model_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    print("\n" + "=" * 60)
    print("🎮 Starting training...")
    print("=" * 60)
    
    total_episodes = 0
    total_timesteps = 0
    update_count = 0
    
    try:
        while total_episodes < args.episodes:
            # Reset environment
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_success = False
            episode_collision = False
            
            # Episode loop
            done = False
            while not done:
                # Select action
                action, log_prob, value = agent.select_action(obs)
                
                # Step environment
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                
                # Store transition
                agent.store_transition(
                    obs, action, reward, next_obs, done, log_prob, value
                )
                
                # Update counters
                episode_reward += reward
                episode_length += 1
                total_timesteps += 1
                obs = next_obs
                
                # Check episode outcome
                if terminated:
                    if step_info.get('success', False):
                        episode_success = True
                    if step_info.get('collision', False):
                        episode_collision = True
                
                # PPO Update when buffer is full
                if len(agent.buffer) >= args.rollout_length:
                    update_stats = agent.update()
                    update_count += 1
                    logger.log_update(update_count, update_stats, total_timesteps)
                    
                    if args.verbose and update_count % 10 == 0:
                        print(f"  📊 Update {update_count}: "
                              f"Policy Loss: {update_stats['policy_loss']:.4f}, "
                              f"Value Loss: {update_stats['value_loss']:.4f}, "
                              f"Entropy: {update_stats['entropy']:.4f}")
            
            # Episode completed
            total_episodes += 1
            agent.episode_count = total_episodes
            
            # Log episode
            logger.log_episode(
                total_episodes, episode_reward, episode_length,
                episode_success, episode_collision, step_info
            )
            
            # Print progress
            if total_episodes % args.log_interval == 0:
                window = min(100, total_episodes)
                avg_reward = np.mean(logger.episode_rewards[-window:])
                avg_success = np.mean(logger.episode_successes[-window:])
                avg_length = np.mean(logger.episode_lengths[-window:])
                
                status = "🎉 SUCCESS" if episode_success else ("💥 COLLISION" if episode_collision else "⏰ TIMEOUT")
                
                print(f"\n📈 Episode {total_episodes}/{args.episodes} {status}")
                print(f"   Reward: {episode_reward:.2f} (avg: {avg_reward:.2f})")
                print(f"   Length: {episode_length} (avg: {avg_length:.1f})")
                print(f"   Success Rate: {avg_success*100:.1f}%")
                print(f"   Total Timesteps: {total_timesteps}")
            
            # Save checkpoint
            if total_episodes % args.save_interval == 0:
                checkpoint_path = model_dir / f"checkpoint_{total_episodes}.pt"
                agent.save(str(checkpoint_path))
                
                # Check for best model
                window = min(100, total_episodes)
                avg_reward = np.mean(logger.episode_rewards[-window:])
                avg_success = np.mean(logger.episode_successes[-window:])
                reward_best, success_best = logger.check_best(avg_reward, avg_success)
                
                if reward_best:
                    best_path = model_dir / "best_reward.pt"
                    agent.save(str(best_path))
                    print(f"   🏆 New best reward model saved!")
                
                if success_best:
                    best_path = model_dir / "best_success.pt"
                    agent.save(str(best_path))
                    print(f"   🏆 New best success rate model saved!")
            
            # Evaluation
            if args.eval_interval > 0 and total_episodes % args.eval_interval == 0:
                print("\n🔍 Running evaluation...")
                eval_results = evaluate_agent(env, agent, n_episodes=args.eval_episodes)
                logger.log_evaluation(
                    total_episodes,
                    eval_results['success_rate'],
                    eval_results['avg_reward'],
                    eval_results['avg_length']
                )
                print(f"   Eval Success Rate: {eval_results['success_rate']*100:.1f}%")
                print(f"   Eval Avg Reward: {eval_results['avg_reward']:.2f}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
    
    finally:
        # Save final model
        final_path = model_dir / "final_model.pt"
        agent.save(str(final_path))
        
        # Save training history
        logger.close()
        
        print("\n" + "=" * 60)
        print("🏁 Training Complete!")
        print("=" * 60)
        print(f"   Total Episodes: {total_episodes}")
        print(f"   Total Timesteps: {total_timesteps}")
        print(f"   Best Reward: {logger.best_reward:.2f}")
        print(f"   Best Success Rate: {logger.best_success_rate*100:.1f}%")
        print(f"   Models saved to: {model_dir}")
        
        # Clean up
        env.close()


def evaluate_agent(
    env: RosiMazeEnv,
    agent: PPOAgent,
    n_episodes: int = 10
) -> Dict[str, float]:
    """
    Evaluate the agent without training.
    
    Args:
        env: Environment to evaluate in
        agent: Agent to evaluate
        n_episodes: Number of evaluation episodes
    
    Returns:
        Dictionary of evaluation metrics
    """
    rewards = []
    lengths = []
    successes = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        success = False
        
        done = False
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if info.get('success', False):
                success = True
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        successes.append(float(success))
    
    return {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_length': np.mean(lengths),
        'success_rate': np.mean(successes),
        'n_episodes': n_episodes,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ROSI the TurtleBot3 with PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment arguments
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--maze", type=str, default="maze_1.world",
                          help="Maze world file name")
    env_group.add_argument("--goal", type=float, nargs=2, default=[2.0, -2.0],
                          help="Goal position (x, y)")
    env_group.add_argument("--randomize-goal", action="store_true",
                          help="Randomize goal position each episode")
    env_group.add_argument("--lidar-samples", type=int, default=24,
                          help="Number of downsampled LiDAR rays")
    env_group.add_argument("--max-steps", type=int, default=500,
                          help="Maximum steps per episode")
    env_group.add_argument("--collision-threshold", type=float, default=0.18,
                          help="Distance threshold for collision detection")
    env_group.add_argument("--goal-threshold", type=float, default=0.3,
                          help="Distance threshold for goal reached")
    env_group.add_argument("--progress-scale", type=float, default=10.0,
                          help="Scaling factor for progress reward")
    
    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--episodes", type=int, default=1000,
                            help="Total training episodes")
    train_group.add_argument("--rollout-length", type=int, default=2048,
                            help="Steps before PPO update")
    train_group.add_argument("--n-epochs", type=int, default=10,
                            help="PPO epochs per update")
    train_group.add_argument("--batch-size", type=int, default=64,
                            help="Mini-batch size for PPO")
    
    # PPO hyperparameters
    ppo_group = parser.add_argument_group("PPO Hyperparameters")
    ppo_group.add_argument("--lr", type=float, default=3e-4,
                          help="Learning rate")
    ppo_group.add_argument("--gamma", type=float, default=0.99,
                          help="Discount factor")
    ppo_group.add_argument("--gae-lambda", type=float, default=0.95,
                          help="GAE lambda parameter")
    ppo_group.add_argument("--clip-ratio", type=float, default=0.2,
                          help="PPO clip ratio")
    ppo_group.add_argument("--entropy-coef", type=float, default=0.01,
                          help="Entropy coefficient")
    ppo_group.add_argument("--normalize-obs", action="store_true",
                          help="Normalize observations")
    
    # Network arguments
    net_group = parser.add_argument_group("Network")
    net_group.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256],
                          help="Hidden layer sizes")
    
    # Logging arguments
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log-dir", type=str, default="./logs",
                          help="Directory for logs")
    log_group.add_argument("--model-dir", type=str, default="./models",
                          help="Directory for saved models")
    log_group.add_argument("--log-interval", type=int, default=10,
                          help="Episodes between log prints")
    log_group.add_argument("--save-interval", type=int, default=100,
                          help="Episodes between checkpoints")
    log_group.add_argument("--eval-interval", type=int, default=100,
                          help="Episodes between evaluations (0 to disable)")
    log_group.add_argument("--eval-episodes", type=int, default=10,
                          help="Number of evaluation episodes")
    log_group.add_argument("--no-tensorboard", action="store_true",
                          help="Disable TensorBoard logging")
    log_group.add_argument("--verbose", action="store_true",
                          help="Print detailed training info")
    
    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        train_rosi(args)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
