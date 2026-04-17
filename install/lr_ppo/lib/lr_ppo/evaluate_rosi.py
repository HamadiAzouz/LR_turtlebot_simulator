#!/usr/bin/env python3
"""
ROSI Evaluation Script
=======================

Evaluate a trained ROSI agent across multiple mazes and record
comprehensive metrics including success rate, path length, and
collision statistics.

Usage:
    ros2 run lr_ppo evaluate_rosi.py --model path/to/model.pt --maze maze_1.world

Prerequisites:
    1. Launch the simulation first:
       ros2 launch lr_turtlebot_sim turtlebot_in_maze.launch.py world:=maze_1.world
    
    2. Then run evaluation:
       ros2 run lr_ppo evaluate_rosi.py --model models/best_model.pt
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

try:
    import torch
except ImportError:
    print("❌ PyTorch not found! Install it with: pip install torch")
    sys.exit(1)

import rclpy
from rclpy.node import Node

from lr_ppo.environment import RosiMazeEnv
from lr_ppo.ppo_agent import PPOAgent


class EpisodeRecorder:
    """Record detailed episode data for analysis."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset recorder for new episode."""
        self.positions = []
        self.actions = []
        self.rewards = []
        self.distances_to_goal = []
        self.min_obstacle_distances = []
    
    def record_step(
        self,
        position: np.ndarray,
        action: np.ndarray,
        reward: float,
        info: Dict[str, Any]
    ):
        """Record a single step."""
        self.positions.append(position.tolist())
        self.actions.append(action.tolist())
        self.rewards.append(reward)
        self.distances_to_goal.append(info.get('distance_to_goal', 0))
        self.min_obstacle_distances.append(info.get('min_obstacle_distance', 0))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get episode summary statistics."""
        positions = np.array(self.positions)
        
        # Calculate path length
        if len(positions) > 1:
            path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        else:
            path_length = 0.0
        
        return {
            'path_length': float(path_length),
            'total_reward': float(sum(self.rewards)),
            'num_steps': len(self.rewards),
            'min_distance_to_goal': float(min(self.distances_to_goal)) if self.distances_to_goal else 0,
            'min_obstacle_distance': float(min(self.min_obstacle_distances)) if self.min_obstacle_distances else 0,
            'positions': self.positions,
        }


def evaluate_model(
    env: RosiMazeEnv,
    agent: PPOAgent,
    n_episodes: int,
    deterministic: bool = True,
    verbose: bool = True,
    record_trajectories: bool = False,
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation of a trained model.
    
    Args:
        env: Environment to evaluate in
        agent: Trained PPO agent
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy (no exploration)
        verbose: Print episode results
        record_trajectories: Record full trajectories
    
    Returns:
        Dictionary of evaluation results
    """
    results = {
        'episodes': [],
        'rewards': [],
        'lengths': [],
        'successes': [],
        'collisions': [],
        'timeouts': [],
        'path_lengths': [],
        'min_distances_to_goal': [],
    }
    
    trajectories = [] if record_trajectories else None
    
    print(f"\n🔍 Evaluating for {n_episodes} episodes...")
    print("-" * 50)
    
    for episode in range(n_episodes):
        recorder = EpisodeRecorder()
        obs, info = env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        success = False
        collision = False
        timeout = False
        
        done = False
        while not done:
            # Get action
            action, _, _ = agent.select_action(obs, deterministic=deterministic)
            
            # Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            # Record step
            state = env.get_state()
            recorder.record_step(state['position'], action, reward, step_info)
            
            # Update counters
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            # Check outcome
            if step_info.get('success', False):
                success = True
            if step_info.get('collision', False):
                collision = True
            if step_info.get('timeout', False):
                timeout = True
        
        # Get episode summary
        summary = recorder.get_summary()
        
        # Store results
        results['episodes'].append(episode + 1)
        results['rewards'].append(episode_reward)
        results['lengths'].append(episode_length)
        results['successes'].append(success)
        results['collisions'].append(collision)
        results['timeouts'].append(timeout)
        results['path_lengths'].append(summary['path_length'])
        results['min_distances_to_goal'].append(summary['min_distance_to_goal'])
        
        if record_trajectories:
            trajectories.append(summary['positions'])
        
        # Print progress
        if verbose:
            status = "✅" if success else ("💥" if collision else "⏰")
            print(f"  Episode {episode+1:3d}: {status} "
                  f"Reward: {episode_reward:7.2f}, "
                  f"Steps: {episode_length:4d}, "
                  f"Path: {summary['path_length']:6.2f}m")
    
    # Calculate aggregate statistics
    results['aggregate'] = {
        'mean_reward': float(np.mean(results['rewards'])),
        'std_reward': float(np.std(results['rewards'])),
        'mean_length': float(np.mean(results['lengths'])),
        'std_length': float(np.std(results['lengths'])),
        'success_rate': float(np.mean(results['successes'])),
        'collision_rate': float(np.mean(results['collisions'])),
        'timeout_rate': float(np.mean(results['timeouts'])),
        'mean_path_length': float(np.mean(results['path_lengths'])),
        'n_episodes': n_episodes,
    }
    
    if record_trajectories:
        results['trajectories'] = trajectories
    
    return results


def print_results(results: Dict[str, Any], maze_name: str = ""):
    """Print formatted evaluation results."""
    agg = results['aggregate']
    
    print("\n" + "=" * 60)
    print(f"📊 EVALUATION RESULTS" + (f" - {maze_name}" if maze_name else ""))
    print("=" * 60)
    
    print(f"\n🎯 Performance Metrics:")
    print(f"   Success Rate:    {agg['success_rate']*100:6.1f}%")
    print(f"   Collision Rate:  {agg['collision_rate']*100:6.1f}%")
    print(f"   Timeout Rate:    {agg['timeout_rate']*100:6.1f}%")
    
    print(f"\n📈 Episode Statistics:")
    print(f"   Mean Reward:     {agg['mean_reward']:7.2f} ± {agg['std_reward']:.2f}")
    print(f"   Mean Length:     {agg['mean_length']:7.1f} ± {agg['std_length']:.1f} steps")
    print(f"   Mean Path:       {agg['mean_path_length']:7.2f} m")
    
    print(f"\n📝 Total Episodes: {agg['n_episodes']}")
    print("=" * 60)


def compare_models(
    env: RosiMazeEnv,
    model_paths: List[str],
    n_episodes: int = 20,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models on the same environment.
    
    Args:
        env: Environment for evaluation
        model_paths: List of paths to model checkpoints
        n_episodes: Episodes per model
    
    Returns:
        Dictionary mapping model names to results
    """
    comparison = {}
    
    for model_path in model_paths:
        model_name = Path(model_path).stem
        print(f"\n📂 Loading model: {model_name}")
        
        agent = PPOAgent.load_from_checkpoint(model_path)
        results = evaluate_model(env, agent, n_episodes, verbose=False)
        
        comparison[model_name] = results['aggregate']
        print_results(results, model_name)
    
    return comparison


def evaluate_across_mazes(
    agent: PPOAgent,
    mazes: List[Dict[str, Any]],
    n_episodes: int = 20,
    env_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate agent across multiple maze configurations.
    
    Note: This requires restarting the simulation for each maze.
    In practice, you'd run separate evaluation scripts for each maze.
    
    Args:
        agent: Trained agent
        mazes: List of maze configurations with 'name', 'goal', etc.
        n_episodes: Episodes per maze
        env_config: Base environment configuration
    
    Returns:
        Dictionary mapping maze names to results
    """
    results = {}
    env_config = env_config or {}
    
    for maze_config in mazes:
        maze_name = maze_config['name']
        print(f"\n🌍 Evaluating on: {maze_name}")
        
        # Update config with maze-specific settings
        config = {**env_config, **maze_config}
        
        env = RosiMazeEnv(config=config)
        maze_results = evaluate_model(env, agent, n_episodes, verbose=True)
        results[maze_name] = maze_results
        
        print_results(maze_results, maze_name)
        env.close()
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ROSI model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint")
    
    # Environment arguments
    parser.add_argument("--maze", type=str, default="maze_1.world",
                       help="Maze world file name")
    parser.add_argument("--goal", type=float, nargs=2, default=[2.0, 2.0],
                       help="Goal position (x, y)")
    parser.add_argument("--lidar-samples", type=int, default=24,
                       help="Number of downsampled LiDAR rays")
    parser.add_argument("--max-steps", type=int, default=500,
                       help="Maximum steps per episode")
    
    # Evaluation arguments
    parser.add_argument("--episodes", type=int, default=50,
                       help="Number of evaluation episodes")
    parser.add_argument("--stochastic", action="store_true",
                       help="Use stochastic policy (with exploration)")
    parser.add_argument("--record-trajectories", action="store_true",
                       help="Record full trajectories for visualization")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                       help="Directory for evaluation results")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    # Multiple model comparison
    parser.add_argument("--compare", type=str, nargs="+",
                       help="Compare multiple models (provide paths)")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Environment configuration
        env_config = {
            "num_lidar_samples": args.lidar_samples,
            "max_episode_steps": args.max_steps,
            "goal_position": tuple(args.goal),
        }
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.compare:
            # Compare multiple models
            print("🔄 Comparing multiple models...")
            env = RosiMazeEnv(config=env_config)
            comparison = compare_models(env, args.compare, args.episodes)
            env.close()
            
            # Save comparison results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"comparison_{timestamp}.json"
            with open(output_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\n💾 Comparison saved to: {output_path}")
            
        else:
            # Single model evaluation
            print(f"\n📂 Loading model: {args.model}")
            agent = PPOAgent.load_from_checkpoint(args.model)
            
            print(f"🌍 Creating environment...")
            env = RosiMazeEnv(config=env_config)
            
            results = evaluate_model(
                env, agent, args.episodes,
                deterministic=not args.stochastic,
                verbose=not args.quiet,
                record_trajectories=args.record_trajectories,
            )
            
            print_results(results, args.maze)
            
            # Save results
            model_name = Path(args.model).stem
            maze_name = args.maze.replace('.world', '')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"eval_{model_name}_{maze_name}_{timestamp}.json"
            
            # Remove trajectories for JSON (can be large)
            save_results = {k: v for k, v in results.items() if k != 'trajectories'}
            save_results['model'] = args.model
            save_results['maze'] = args.maze
            save_results['goal'] = args.goal
            
            with open(output_path, 'w') as f:
                json.dump(save_results, f, indent=2)
            print(f"\n💾 Results saved to: {output_path}")
            
            env.close()
    
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted!")
    
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
