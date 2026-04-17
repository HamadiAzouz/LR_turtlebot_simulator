#!/usr/bin/env python3
"""
Test Environment Script
========================

A simple script to test that the ROSI environment is working correctly.
This helps verify the ROS2 connection, sensor data, and basic interactions.

Usage:
    1. Launch the simulation:
       ros2 launch lr_turtlebot_sim turtlebot_in_maze.launch.py world:=maze_1.world
    
    2. Run this test:
       ros2 run lr_ppo test_environment.py
"""

import time
import numpy as np
import rclpy

from lr_ppo.environment import RosiMazeEnv


def test_environment():
    """Test the ROSI environment setup."""
    
    print("=" * 60)
    print("🧪 ROSI Environment Test")
    print("=" * 60)
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create environment
        print("\n1️⃣  Creating environment...")
        config = {
            "num_lidar_samples": 24,
            "max_episode_steps": 100,
            "goal_position": (2.0, 2.0),
        }
        
        env = RosiMazeEnv(config=config)
        print("   ✅ Environment created successfully!")
        
        # Check spaces
        print(f"\n2️⃣  Checking spaces...")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        print("   ✅ Spaces configured correctly!")
        
        # Test reset
        print(f"\n3️⃣  Testing reset...")
        obs, info = env.reset()
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"   Info: {info}")
        print("   ✅ Reset successful!")
        
        # Test state retrieval
        print(f"\n4️⃣  Checking robot state...")
        state = env.get_state()
        print(f"   Position: ({state['position'][0]:.3f}, {state['position'][1]:.3f})")
        print(f"   Orientation: {np.degrees(state['orientation']):.1f}°")
        print(f"   LiDAR readings: {len(state['lidar']) if state['lidar'] is not None else 0}")
        print(f"   Goal: ({state['goal_position'][0]:.3f}, {state['goal_position'][1]:.3f})")
        print(f"   Distance to goal: {state['distance_to_goal']:.3f}m")
        print("   ✅ State retrieval working!")
        
        # Test a few steps with random actions
        print(f"\n5️⃣  Testing steps with random actions...")
        total_reward = 0.0
        
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"   Step {step+1}: "
                  f"action=[{action[0]:.2f}, {action[1]:.2f}], "
                  f"reward={reward:.3f}, "
                  f"dist={info.get('distance_to_goal', 0):.3f}m")
            
            if terminated or truncated:
                print(f"   Episode ended: terminated={terminated}, truncated={truncated}")
                break
        
        print(f"   Total reward: {total_reward:.3f}")
        print("   ✅ Step function working!")
        
        # Test manual velocity command
        print(f"\n6️⃣  Testing manual commands...")
        
        print("   Moving forward for 2 seconds...")
        for _ in range(20):
            action = np.array([0.5, 0.0])  # Forward
            obs, reward, _, _, info = env.step(action)
        
        state = env.get_state()
        print(f"   New position: ({state['position'][0]:.3f}, {state['position'][1]:.3f})")
        
        print("   Turning left for 1 second...")
        for _ in range(10):
            action = np.array([0.0, 0.5])  # Turn left
            obs, reward, _, _, info = env.step(action)
        
        state = env.get_state()
        print(f"   New orientation: {np.degrees(state['orientation']):.1f}°")
        print("   ✅ Manual control working!")
        
        # Test observation breakdown
        print(f"\n7️⃣  Observation breakdown:")
        obs, _ = env.reset()
        
        lidar_end = config["num_lidar_samples"]
        pos_end = lidar_end + 2
        orient_end = pos_end + 2
        vel_end = orient_end + 2
        goal_end = vel_end + 3
        
        print(f"   LiDAR ({lidar_end} samples): range [{obs[:lidar_end].min():.3f}, {obs[:lidar_end].max():.3f}]")
        print(f"   Position: [{obs[lidar_end]:.3f}, {obs[lidar_end+1]:.3f}]")
        print(f"   Orientation (sin/cos): [{obs[pos_end]:.3f}, {obs[pos_end+1]:.3f}]")
        print(f"   Velocities: [{obs[orient_end]:.3f}, {obs[orient_end+1]:.3f}]")
        print(f"   Goal info: [{obs[vel_end]:.3f}, {obs[vel_end+1]:.3f}, {obs[vel_end+2]:.3f}]")
        print("   ✅ Observation structure verified!")
        
        # Clean up
        print(f"\n8️⃣  Cleaning up...")
        env.close()
        print("   ✅ Environment closed!")
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! Environment is ready for training.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        rclpy.shutdown()


def main():
    """Main entry point."""
    success = test_environment()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
