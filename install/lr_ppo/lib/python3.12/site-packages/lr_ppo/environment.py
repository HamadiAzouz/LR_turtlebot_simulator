"""
ROSI Maze Environment - Gymnasium Environment for TurtleBot3 Navigation
========================================================================

This module provides a Gymnasium-compatible environment for training
ROSI the TurtleBot3 to navigate through mazes using reinforcement learning.

Observation Space:
- LiDAR readings (downsampled to N rays)
- Robot position (x, y)
- Robot orientation (yaw as sin/cos)
- Current velocities (linear, angular)
- Goal position relative to robot

Action Space:
- Continuous: [linear_velocity, angular_velocity]

Rewards:
- Goal reached: +100
- Collision: -100
- Progress toward goal: scaled positive
- Time penalty: small negative per step
"""

import math
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from lr_ppo.utils import normalize_angle, euclidean_distance, angle_to_goal


class RosiMazeEnv(gym.Env):
    """
    Gymnasium environment for ROSI TurtleBot3 maze navigation.
    
    This environment interfaces with ROS2 and Gazebo to provide
    a reinforcement learning training environment for ROSI.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    # Default configuration
    DEFAULT_CONFIG = {
        # LiDAR configuration
        "num_lidar_samples": 24,         # Downsample 360 rays to this
        "lidar_max_range": 3.5,          # Maximum LiDAR range (meters)
        "lidar_min_range": 0.12,         # Minimum LiDAR range (meters)
        
        # Action limits (increased for faster movement)
        "max_linear_velocity": 0.8,      # m/s (increased from 0.22)
        "max_angular_velocity": 6.28318531,     # rad/s (doubled from 4.0)
        
        # Collision parameters
        "collision_threshold": 0.05,     # Distance to consider collision (meters) - actual wall contact only
        
        # Episode limits
        "max_episode_steps": 500,        # Maximum steps per episode
        "step_duration": 0.1,            # Duration of each step (seconds)
        
        # Exploration reward parameters
        "cell_size": 0.5,                # Size of grid cells for exploration (meters)
        "reward_new_cell": 10.0,         # Reward for visiting a new cell
        "reward_collision": -100.0,      # Penalty for collision
        "reward_survival": 0.1,          # Small reward for each step survived
        "reward_forward_motion": 0.5,    # Reward for moving forward (increased)
        "reward_rotation_penalty": -0.1, # Penalty for excessive rotation (increased)
        "reward_same_direction_spin": -0.5,  # Penalty for spinning in same direction
        "spin_detection_threshold": 5,   # Steps of same-direction rotation to trigger penalty
        
        # Initial position
        "initial_position": (-2.0, -0.5),
        
        # Maze bounds for grid
        "maze_x_min": -5.0,
        "maze_x_max": 5.0,
        "maze_y_min": -5.0,
        "maze_y_max": 5.0,
        
        # World name for Gazebo services (maze_world_1, maze_world_2, maze_world_3)
        "world_name": "maze_world_1",
    }
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        node: Optional[Node] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the ROSI Maze Environment.
        
        Args:
            config: Configuration dictionary (overrides defaults)
            node: Existing ROS2 node (creates new one if None)
            render_mode: Rendering mode (currently only "human" supported)
        """
        super().__init__()
        
        # Merge config with defaults
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.render_mode = render_mode
        
        # Calculate observation space size
        # [lidar_samples + position(2) + orientation(2) + velocities(2) + exploration_info(1)]
        self.obs_size = (
            self.config["num_lidar_samples"] +  # Downsampled LiDAR
            2 +  # Position (x, y) - normalized
            2 +  # Orientation (sin(yaw), cos(yaw))
            2 +  # Velocities (linear, angular) - normalized
            1    # Exploration: percentage of cells visited
        )
        
        # Define observation space (all normalized to [-1, 1] or [0, 1])
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_size,),
            dtype=np.float32
        )
        
        # Define action space (continuous: linear_vel, angular_vel)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # ROS2 initialization
        self._owns_node = node is None
        if self._owns_node:
            if not rclpy.ok():
                rclpy.init()
            self.node = rclpy.create_node('rosi_maze_env')
        else:
            self.node = node
        
        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        self.cmd_vel_pub = self.node.create_publisher(
            TwistStamped, '/cmd_vel', 10
        )
        
        # Subscribers
        self.scan_sub = self.node.create_subscription(
            LaserScan, '/scan', self._scan_callback, sensor_qos
        )
        self.odom_sub = self.node.create_subscription(
            Odometry, '/odom', self._odom_callback, sensor_qos
        )
        
        # Service clients for simulation control (use configurable world name)
        world_name = self.config.get("world_name", "maze_world_1")
        self.reset_world_client = self.node.create_client(Empty, f'/world/{world_name}/control/reset')
        self.pause_client = self.node.create_client(Empty, f'/world/{world_name}/control/pause')
        self.unpause_client = self.node.create_client(Empty, f'/world/{world_name}/control/unpause')
        
        # State variables
        self._scan_data: Optional[np.ndarray] = None
        self._raw_odom_position: np.ndarray = np.array([0.0, 0.0])  # Raw odometry
        self._odom_offset: np.ndarray = np.array([0.0, 0.0])  # Offset to apply
        self._position: np.ndarray = np.array([0.0, 0.0])  # Corrected position
        self._orientation: float = 0.0  # yaw angle
        self._linear_velocity: float = 0.0
        self._angular_velocity: float = 0.0
        
        # Initial/start position for reset
        self._start_position: np.ndarray = np.array(self.config["initial_position"])
        
        # Exploration grid - track visited cells
        self._cell_size = self.config["cell_size"]
        self._visited_cells: set = set()
        self._total_possible_cells = self._calculate_total_cells()
        
        # Episode tracking
        self._episode_step = 0
        self._cumulative_reward = 0.0
        self._first_reset = True  # Track if this is the first reset
        
        # Anti-spin tracking (detect robot spinning in same direction)
        self._consecutive_same_rotation = 0
        self._last_rotation_sign = 0  # -1 for left, 1 for right, 0 for none
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Spin thread for ROS2 callbacks
        self._spinning = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()
        
        # Wait for first sensor data
        self._wait_for_sensors()
        
        # Set initial odometry offset based on spawn position
        # Robot spawns at initial_position, but odometry starts at (0,0)
        time.sleep(0.3)  # Wait for odometry to be available
        with self._lock:
            self._odom_offset[0] = self._start_position[0] - self._raw_odom_position[0]
            self._odom_offset[1] = self._start_position[1] - self._raw_odom_position[1]
            self._position[0] = self._start_position[0]
            self._position[1] = self._start_position[1]
        
        self.node.get_logger().info("🤖 ROSI Maze Environment initialized!")
    
    def _spin_loop(self):
        """Background thread for spinning ROS2 callbacks."""
        while self._spinning and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.01)
    
    def _calculate_total_cells(self) -> int:
        """Calculate the total number of possible cells in the maze grid."""
        x_cells = int((self.config["maze_x_max"] - self.config["maze_x_min"]) / self._cell_size)
        y_cells = int((self.config["maze_y_max"] - self.config["maze_y_min"]) / self._cell_size)
        return x_cells * y_cells
    
    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell coordinates."""
        cell_x = int((x - self.config["maze_x_min"]) / self._cell_size)
        cell_y = int((y - self.config["maze_y_min"]) / self._cell_size)
        return (cell_x, cell_y)
    
    def _mark_cell_visited(self) -> bool:
        """Mark current cell as visited. Returns True if it's a new cell."""
        cell = self._get_cell(self._position[0], self._position[1])
        if cell not in self._visited_cells:
            self._visited_cells.add(cell)
            return True
        return False
    
    def _scan_callback(self, msg: LaserScan):
        """Process incoming LiDAR scan data."""
        with self._lock:
            ranges = np.array(msg.ranges)
            # Replace inf values with max range
            ranges = np.where(np.isinf(ranges), self.config["lidar_max_range"], ranges)
            # Clip to valid range
            ranges = np.clip(ranges, self.config["lidar_min_range"], self.config["lidar_max_range"])
            self._scan_data = ranges
    
    def _odom_callback(self, msg: Odometry):
        """Process incoming odometry data."""
        with self._lock:
            # Extract raw odometry position
            self._raw_odom_position[0] = msg.pose.pose.position.x
            self._raw_odom_position[1] = msg.pose.pose.position.y
            
            # Apply offset to get corrected world position
            self._position[0] = self._raw_odom_position[0] + self._odom_offset[0]
            self._position[1] = self._raw_odom_position[1] + self._odom_offset[1]
            
            # Extract orientation (yaw from quaternion)
            q = msg.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self._orientation = math.atan2(siny_cosp, cosy_cosp)
            
            # Extract velocities
            self._linear_velocity = msg.twist.twist.linear.x
            self._angular_velocity = msg.twist.twist.angular.z
    
    def _wait_for_sensors(self, timeout: float = 10.0):
        """Wait for sensor data to become available."""
        start_time = time.time()
        while self._scan_data is None:
            if time.time() - start_time > timeout:
                self.node.get_logger().warn(
                    "Timeout waiting for sensor data. Make sure Gazebo is running!"
                )
                # Initialize with default values
                self._scan_data = np.full(360, self.config["lidar_max_range"])
                break
            time.sleep(0.1)
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation vector for the neural network.
        
        Returns:
            Normalized observation vector
        """
        with self._lock:
            # 1. Downsample LiDAR data
            if self._scan_data is not None:
                # Downsample from 360 to num_lidar_samples
                indices = np.linspace(0, len(self._scan_data) - 1, 
                                      self.config["num_lidar_samples"], dtype=int)
                lidar = self._scan_data[indices]
            else:
                lidar = np.full(self.config["num_lidar_samples"], 
                               self.config["lidar_max_range"])
            
            # Normalize LiDAR to [0, 1]
            lidar_normalized = lidar / self.config["lidar_max_range"]
            # Convert to [-1, 1] range
            lidar_normalized = 2.0 * lidar_normalized - 1.0
            
            # 2. Normalize position (assume maze is roughly 12x12 centered at origin)
            pos_normalized = self._position / 6.0  # [-1, 1] for 12m maze
            pos_normalized = np.clip(pos_normalized, -1.0, 1.0)
            
            # 3. Orientation as sin/cos (naturally in [-1, 1])
            orientation = np.array([
                np.sin(self._orientation),
                np.cos(self._orientation)
            ])
            
            # 4. Normalize velocities
            lin_vel_norm = self._linear_velocity / self.config["max_linear_velocity"]
            ang_vel_norm = self._angular_velocity / self.config["max_angular_velocity"]
            velocities = np.clip([lin_vel_norm, ang_vel_norm], -1.0, 1.0)
            
            # 5. Exploration information - percentage of cells visited
            exploration_progress = len(self._visited_cells) / max(self._total_possible_cells, 1)
            exploration_info = np.array([exploration_progress * 2.0 - 1.0])  # Normalize to [-1, 1]
            
            # Combine all observations
            observation = np.concatenate([
                lidar_normalized,
                pos_normalized,
                orientation,
                velocities,
                exploration_info
            ]).astype(np.float32)
            
            return observation
    
    def _compute_reward(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """
        Compute the reward for the current state.
        
        Returns:
            Tuple of (reward, terminated, truncated, info)
        """
        with self._lock:
            info = {}
            terminated = False
            truncated = False
            reward = 0.0
            
            # Track visited cells
            cells_visited = len(self._visited_cells)
            is_new_cell = self._mark_cell_visited()
            info["cells_visited"] = len(self._visited_cells)
            info["new_cell"] = is_new_cell
            
            # Check for collision
            if self._scan_data is not None:
                min_distance = np.min(self._scan_data)
                info["min_obstacle_distance"] = min_distance
                
                if min_distance < self.config["collision_threshold"]:
                    reward = self.config["reward_collision"]
                    terminated = True
                    info["collision"] = True
                    self.node.get_logger().info(
                        f"💥 ROSI collided! Min distance: {min_distance:.3f}m | Position: ({self._position[0]:.2f}, {self._position[1]:.2f}) | Cells visited: {len(self._visited_cells)}"
                    )
                    return reward, terminated, truncated, info
            
            # Check for timeout (episode ends but not a failure)
            if self._episode_step >= self.config["max_episode_steps"]:
                truncated = True
                info["timeout"] = True
                # Bonus for surviving the full episode
                reward = len(self._visited_cells) * 1.0  # Bonus based on cells explored
                self.node.get_logger().info(f"⏰ Episode timeout! Explored {len(self._visited_cells)} cells")
                return reward, terminated, truncated, info
            
            # Reward for visiting a new cell
            if is_new_cell:
                reward += self.config["reward_new_cell"]
            
            # Survival reward (small positive for each step without collision)
            reward += self.config["reward_survival"]
            
            # Reward for forward motion (encourage movement)
            if self._linear_velocity > 0.1:
                reward += self.config["reward_forward_motion"]
            
            # Rotation penalty (discourage spinning in place)
            rotation_penalty = abs(self._angular_velocity) * self.config["reward_rotation_penalty"]
            reward += rotation_penalty
            
            # Anti-spin penalty: detect and penalize continuous same-direction rotation
            current_rotation_sign = 0
            if self._angular_velocity > 0.3:  # Turning left
                current_rotation_sign = 1
            elif self._angular_velocity < -0.3:  # Turning right
                current_rotation_sign = -1
            
            if current_rotation_sign != 0 and current_rotation_sign == self._last_rotation_sign:
                self._consecutive_same_rotation += 1
            else:
                self._consecutive_same_rotation = 0
            self._last_rotation_sign = current_rotation_sign
            
            # Apply penalty if spinning in same direction for too long
            if self._consecutive_same_rotation >= self.config["spin_detection_threshold"]:
                spin_penalty = self.config["reward_same_direction_spin"]
                reward += spin_penalty
                info["spinning"] = True
            else:
                info["spinning"] = False
            
            info["success"] = False
            info["collision"] = False
            info["timeout"] = False
            
            return reward, terminated, truncated, info
    
    def _send_velocity_command(self, linear: float, angular: float):
        """Send velocity command to ROSI."""
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.cmd_vel_pub.publish(msg)
    
    def _stop_robot(self):
        """Stop ROSI immediately."""
        self._send_velocity_command(0.0, 0.0)
    
    def _teleport_robot(self, x: float, y: float, yaw: float = 0.0):
        """
        Teleport the robot to a specific position in Gazebo.
        
        Args:
            x: Target x coordinate
            y: Target y coordinate
            yaw: Target yaw angle (radians)
        """
        import subprocess
        
        # Use gz service to set the robot pose
        # Convert yaw to quaternion (only rotation around z-axis)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        
        # Get world name from config (supports maze_world_1, maze_world_2, maze_world_3)
        world_name = self.config.get("world_name", "maze_world_1")
        
        cmd = [
            'gz', 'service', '-s', f'/world/{world_name}/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req', f'name: "burger", position: {{x: {x}, y: {y}, z: 0.01}}, orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if result.returncode != 0:
                self.node.get_logger().warn(f"Teleport may have failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            self.node.get_logger().warn("Teleport command timed out")
        except Exception as e:
            self.node.get_logger().warn(f"Teleport error: {e}")
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: [linear_velocity_normalized, angular_velocity_normalized]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self._episode_step += 1
        
        # Scale action from [-1, 1] to actual velocity ranges
        linear_vel = float(action[0]) * self.config["max_linear_velocity"]
        angular_vel = float(action[1]) * self.config["max_angular_velocity"]
        
        # Clamp velocities
        linear_vel = np.clip(linear_vel, -self.config["max_linear_velocity"],
                            self.config["max_linear_velocity"])
        angular_vel = np.clip(angular_vel, -self.config["max_angular_velocity"],
                             self.config["max_angular_velocity"])
        
        # Send command
        self._send_velocity_command(linear_vel, angular_vel)
        
        # Wait for step duration
        time.sleep(self.config["step_duration"])
        
        # Get new observation
        observation = self._get_observation()
        
        # Compute reward
        reward, terminated, truncated, info = self._compute_reward()
        
        # Update cumulative reward
        self._cumulative_reward += reward
        info["cumulative_reward"] = self._cumulative_reward
        info["episode_step"] = self._episode_step
        
        # Stop robot if episode ended
        if terminated or truncated:
            self._stop_robot()
        
        return observation, reward, terminated, truncated, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Additional options (can include 'goal_position')
        
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        # Reset episode tracking
        self._episode_step = 0
        self._cumulative_reward = 0.0
        
        # Reset anti-spin tracking
        self._consecutive_same_rotation = 0
        self._last_rotation_sign = 0
        
        # Reset exploration tracking - clear visited cells for new episode
        self._visited_cells.clear()
        
        # Stop the robot first
        self._stop_robot()
        time.sleep(0.2)
        
        # Handle options
        options = options or {}
        
        # Check if robot is in a dangerous position (too close to wall)
        # If so, teleport back to start position
        with self._lock:
            if self._scan_data is not None:
                min_distance = np.min(self._scan_data)
                # If robot is dangerously close to wall, teleport to safe position
                if min_distance < 0.2:  # 20cm safety threshold
                    self.node.get_logger().info(
                        f"⚠️ Robot too close to wall ({min_distance:.3f}m), teleporting to start position"
                    )
                    self._teleport_robot(
                        self._start_position[0],
                        self._start_position[1],
                        0.0  # Reset orientation to 0
                    )
                    time.sleep(0.5)  # Wait for teleport to complete
                    # Update odometry offset after teleport
                    self._odom_offset[0] = self._start_position[0] - self._raw_odom_position[0]
                    self._odom_offset[1] = self._start_position[1] - self._raw_odom_position[1]
        
        # Wait briefly for sensors to stabilize
        time.sleep(0.2)
        
        # Mark initial cell as visited
        with self._lock:
            self._mark_cell_visited()
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            "robot_position": self._position.tolist(),
            "cells_visited": len(self._visited_cells),
        }
        
        self.node.get_logger().info(
            f"🔄 Episode reset. Exploration mode - starting at ({self._position[0]:.2f}, {self._position[1]:.2f})"
        )
        
        return observation, info
    
    def render(self):
        """Render the environment (Gazebo handles visualization)."""
        if self.render_mode == "human":
            # Gazebo provides visualization
            pass
    
    def close(self):
        """Clean up resources."""
        self._spinning = False
        self._stop_robot()
        
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        
        if self._owns_node:
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        
        self.node.get_logger().info("🛑 ROSI Maze Environment closed.")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of ROSI.
        
        Returns:
            Dictionary with position, orientation, velocities, and sensor data
        """
        with self._lock:
            return {
                "position": self._position.copy(),
                "orientation": self._orientation,
                "linear_velocity": self._linear_velocity,
                "angular_velocity": self._angular_velocity,
                "lidar": self._scan_data.copy() if self._scan_data is not None else None,
                "cells_visited": len(self._visited_cells),
                "exploration_progress": len(self._visited_cells) / max(self._total_possible_cells, 1),
            }


def make_rosi_env(config: Optional[Dict[str, Any]] = None) -> RosiMazeEnv:
    """
    Factory function to create a ROSI Maze Environment.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Configured RosiMazeEnv instance
    """
    return RosiMazeEnv(config=config, render_mode="human")
