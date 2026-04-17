# 🤖 lr_ppo - PPO Training Package for ROSI the TurtleBot3

Train ROSI the TurtleBot3 to navigate through mazes using Proximal Policy Optimization (PPO)!

## 📋 Overview

This package provides a complete reinforcement learning framework for training ROSI to navigate mazes autonomously. ROSI learns through trial and error, using:

- **LiDAR** - 360° laser distance sensor for obstacle detection
- **Odometry** - Position and orientation tracking
- **PPO Brain** - A neural network that learns optimal navigation policies

## 🚀 Quick Start

### Prerequisites

1. **ROS2 Humble** (or compatible version)
2. **Gazebo Harmonic**
3. **Python packages**:
   ```bash
   pip install torch numpy gymnasium matplotlib tensorboard
   ```

### Installation

```bash
# Navigate to your ROS2 workspace
cd ~/ros2_ws/src

# Clone or copy lr_ppo and lr_turtlebot_sim packages

# Build
cd ~/ros2_ws
colcon build --packages-select lr_turtlebot_sim lr_ppo
source install/setup.bash
```

### Training ROSI

**Step 1: Launch the simulation**
```bash
# Terminal 1: Start Gazebo with maze
ros2 launch lr_ppo train.launch.py maze:=maze_1.world
```

**Step 2: Run training**
```bash
# Terminal 2: Start PPO training
ros2 run lr_ppo train_rosi.py --episodes 2000 --goal 2.0 2.0
```

### Evaluating a Trained Model

```bash
# Terminal 1: Launch simulation
ros2 launch lr_ppo evaluate.launch.py maze:=maze_2.world

# Terminal 2: Run evaluation
ros2 run lr_ppo evaluate_rosi.py --model models/best_success.pt --episodes 50
```

## 📁 Package Structure

```
lr_ppo/
├── lr_ppo/                    # Python module
│   ├── __init__.py
│   ├── environment.py         # Gymnasium environment for ROSI
│   ├── networks.py            # Actor-Critic neural networks
│   ├── ppo_agent.py           # PPO algorithm implementation
│   └── utils.py               # Helper functions
├── scripts/
│   ├── train_rosi.py          # Training script
│   ├── evaluate_rosi.py       # Evaluation script
│   └── test_environment.py    # Environment testing
├── launch/
│   ├── train.launch.py        # Training launch file
│   └── evaluate.launch.py     # Evaluation launch file
├── config/
│   └── training_config.yaml   # Default hyperparameters
├── models/                    # Saved model checkpoints
├── CMakeLists.txt
├── package.xml
└── README.md
```

## 🧠 How It Works

### Observation Vector

ROSI's neural network receives a normalized observation vector containing:

| Component | Size | Description |
|-----------|------|-------------|
| LiDAR | 24 | Downsampled 360° scan, normalized to [-1, 1] |
| Position | 2 | Robot (x, y) normalized by maze size |
| Orientation | 2 | sin(yaw), cos(yaw) for continuous representation |
| Velocities | 2 | Linear and angular velocities, normalized |
| Goal Info | 3 | Distance to goal, angle to goal (sin/cos) |

**Total: 33 dimensions** (with default 24 LiDAR samples)

### Action Space

Continuous actions:
- **Linear velocity**: [-1, 1] → scaled to [-0.22, 0.22] m/s
- **Angular velocity**: [-1, 1] → scaled to [-2.84, 2.84] rad/s

### Reward Function

```
R = R_goal + R_collision + R_progress + R_time + R_rotation

Where:
- R_goal = +100 if goal reached
- R_collision = -100 if collision detected
- R_progress = 10 × (distance_decreased)
- R_time = -0.1 per timestep
- R_rotation = -0.05 × |angular_velocity|
```

## 🎛️ Configuration

### Training Arguments

```bash
ros2 run lr_ppo train_rosi.py \
    --episodes 2000 \           # Total training episodes
    --goal 2.0 2.0 \            # Goal position (x, y)
    --lr 0.0003 \               # Learning rate
    --gamma 0.99 \              # Discount factor
    --clip-ratio 0.2 \          # PPO clip ratio
    --hidden-sizes 256 256 \    # Network architecture
    --rollout-length 2048 \     # Steps before update
    --save-interval 100         # Checkpoint frequency
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 3e-4 | Adam optimizer learning rate |
| `gamma` | 0.99 | Discount factor for future rewards |
| `gae_lambda` | 0.95 | GAE parameter for advantage estimation |
| `clip_ratio` | 0.2 | PPO clipping parameter |
| `n_epochs` | 10 | PPO update epochs |
| `batch_size` | 64 | Mini-batch size |
| `entropy_coef` | 0.01 | Exploration bonus coefficient |

## 📊 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/
```

View metrics:
- Episode rewards and lengths
- Success/collision rates
- Policy and value losses
- Entropy and KL divergence

### Training Output

```
📈 Episode 100/2000 🎉 SUCCESS
   Reward: 87.32 (avg: 42.15)
   Length: 156 (avg: 289.4)
   Success Rate: 35.0%
   Total Timesteps: 28942

💾 Agent saved to models/rosi_ppo_maze_1_xxx/checkpoint_100.pt
🏆 New best success rate model saved!
```

## 🏆 Tips for Success

### 1. Start Simple
Begin with Maze 1 and simple goals. ROSI needs to learn basic navigation before tackling complex mazes.

### 2. Reward Shaping
The default reward function works well, but you can experiment:
- Increase `progress_scale` for faster initial learning
- Adjust collision penalty if ROSI is too cautious
- Add waypoint rewards for complex mazes

### 3. Hyperparameter Tuning
- **If training is unstable**: Decrease learning rate, increase batch size
- **If learning is slow**: Increase learning rate, decrease entropy coefficient
- **If ROSI doesn't explore**: Increase initial entropy coefficient

### 4. Curriculum Learning
Train on easy goals first, then gradually increase difficulty:
```bash
# Stage 1: Close goal
ros2 run lr_ppo train_rosi.py --goal 1.0 0.0 --episodes 500

# Stage 2: Medium goal (load previous model)
ros2 run lr_ppo train_rosi.py --goal 2.0 2.0 --checkpoint models/.../checkpoint.pt

# Stage 3: Far goal
ros2 run lr_ppo train_rosi.py --goal 4.0 4.0 --checkpoint models/.../checkpoint.pt
```

## 🐛 Troubleshooting

### "Timeout waiting for sensor data"
- Ensure Gazebo is running and robot is spawned
- Check ROS2 topics: `ros2 topic list`
- Verify bridge: `ros2 topic echo /scan`

### Training not improving
- Check reward values during training
- Verify goal is reachable in the maze
- Try resetting with different initial positions

### High collision rate
- Increase `collision_threshold`
- Add more penalty for getting close to obstacles
- Verify LiDAR data is being received

## 📚 API Reference

### RosiMazeEnv

```python
from lr_ppo.environment import RosiMazeEnv

# Create environment
env = RosiMazeEnv(config={
    "goal_position": (2.0, 2.0),
    "max_episode_steps": 500,
})

# Standard Gymnasium interface
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)

# Additional methods
state = env.get_state()  # Get full robot state
env.set_goal(x, y)       # Change goal position
```

### PPOAgent

```python
from lr_ppo.ppo_agent import PPOAgent

# Create agent
agent = PPOAgent(obs_dim=33, action_dim=2, config={...})

# Select action
action, log_prob, value = agent.select_action(observation)

# Update policy
stats = agent.update()

# Save/Load
agent.save("path/to/model.pt")
agent.load("path/to/model.pt")
```

## 🎓 Learning Resources

- [PPO Paper](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization Algorithms
- [Spinning Up RL](https://spinningup.openai.com/) - OpenAI's RL introduction
- [TurtleBot3 Manual](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)
- [ROS2 Documentation](https://docs.ros.org/)

## 📝 License

Apache 2.0 - See LICENSE file

---

🤖 **Good luck training ROSI! May your rewards be high and your collisions few!** 🎮
