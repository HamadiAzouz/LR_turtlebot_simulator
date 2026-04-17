# ROSI TurtleBot3 PPO Navigation

A reinforcement learning project for training a TurtleBot3 robot to navigate maze environments using Proximal Policy Optimization (PPO).

## Prerequisites

- **Ubuntu 24.04** (or WSL2 with Ubuntu 24.04)
- **ROS2 Jazzy**
- **Gazebo Harmonic** (gz-sim)
- **Python 3.12+**

## Installation

### 1. Install ROS2 Jazzy

Follow the official ROS2 installation guide: https://docs.ros.org/en/jazzy/Installation.html

```bash
# Add ROS2 repository
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Jazzy Desktop
sudo apt update
sudo apt install ros-jazzy-desktop -y
```

### 2. Install Gazebo Harmonic

```bash
sudo apt install ros-jazzy-ros-gz -y
```

### 3. Install Python Dependencies

```bash
pip install torch numpy gymnasium tensorboard
```

### 4. Build the Workspace

```bash
# Navigate to workspace
cd ~/path/to/lr_turtlebot_sim1

# Source ROS2
source /opt/ros/jazzy/setup.bash

# Build all packages
colcon build

# Source the workspace
source install/setup.bash
```

## Running the Code

### Step 1: Launch the Simulation

Open a terminal and launch Gazebo with your chosen maze:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Launch with maze_1 (12x12m, plus-shaped walls)
ros2 launch lr_turtlebot_sim turtlebot_in_maze.launch.py world:=maze_1.world x_pose:=-2.0 y_pose:=-0.5

# OR launch with maze_2 (14x14m, grid-like structure)
ros2 launch lr_turtlebot_sim turtlebot_in_maze.launch.py world:=maze_2.world x_pose:=-2.0 y_pose:=-0.5

# OR launch with maze_3 (16x10m, corridor maze)
ros2 launch lr_turtlebot_sim turtlebot_in_maze.launch.py world:=maze_3.world x_pose:=-6.0 y_pose:=-4.0
```

### Step 2: Train a Model

Open a **new terminal** and run the training script:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Train on maze_1
python3 install/lr_ppo/lib/lr_ppo/train_rosi.py --maze maze_1.world --episodes 1000

# Train on maze_2
python3 install/lr_ppo/lib/lr_ppo/train_rosi.py --maze maze_2.world --episodes 1000

# Train on maze_3
python3 install/lr_ppo/lib/lr_ppo/train_rosi.py --maze maze_3.world --episodes 1000
```

**Training Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--maze` | maze_1.world | Maze world file |
| `--episodes` | 1000 | Number of training episodes |
| `--max-steps` | 500 | Max steps per episode |
| `--lr` | 3e-4 | Learning rate |
| `--entropy-coef` | 0.01 | Entropy coefficient for exploration |
| `--batch-size` | 64 | Mini-batch size |
| `--rollout-length` | 2048 | Steps before PPO update |

Models are saved to: `./models/rosi_ppo_<maze>_<timestamp>/`

### Step 3: Monitor Training (Optional)

```bash
# View training logs with TensorBoard
tensorboard --logdir=./logs
```

Then open http://localhost:6006 in your browser.

## Testing Models

### Evaluate a Trained Model

Make sure the simulation is running (Step 1), then:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Evaluate on the same maze used for training
python3 install/lr_ppo/lib/lr_ppo/evaluate_rosi.py \
    --model models/rosi_ppo_maze_1_20260125_043630/final_model.pt \
    --episodes 10

# Evaluate with deterministic policy (no exploration noise)
python3 install/lr_ppo/lib/lr_ppo/evaluate_rosi.py \
    --model models/rosi_ppo_maze_1_20260125_043630/best_reward.pt \
    --episodes 20 \
    --deterministic
```

**Evaluation Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | Path to model checkpoint (.pt file) |
| `--episodes` | 10 | Number of evaluation episodes |
| `--deterministic` | False | Use deterministic policy |
| `--verbose` | True | Print per-episode results |

### Model Checkpoints

Each training run saves:
- `checkpoint_100.pt`, `checkpoint_200.pt`, ... - Periodic checkpoints
- `best_reward.pt` - Best average reward model
- `final_model.pt` - Final model after training
- `config.json` - Training configuration

## Project Structure

```
lr_turtlebot_sim1/
├── lr_turtlebot_sim/           # Main ROS2 package
│   ├── launch/                 # Launch files
│   │   ├── maze_1.launch.py
│   │   ├── maze_2.launch.py
│   │   ├── maze_3.launch.py
│   │   └── turtlebot_in_maze.launch.py
│   ├── worlds/                 # Gazebo world files
│   │   ├── maze_1.world
│   │   ├── maze_2.world
│   │   └── maze_3.world
│   ├── models/                 # Robot models (SDF/URDF)
│   └── urdf/                   # Robot descriptions
│
├── lr_ppo/                     # PPO training package
│   ├── lr_ppo/
│   │   ├── environment.py      # Gymnasium environment
│   │   ├── ppo_agent.py        # PPO algorithm
│   │   ├── networks.py         # Neural network architectures
│   │   └── utils.py            # Utilities (GAE, buffers)
│   └── scripts/
│       ├── train_rosi.py       # Training script
│       └── evaluate_rosi.py    # Evaluation script
│
├── models/                     # Saved trained models
├── logs/                       # TensorBoard logs
└── README.md
```

## Troubleshooting

### Simulation not starting
```bash
# Make sure Gazebo is installed
gz sim --version

# Check ROS2 is sourced
echo $ROS_DISTRO  # Should print "jazzy"
```

### Robot not moving during training
- Ensure the simulation is running BEFORE starting training
- Check that `/cmd_vel` topic exists: `ros2 topic list | grep cmd_vel`
- Verify LiDAR is publishing: `ros2 topic echo /scan --once`

### CUDA not available
```bash
# Check PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Training will automatically fall back to CPU if CUDA is unavailable
```

### Model not loading
- Ensure the model path is correct
- Check that the model was trained with the same network architecture

## Quick Start Example

```bash
# Terminal 1: Launch simulation
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch lr_turtlebot_sim turtlebot_in_maze.launch.py world:=maze_1.world x_pose:=-2.0 y_pose:=-0.5

# Terminal 2: Train model (wait for simulation to fully load)
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 install/lr_ppo/lib/lr_ppo/train_rosi.py --maze maze_1.world --episodes 500

# Terminal 2: After training, evaluate
python3 install/lr_ppo/lib/lr_ppo/evaluate_rosi.py --model models/rosi_ppo_maze_1_*/final_model.pt --episodes 10
```

## License

See [LICENSE](LICENSE) file.

