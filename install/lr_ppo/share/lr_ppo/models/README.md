# ROSI PPO Training Package - Models Directory

This directory stores trained model checkpoints.

## Directory Structure

After training, you'll find:

```
models/
├── rosi_ppo_maze_1_TIMESTAMP/
│   ├── config.json           # Training configuration
│   ├── checkpoint_100.pt     # Periodic checkpoints
│   ├── checkpoint_200.pt
│   ├── ...
│   ├── best_reward.pt        # Best average reward model
│   ├── best_success.pt       # Best success rate model
│   └── final_model.pt        # Final training model
```

## Loading a Model

```python
from lr_ppo.ppo_agent import PPOAgent

# Load a trained agent
agent = PPOAgent.load_from_checkpoint("models/rosi_ppo_maze_1_xxx/best_success.pt")

# Use for evaluation
action, log_prob, value = agent.select_action(observation, deterministic=True)
```

## Pre-trained Models

Pre-trained models (if provided) will be stored here:

- `pretrained_maze_1.pt` - Trained on Maze 1
- `pretrained_maze_2.pt` - Trained on Maze 2
- `pretrained_maze_3.pt` - Trained on Maze 3
