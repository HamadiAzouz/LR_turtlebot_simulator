# Neural Network Architecture for ROSI TurtleBot3

## Overview

This document explains the neural network architecture designed for the **ROSI (Robotic Obstacle-aware Spatial Intelligence)** TurtleBot3 robot trained using Proximal Policy Optimization (PPO) for autonomous maze exploration.

---

## 🧠 Architecture Evolution

### Basic Actor-Critic (v1)
The initial approach used a standard Multi-Layer Perceptron (MLP):

```
Input (31) → Dense(256) → Dense(256) → Action/Value
```

**Problems:**
- Treats all inputs equally (LiDAR readings vs. position)
- No spatial awareness of obstacles
- Limited learning capacity for complex environments

### Enhanced Actor-Critic (v2) ✅ **Current**
The enhanced architecture specifically designed for robot navigation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OBSERVATION (31 dimensions)                      │
├──────────────────────────────┬──────────────────────────────────────┤
│   LiDAR (24 samples)         │  State (7 values)                    │
│   [distance readings]        │  [x, y, cos(θ), sin(θ), v, ω, prog]  │
└───────────────┬──────────────┴────────────────┬─────────────────────┘
                │                               │
                ▼                               ▼
        ┌───────────────┐               ┌───────────────┐
        │ LiDAR Encoder │               │ State Encoder │
        │   (1D CNN)    │               │    (MLP)      │
        │               │               │               │
        │ Conv1d(32)    │               │ Dense(64)     │
        │ Conv1d(64)    │               │ LayerNorm     │
        │ Conv1d(64)    │               │ Dense(64)     │
        │ AvgPool       │               │ LayerNorm     │
        └───────┬───────┘               └───────┬───────┘
                │ (256 dim)                     │ (64 dim)
                └───────────┬───────────────────┘
                            ▼
                    ┌───────────────┐
                    │ Feature Fusion│
                    │  (320 → 256)  │
                    │  LayerNorm    │
                    └───────┬───────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
    ┌───────────────┐               ┌───────────────┐
    │    ACTOR      │               │    CRITIC     │
    │   (Policy)    │               │   (Value)     │
    │               │               │               │
    │ ResBlock x3   │               │ ResBlock x3   │
    │ LayerNorm     │               │ LayerNorm     │
    │ Dense(128)    │               │ Dense(128)    │
    │ Dense(2)      │               │ Dense(1)      │
    └───────┬───────┘               └───────┬───────┘
            │                               │
            ▼                               ▼
    ┌───────────────┐               ┌───────────────┐
    │ Action Mean   │               │    Value      │
    │ + State-dep   │               │   Estimate    │
    │   log_std     │               │               │
    └───────────────┘               └───────────────┘
```

---

## 🔧 Component Details

### 1. LiDAR Encoder (1D Convolutional Neural Network)

```python
class LidarEncoder(nn.Module):
    Conv1d(1 → 32, kernel=3)   # Detect local patterns
    Conv1d(32 → 64, kernel=3)  # Combine local features  
    Conv1d(64 → 64, kernel=3)  # Higher-level features
    AdaptiveAvgPool1d(4)       # Fixed output size
    LayerNorm                  # Normalize features
```

**Why CNN for LiDAR?**

| Reason | Explanation |
|--------|-------------|
| **Spatial Locality** | Adjacent LiDAR readings are spatially correlated. A wall appears as consecutive similar values. CNNs exploit this with local receptive fields. |
| **Translation Invariance** | The robot should recognize a wall whether it's on the left, right, or ahead. Convolutions share weights across positions. |
| **Feature Hierarchy** | Layer 1 detects edges (sudden distance changes). Layer 2 detects corners (combination of edges). Layer 3 detects openings/passages. |
| **Parameter Efficiency** | Instead of 24×256 = 6,144 parameters (fully connected), we use 3×32 + 3×32×64 + 3×64×64 ≈ 18,528 parameters but with much richer features. |

**Example: What the CNN learns**
```
Raw LiDAR: [3.5, 3.5, 3.5, 0.3, 0.3, 0.3, 3.5, 3.5, ...]
           └─────────┘    └─────────┘    └─────────┘
              open          wall           open
              
Conv Layer 1: Detects the "wall edge" at positions 3 and 6
Conv Layer 2: Combines edges into "wall segment" feature
Conv Layer 3: Recognizes "corridor opening" pattern
```

---

### 2. State Encoder (MLP)

```python
class StateEncoder(nn.Sequential):
    Dense(7 → 64)    # Expand state representation
    LayerNorm        # Normalize for stable training
    ReLU
    Dense(64 → 64)   # Refine features
    LayerNorm
    ReLU
```

**State Components:**
| Dimension | Content | Why Important |
|-----------|---------|---------------|
| 0-1 | Position (x, y) | Track where robot has been |
| 2-3 | Orientation (cos θ, sin θ) | Current heading (sin/cos avoids discontinuity at ±π) |
| 4-5 | Velocities (linear, angular) | Current motion state |
| 6 | Exploration progress | How much of maze explored (0-1) |

**Why separate from LiDAR?**
- LiDAR data is **high-dimensional and spatial** → needs CNN
- State data is **low-dimensional and semantic** → simple MLP is enough
- Mixing them early would dilute the spatial patterns

---

### 3. Residual Blocks

```python
class ResidualBlock(nn.Module):
    def forward(self, x):
        residual = x
        x = LayerNorm(x)
        x = ReLU(Dense(x))
        x = LayerNorm(x)
        x = Dense(x)
        return x + residual  # Skip connection!
```

**Why Residual Connections?**

| Problem | How Residuals Help |
|---------|-------------------|
| **Vanishing Gradients** | Skip connections allow gradients to flow directly backward |
| **Deep Network Training** | Can train 3+ block networks without degradation |
| **Feature Preservation** | Earlier features remain accessible to later layers |
| **Identity Mapping** | Network can easily learn identity if needed |

```
Without residual:  f(f(f(x))) - gradients must flow through all layers
With residual:     x + f(x + f(x + f(x))) - direct paths exist
```

---

### 4. State-Dependent Exploration

Traditional PPO uses a **fixed** log standard deviation:
```python
self.log_std = nn.Parameter(torch.tensor([-0.5, -0.5]))  # Fixed value
```

Our approach learns **when to explore more**:
```python
log_std = self.log_std_net(features) + self.log_std_bias
# Network learns: "uncertain situation → higher std → more exploration"
```

**Why State-Dependent?**
| Situation | Fixed std | State-dependent std |
|-----------|-----------|-------------------|
| Open area | Same exploration | Lower std (confident) |
| Near wall | Same exploration | Higher std (careful) |
| Stuck/spinning | Same exploration | Higher std (try something new) |
| New area | Same exploration | Higher std (explore) |

---

### 5. Layer Normalization

```python
x = LayerNorm(x)  # Normalizes across features (not batch)
```

**Why LayerNorm instead of BatchNorm?**
| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Normalization | Across batch | Across features |
| Batch size dependency | Yes (needs large batches) | No |
| RL suitability | Poor (small batches, varying distributions) | Good |
| Inference | Different behavior | Same as training |

---

## 📊 Network Statistics

| Component | Parameters | Purpose |
|-----------|------------|---------|
| LiDAR Encoder | ~18,500 | Spatial pattern recognition |
| State Encoder | ~8,500 | State feature extraction |
| Fusion Layer | ~82,000 | Combine modalities |
| Actor (3 ResBlocks + Head) | ~200,000 | Policy network |
| Critic (3 ResBlocks + Head) | ~200,000 | Value network |
| **Total** | **~509,000** | Full network |

Compare to basic MLP: ~135,000 parameters but much less expressive.

---

## 🎯 Why This Architecture for Maze Navigation?

### 1. **Multi-Modal Input Processing**
The robot receives fundamentally different types of information:
- **LiDAR**: 24 distance measurements (spatial/perceptual)
- **State**: Position, orientation, velocities (proprioceptive)

Processing them separately before fusion respects their different natures.

### 2. **Spatial Awareness**
The 1D CNN gives the robot "spatial intuition":
- Recognizes walls, corners, openings as patterns
- Understands "left is blocked, right is open"
- Detects corridors and dead ends

### 3. **Training Stability**
Layer normalization + residual connections enable:
- Stable training over 40,000+ episodes
- Consistent gradients throughout the network
- No exploding/vanishing activations

### 4. **Adaptive Exploration**
State-dependent std allows the robot to:
- Be confident in familiar situations
- Explore more when uncertain
- Break out of loops by trying new actions

### 5. **Separate Actor-Critic Processing**
After fusion, actor and critic have separate residual stacks:
- **Actor**: Focus on action selection (what to do)
- **Critic**: Focus on value estimation (how good is this state)
- They can specialize without interfering

---

## 🔬 Design Decisions Explained

### Q: Why 3 residual blocks?
**A:** Empirically, 2-4 blocks work well for this complexity. 3 gives enough depth without overfitting. Each block adds ~64K parameters and another layer of abstraction.

### Q: Why 256 hidden dimension?
**A:** Balance between capacity and efficiency. 256 neurons can represent complex decision boundaries while keeping training fast. Larger (512) showed diminishing returns.

### Q: Why tanh squashing for actions?
**A:** Robot actions (linear/angular velocity) are bounded. Tanh naturally bounds output to [-1, 1], which we then scale to actual velocity limits.

### Q: Why Gaussian distribution for actions?
**A:** Continuous action space needs a continuous distribution. Gaussian is:
- Differentiable (for backprop)
- Has analytical entropy (for exploration bonus)
- Supports reparameterization trick (low variance gradients)

### Q: Why orthogonal weight initialization?
**A:** Orthogonal matrices preserve gradient magnitude during backprop. This prevents vanishing/exploding gradients at initialization, giving training a stable start.

---

## 📈 Training Recommendations

| Hyperparameter | Recommended Value | Reason |
|----------------|-------------------|--------|
| Learning rate | 3e-4 | Standard for Adam with this architecture |
| Batch size | 2048 | Large enough for stable gradient estimates |
| GAE lambda | 0.95 | Balance bias/variance in advantage estimation |
| Clip ratio | 0.2 | Standard PPO clipping |
| Entropy coefficient | 0.01 | Encourage exploration without chaos |
| Value loss coefficient | 0.5 | Standard weight for critic loss |

---

## 🚀 Future Improvements

1. **Attention over LiDAR**: Self-attention to focus on relevant obstacles
2. **Recurrent layers (LSTM/GRU)**: Memory for partially observable states
3. **Auxiliary tasks**: Predict next LiDAR reading to improve representations
4. **Multi-scale CNN**: Different kernel sizes for different obstacle scales
5. **Curiosity-driven exploration**: Intrinsic reward for novel states

---

## 📚 References

- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **Residual Learning**: He et al., "Deep Residual Learning" (2015)
- **Layer Normalization**: Ba et al., "Layer Normalization" (2016)
- **Orthogonal Initialization**: Saxe et al., "Exact solutions to nonlinear dynamics" (2013)

---

## 🏗️ Code Location

All network implementations are in:
```
lr_ppo/lr_ppo/networks.py
```

Key classes:
- `LidarEncoder`: 1D CNN for laser scans
- `ResidualBlock`: Skip connection building block
- `EnhancedActorCritic`: Main network (recommended)
- `ActorCritic`: Basic MLP version (simpler, less capable)
