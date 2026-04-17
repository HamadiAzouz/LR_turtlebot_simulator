# API_REFERENCE.md
# lr_turtlebot_sim - API Reference

This document describes important **launch files** and **scripts** that students can use when working with the `lr_turtlebot_sim` package.

---

# Package Interfaces

## ROS 2 Nodes
This package provides:

| Node | Purpose |
|------|---------|
| `gazebo` | Simulation engine (external) |
| `robot_state_publisher` | Publishes TF from URDF |
| `teleop_node` | Keyboard teleop |

---

# Launch Files API

## 1. "spawn_turtlebot.launch.py"

### Description
Spawns the TurtleBot3 Burger into a running Gazebo using SDF and URDF and starts robot_state_publisher.

### Arguments
| Argument | Type | Default | Description |
|----------|-------|---------|-------------|
| `x` | float | `0.0` | Spawn X coordinate |
| `y` | float | `0.0` | Spawn Y coordinate |
| `yaw` | float | `0.0` | Initial yaw (rad) |
| `use_params` | bool | `false` | Load parameters from YAML |

### Example Call
```
ros2 launch lr_turtlebot_sim spawn_turtlebot.launch.py x:=1.2 y:=3.0 yaw:=1.57
```

---

## 2. "maze_x.launch.py"

### Description
Opens a specific maze world in Gazebo.

### Example Call
```
ros2 launch lr_turtlebot_sim maze_x.launch.py
```

---

## 3. "turtlebot_in_maze.launch.py"

### Description
Spawns the TurtleBot3 Burger into a specified Gazebo world using SDF and URDF and starts robot_state_publisher.

### Arguments
| Argument | Type | Default | Description |
|----------|-------|---------|-------------|
| `x` | float | `0.0` | Spawn X coordinate |
| `y` | float | `0.0` | Spawn Y coordinate |
| `use_sim_time` | book | `True` | Use simulation time |
| `world` | String | `maze_1.world` | Load parameters from YAML |

### Example Call
```
ros2 launch lr_turtlebot_sim turtlebot_in_maze.launch.py world:='maze_1.world' x:=1.2 y:=3.0 use_sim_time:=False
```


---

# Python Scripts API

## `teleop_keyboard.py`

### Description
Allows operating TurtleBot using keyboard by publishing velocity commands to `/cmd_vel`.

### Node:
`teleop_keyboard`

### Publishes:
| Topic | Type |
|--------|-----|
| `/cmd_vel` | geometry_msgs/Twist |

---

# Models API

## URDF (`urdf/turtlebot3_burger.urdf`)
Defines:
- Links and joints  
- LIDAR frame  
- Wheel geometry  
- TF frames  
- Visual & collision models  

## SDF (`models/turtlebot3_burger/model.sdf`)
Defines:
- Physics  
- Sensors  
- Plugins (`libgazebo_ros_diff_drive.so`, `libgazebo_ros_laser.so`)  
- Inertial properties  

---

# Gazebo Worlds API

Worlds should be SDF files.

There are three worlds:
```
maze_1.world
maze_2.world
maze_3.world
```

---

# Topics API

## Published
| Topic | Type | Description |
|-------|------|-------------|
| `/scan` | LaserScan | LIDAR |
| `/odom` | Odometry | Wheel odometry |
| `/tf` | TF | Transform tree |
| `/tf_static` | TF | Static transforms |

## Subscribed
| Topic | Type | Description |
|-------|------|-------------|
| `/cmd_vel` | Twist | Velocity commands |

---

# End of API Reference
