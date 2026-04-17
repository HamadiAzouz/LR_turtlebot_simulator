# TUTORIAL.md
# lr_turtlebot_sim - Tutorial

Welcome to the short tutorial for using the `lr_turtlebot_sim` package.  

---

# Part 1 - Setting Up the Workspace

Create a ROS 2 workspace:
```
mkdir -p ~/lr_ws/src
cd ~/lr_ws/src
```

Copy the package to the src of the package:
```
cp -r <path_to_package>/lr_turtlebot_sim .
```

Build:
```
cd ~/ros2_ws
colcon build
source install/setup.bash
```

---

# Part 2 - Launch the Robot in Gazebo

Start simulation:
```
ros2 launch lr_turtlebot_sim turtlebot_in_maze.launch.py
```

You should now see:

- TurtleBot3 Burger  
- LIDAR spinning  
- Physics working  

---

# Part 3 - Move the Robot

## Option A: Keyboard Teleop
```
ros2 run lr_turtlebot_sim teleop.launch.py
```

Controls:
- **W** - forward  
- **X** - backward  
- **A** - rotate left  
- **D** - rotate right  
- **S** - stop  

## Option B: Write your own velocity publisher

Write e.g. `scripts/send_cmd_vel.py`.

---

# Part 4 - Subscribe to Robot Topics

Check odometry:
```
ros2 topic echo /odom
```

Check lidar:
```
ros2 topic echo /scan
```

List active topics:
```
ros2 topic list
```

---




