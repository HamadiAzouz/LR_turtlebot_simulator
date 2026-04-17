"""
Launch file for ROSI Evaluation
================================

Launch the simulation for evaluating a trained ROSI model.

Usage:
    ros2 launch lr_ppo evaluate.launch.py maze:=maze_2.world
    
Then in another terminal:
    ros2 run lr_ppo evaluate_rosi.py --model path/to/model.pt --episodes 50
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Get package directories
    lr_turtlebot_sim_dir = get_package_share_directory('lr_turtlebot_sim')
    
    # Launch configurations
    maze = LaunchConfiguration('maze', default='maze_1.world')
    x_pose = LaunchConfiguration('x_pose', default='-2.0')
    y_pose = LaunchConfiguration('y_pose', default='-0.5')
    
    # Include turtlebot in maze launch
    turtlebot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(lr_turtlebot_sim_dir, 'launch', 'turtlebot_in_maze.launch.py')
        ]),
        launch_arguments={
            'world': maze,
            'x_pose': x_pose,
            'y_pose': y_pose,
            'use_sim_time': 'true',
        }.items()
    )
    
    # Create launch description
    ld = LaunchDescription()
    
    # Declare arguments
    ld.add_action(DeclareLaunchArgument(
        'maze',
        default_value='maze_1.world',
        description='Maze world to evaluate in'
    ))
    
    ld.add_action(DeclareLaunchArgument(
        'x_pose',
        default_value='-2.0',
        description='Initial robot X position'
    ))
    
    ld.add_action(DeclareLaunchArgument(
        'y_pose',
        default_value='-0.5',
        description='Initial robot Y position'
    ))
    
    # Add launch actions
    ld.add_action(turtlebot_launch)
    
    return ld
