"""
Launch file for ROSI PPO Training
==================================

This launch file starts the Gazebo simulation with a maze world
and prepares everything for PPO training.

Usage:
    ros2 launch lr_ppo train.launch.py maze:=maze_1.world
    
Then in another terminal:
    ros2 run lr_ppo train_rosi.py --episodes 1000
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    # Get package directories
    lr_turtlebot_sim_dir = get_package_share_directory('lr_turtlebot_sim')
    lr_ppo_dir = get_package_share_directory('lr_ppo')
    
    # Launch configurations
    maze = LaunchConfiguration('maze', default='maze_1.world')
    x_pose = LaunchConfiguration('x_pose', default='-2.0')
    y_pose = LaunchConfiguration('y_pose', default='-0.5')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    headless = LaunchConfiguration('headless', default='false')
    
    # Include the turtlebot in maze launch
    turtlebot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(lr_turtlebot_sim_dir, 'launch', 'turtlebot_in_maze.launch.py')
        ]),
        launch_arguments={
            'world': maze,
            'x_pose': x_pose,
            'y_pose': y_pose,
            'use_sim_time': use_sim_time,
        }.items()
    )
    
    # Create launch description
    ld = LaunchDescription()
    
    # Declare arguments
    ld.add_action(DeclareLaunchArgument(
        'maze',
        default_value='maze_1.world',
        description='Maze world file (maze_1.world, maze_2.world, maze_3.world)'
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
    
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    ))
    
    ld.add_action(DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo in headless mode (no GUI)'
    ))
    
    # Add actions
    ld.add_action(turtlebot_launch)
    
    return ld
