from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=PathJoinSubstitution([
            get_package_share_directory('lr_turtlebot_sim'),
            'worlds',
            'maze_2.world'
        ]),
        description='Full path to the maze world SDF file'
    )

    world = LaunchConfiguration('world')

    # Start Gazebo (gz-sim) with the world
    gz_sim = ExecuteProcess(
        cmd=[
            'gz', 'sim', '-r', world
        ],
        output='screen'
    )

    # Start parameter bridge (optional, but recommended for simulation)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        gz_sim,
        bridge
    ])
