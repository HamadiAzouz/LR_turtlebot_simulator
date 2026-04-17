#!/bin/bash
# ROSI Training Runner Script
# ============================
# This script sets up the environment and launches the simulation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}🤖 ROSI TurtleBot3 Training Launcher${NC}"
echo -e "${BLUE}======================================${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$SCRIPT_DIR"

echo -e "\n${YELLOW}📁 Workspace: $WORKSPACE_DIR${NC}"

# Source ROS2
echo -e "\n${GREEN}🔧 Sourcing ROS2 Jazzy...${NC}"
source /opt/ros/jazzy/setup.bash

# Build if needed
if [ ! -d "$WORKSPACE_DIR/install" ]; then
    echo -e "\n${YELLOW}🔨 Building packages...${NC}"
    cd "$WORKSPACE_DIR"
    colcon build
fi

# Source the workspace
echo -e "${GREEN}🔧 Sourcing workspace...${NC}"
source "$WORKSPACE_DIR/install/setup.bash"

# Set Gazebo resource paths
MODELS_PATH="$WORKSPACE_DIR/install/lr_turtlebot_sim/share/lr_turtlebot_sim/models"
SRC_MODELS_PATH="$WORKSPACE_DIR/lr_turtlebot_sim/models"

echo -e "${GREEN}📂 Setting GZ_SIM_RESOURCE_PATH...${NC}"
export GZ_SIM_RESOURCE_PATH="${MODELS_PATH}:${SRC_MODELS_PATH}:${GZ_SIM_RESOURCE_PATH:-}"
echo "   $GZ_SIM_RESOURCE_PATH"

# Parse arguments
MAZE="${1:-maze_1.world}"
X_POSE="${2:--2.0}"
Y_POSE="${3:--0.5}"

echo -e "\n${BLUE}🌍 Launching simulation...${NC}"
echo "   Maze: $MAZE"
echo "   Start position: ($X_POSE, $Y_POSE)"
echo ""

# Launch the simulation
ros2 launch lr_turtlebot_sim turtlebot_in_maze.launch.py \
    world:="$MAZE" \
    x_pose:="$X_POSE" \
    y_pose:="$Y_POSE"
