#!/bin/bash
# Sequential Training Script for ROSI
# =====================================
# Trains the robot on maze_2, then maze_3
# Total: 40,000 episodes (20,000 per maze)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
EPISODES_PER_MAZE=1000
WORKSPACE_DIR="/mnt/c/users/dell/Desktop/lr_turtlebot_sim1"
START_X="-2.0"
START_Y="-0.5"

# Function to print header
print_header() {
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}\n"
}

# Function to kill all simulation processes
kill_simulation() {
    echo -e "${YELLOW}🛑 Stopping all simulation processes...${NC}"
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f "ruby.*gz" 2>/dev/null || true
    pkill -9 -f "parameter_bridge" 2>/dev/null || true
    pkill -9 -f "robot_state_pub" 2>/dev/null || true
    pkill -9 -f "train_rosi" 2>/dev/null || true
    pkill -9 -f "rosi_maze_env" 2>/dev/null || true
    sleep 3
    echo -e "${GREEN}✓ All processes stopped${NC}"
}

# Function to launch Gazebo
launch_gazebo() {
    local maze=$1
    echo -e "${CYAN}🌍 Launching Gazebo with ${maze}...${NC}"
    
    cd "$WORKSPACE_DIR"
    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
    
    # Set Gazebo resource paths
    export GZ_SIM_RESOURCE_PATH="$WORKSPACE_DIR/install/lr_turtlebot_sim/share/lr_turtlebot_sim/models:$WORKSPACE_DIR/lr_turtlebot_sim/models:${GZ_SIM_RESOURCE_PATH:-}"
    
    # Launch in background
    ros2 launch lr_turtlebot_sim turtlebot_in_maze.launch.py \
        world:="${maze}.world" \
        x_pose:="$START_X" \
        y_pose:="$START_Y" &
    
    GAZEBO_PID=$!
    echo -e "${GREEN}✓ Gazebo launched (PID: $GAZEBO_PID)${NC}"
    
    # Wait for Gazebo to initialize
    echo -e "${YELLOW}⏳ Waiting for Gazebo to initialize (30 seconds)...${NC}"
    sleep 30
    echo -e "${GREEN}✓ Gazebo ready${NC}"
}

# Function to run training
run_training() {
    local maze=$1
    local episodes=$2
    local resume_model=$3
    
    echo -e "${CYAN}🤖 Starting training on ${maze} for ${episodes} episodes...${NC}"
    
    cd "$WORKSPACE_DIR"
    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
    
    # Build training command
    local cmd="python3 install/lr_ppo/lib/lr_ppo/train_rosi.py --maze ${maze} --episodes ${episodes} --verbose"
    
    # Add resume option if model provided
    if [ -n "$resume_model" ] && [ -f "$resume_model" ]; then
        echo -e "${YELLOW}📂 Resuming from: ${resume_model}${NC}"
        cmd="$cmd --resume $resume_model"
    fi
    
    # Run training
    echo -e "${GREEN}▶ Running: $cmd${NC}\n"
    eval $cmd
    
    echo -e "\n${GREEN}✓ Training on ${maze} completed!${NC}"
}

# Function to find latest model
find_latest_model() {
    local maze=$1
    local latest=$(ls -td "$WORKSPACE_DIR/models/rosi_ppo_${maze}_"*/ 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        local model_file="${latest}final_model.pt"
        if [ -f "$model_file" ]; then
            echo "$model_file"
        fi
    fi
}

# ============================================
# MAIN SCRIPT
# ============================================

print_header "🚀 ROSI Sequential Training Script"

echo -e "${CYAN}Configuration:${NC}"
echo -e "  • Episodes per maze: ${EPISODES_PER_MAZE}"
echo -e "  • Total episodes: $((EPISODES_PER_MAZE * 2))"
echo -e "  • Mazes: maze_2 → maze_3"
echo -e "  • Start position: (${START_X}, ${START_Y})"
echo ""

# Record start time
START_TIME=$(date +%s)

# ============================================
# PHASE 1: Train on Maze 2
# ============================================
print_header "📍 PHASE 1: Training on Maze 2"

kill_simulation
launch_gazebo "maze_2"
run_training "maze_2" $EPISODES_PER_MAZE ""

# Get the model from maze_2 training
MAZE2_MODEL=$(find_latest_model "maze_2")
echo -e "${GREEN}📦 Maze 2 model saved: ${MAZE2_MODEL}${NC}"

kill_simulation

# ============================================
# PHASE 2: Train on Maze 3
# ============================================
print_header "📍 PHASE 2: Training on Maze 3"

# Small break between phases
echo -e "${YELLOW}⏳ Waiting 10 seconds before starting maze_3...${NC}"
sleep 10

launch_gazebo "maze_3"

# Option: Continue from maze_2 model or start fresh
# Uncomment the next line to continue from maze_2 model:
# run_training "maze_3" $EPISODES_PER_MAZE "$MAZE2_MODEL"

# Start fresh training on maze_3:
run_training "maze_3" $EPISODES_PER_MAZE ""

MAZE3_MODEL=$(find_latest_model "maze_3")
echo -e "${GREEN}📦 Maze 3 model saved: ${MAZE3_MODEL}${NC}"

kill_simulation

# ============================================
# SUMMARY
# ============================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

print_header "🏁 Training Complete!"

echo -e "${GREEN}Summary:${NC}"
echo -e "  • Total episodes: $((EPISODES_PER_MAZE * 2))"
echo -e "  • Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo -e ""
echo -e "${CYAN}Models saved:${NC}"
echo -e "  • Maze 2: ${MAZE2_MODEL:-'Not found'}"
echo -e "  • Maze 3: ${MAZE3_MODEL:-'Not found'}"
echo -e ""
echo -e "${YELLOW}To evaluate the models:${NC}"
echo -e "  # For maze_2 model:"
echo -e "  python3 install/lr_ppo/lib/lr_ppo/evaluate_rosi.py --model ${MAZE2_MODEL:-'<maze2_model>'} --episodes 10"
echo -e ""
echo -e "  # For maze_3 model:"
echo -e "  python3 install/lr_ppo/lib/lr_ppo/evaluate_rosi.py --model ${MAZE3_MODEL:-'<maze3_model>'} --episodes 10"
echo -e ""
echo -e "${GREEN}🎉 All training completed successfully!${NC}"
