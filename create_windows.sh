#!/bin/bash

# Check if tmux is installed
if command -v tmux &> /dev/null; then
    # Create a new tmux session called "robot" and window 1
    tmux new-session -d -s robot -n window1

    # Enable mouse mode
    tmux set -g mouse on

    # Split the window into three equal vertical panes
    tmux split-window -h -t robot:0           # Now: 0.0 (left), 0.1 (right)
    tmux split-window -h -t robot:0.0         # Now: 0.0 (left), 0.2 (middle), 0.1 (right)

    # Even out the panes
    tmux select-layout -t robot:0 even-horizontal

    # Send commands to each pane
    # 1. Connect CAN adapter and bring up interface
    tmux send-keys -t robot:0.0 'conda activate yor-nero' C-m
    tmux send-keys -t robot:0.0 './extra/setup.sh' 
    
    # 2. Launch Robot Driver
    tmux send-keys -t robot:0.1 'conda activate yor-nero' C-m
    tmux send-keys -t robot:0.1 'python robot/yor.py' 

    # 3. Control the Robot (Teleop)
    tmux send-keys -t robot:0.2 'conda activate yor-nero' C-m
    tmux send-keys -t robot:0.2 'python robot/teleop/joystick.py' 

    # Attach to the session
    tmux attach-session -t robot
fi
