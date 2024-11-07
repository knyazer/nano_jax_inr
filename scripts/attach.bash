#!/bin/bash

sudo killall -9 tmux:\ server

sleep 1

# Start a new tmux session named "my_tpu_session"
tmux new-session -d -s my_tpu_session

# Connect to the first server (worker 0) in the first pane
tmux send-keys "gcloud compute tpus tpu-vm ssh pinecone --zone=us-central2-b --worker=0" C-m

# Split the window horizontally to create two side-by-side panes
tmux split-window -h
tmux send-keys "gcloud compute tpus tpu-vm ssh pinecone --zone=us-central2-b --worker=1" C-m

# Go back to the first pane and split it vertically to create a 2x2 grid
tmux select-pane -t 0
tmux split-window -v
tmux send-keys "gcloud compute tpus tpu-vm ssh pinecone --zone=us-central2-b --worker=2" C-m

# Select the second pane (created by horizontal split) and split it vertically
tmux select-pane -t 1
tmux split-window -v
tmux send-keys "gcloud compute tpus tpu-vm ssh pinecone --zone=us-central2-b --worker=3" C-m

# Now each pane will be split horizontally to form the 4x2 layout

# Select pane 0, split horizontally
tmux select-pane -t 0
tmux split-window -h
tmux send-keys "gcloud compute tpus tpu-vm ssh pinecone --zone=us-central2-b --worker=4" C-m

# Select pane 2, split horizontally
tmux select-pane -t 2
tmux split-window -h
tmux send-keys "gcloud compute tpus tpu-vm ssh pinecone --zone=us-central2-b --worker=5" C-m

# Select pane 4, split horizontally
tmux select-pane -t 4
tmux split-window -h
tmux send-keys "gcloud compute tpus tpu-vm ssh pinecone --zone=us-central2-b --worker=6" C-m

# Select pane 6, split horizontally
tmux select-pane -t 6
tmux split-window -h
tmux send-keys "gcloud compute tpus tpu-vm ssh pinecone --zone=us-central2-b --worker=7" C-m

# Adjust panes to be equally spaced (optional)
tmux select-layout tiled

# Attach to the session
tmux attach -t my_tpu_session
