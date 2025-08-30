#!/bin/bash
# Launch Energy Transformer training with Accelerate

# Single GPU training
echo "Launching single-GPU training..."
accelerate launch train.py

# Multi-GPU training (uncomment for multi-GPU)
# echo "Launching multi-GPU training..."
# accelerate launch --multi_gpu train.py

# Distributed training with specific config (uncomment for custom setup)
# echo "Launching distributed training..."
# accelerate launch --num_processes 4 --mixed_precision fp16 train.py