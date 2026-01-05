#!/bin/bash

# Training Script for Conditional Image Generator
# Usage: ./run_training.sh [num_gpus] [epochs] [config_path]

set -e

NUM_GPUS=${1:-2}
NUM_EPOCHS=${2:-150}
CONFIG_PATH=${3:-"config.py"}

echo "=========================================="
echo "Image Generator - Distributed Training"
echo "=========================================="
echo "GPUs: $NUM_GPUS"
echo "Epochs: $NUM_EPOCHS"
echo "Config: $CONFIG_PATH"
echo "=========================================="
echo ""

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config file not found: $CONFIG_PATH"
    exit 1
fi

# Validate data
if [ ! -d "data/initial" ] || [ ! -d "data/target" ]; then
    echo "❌ Data directories not found!"
    exit 1
fi

if [ ! -f "data/training_features.csv" ] || [ ! -f "data/validation_features.csv" ]; then
    echo "❌ CSV files not found!"
    exit 1
fi

echo "✓ Data validated"
mkdir -p checkpoints logs samples test_results

[ -d ".venv" ] && source .venv/bin/activate

# Build arguments
ARGS="--epochs $NUM_EPOCHS --config $CONFIG_PATH"

# Run training
if [ $NUM_GPUS -eq 1 ]; then
    python3 train.py $ARGS
else
    torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --standalone train.py $ARGS
fi

echo ""
echo "✓ Training completed!"
