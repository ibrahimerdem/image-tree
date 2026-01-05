#!/bin/bash

# Training script for GAN generator
# Usage: ./run_gan_training.sh [num_gpus] [epochs] [config_path]

set -e

NUM_GPUS=${1:-2}
NUM_EPOCHS=${2:-150}
CONFIG_PATH=${3:-"config.py"}

echo "=========================================="
echo "GAN Generator - Distributed Training"
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

ARGS="--epochs $NUM_EPOCHS --config $CONFIG_PATH"

if [ $NUM_GPUS -eq 1 ]; then
    python3 train_gan.py $ARGS
else
    torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --standalone train_gan.py $ARGS
fi

echo ""
echo "✓ GAN training completed!"
