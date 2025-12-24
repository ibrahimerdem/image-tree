#!/bin/bash

# Training script for GAN generator
# Usage: ./run_gan_training.sh [num_gpus] [epochs] [checkpoint] [disable_initial_image]

set -e

NUM_GPUS=${1:-2}
NUM_EPOCHS=${2:-150}
RESUME_CHECKPOINT=${3:-""}
DISABLE_INITIAL_IMAGE=${4:-"false"}

echo "=========================================="
echo "GAN Generator - Distributed Training"
echo "=========================================="
echo "GPUs: $NUM_GPUS"
echo "Epochs: $NUM_EPOCHS"
[ "$DISABLE_INITIAL_IMAGE" = "true" ] && echo "Initial image conditioning: disabled" || echo "Initial image conditioning: enabled"
[ -n "$RESUME_CHECKPOINT" ] && echo "Resume: $RESUME_CHECKPOINT"
echo "=========================================="
echo ""

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

ARGS="--epochs $NUM_EPOCHS"
[ "$DISABLE_INITIAL_IMAGE" = "true" ] && ARGS="$ARGS --no_initial_image"
[ -n "$RESUME_CHECKPOINT" ] && [ -f "$RESUME_CHECKPOINT" ] && ARGS="$ARGS --resume $RESUME_CHECKPOINT"

if [ $NUM_GPUS -eq 1 ]; then
    python3 train_gan.py $ARGS
else
    torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --standalone train_gan.py $ARGS
fi

echo ""
echo "✓ GAN training completed!"
