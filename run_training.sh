#!/bin/bash

# Training script for Image Generator
# Usage: ./run_training.sh [num_gpus] [epochs] [resume_checkpoint]

set -e

# Configuration
NUM_GPUS=${1:-2}  # Default: 2 GPUs
NUM_EPOCHS=${2:-200}  # Default: 200 epochs
RESUME_CHECKPOINT=${3:-""}  # Optional: checkpoint path to resume from
USE_VAE=${4:-"true"}  # Default: use VAE (true/false)

echo "=========================================="
echo "Image Generator Training"
echo "=========================================="
echo "GPUs: $NUM_GPUS"
echo "Epochs: $NUM_EPOCHS"
echo "Input Size: 128x128"
echo "Output Size: 512x512"
echo "VAE Enabled: $USE_VAE"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resume from: $RESUME_CHECKPOINT"
fi
echo "=========================================="
echo ""

# Check if data exists
if [ ! -d "data/initial" ] || [ ! -d "data/target" ]; then
    echo "Error: Data directories not found!"
    echo "Expected: data/initial/ and data/target/"
    exit 1
fi

if [ ! -f "data/training_features.csv" ] || [ ! -f "data/validation_features.csv" ]; then
    echo "Error: CSV files not found!"
    echo "Expected: data/training_features.csv and data/validation_features.csv"
    exit 1
fi

echo "Data directories found"
echo " - Initial images: $(ls data/initial/ | wc -l) files"
echo " - Target images: $(ls data/target/ | wc -l) files"
echo " - Training samples: $(($(wc -l < data/training_features.csv) - 1))"
echo " - Validation samples: $(($(wc -l < data/validation_features.csv) - 1))"
echo ""

# Create output directories
mkdir -p checkpoints logs samples test_results

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "✓ Activating virtual environment"
    source .venv/bin/activate
fi

# Build resume argument if checkpoint provided
RESUME_ARG=""
if [ -n "$RESUME_CHECKPOINT" ]; then
    if [ ! -f "$RESUME_CHECKPOINT" ]; then
        echo "❌ Error: Checkpoint file not found: $RESUME_CHECKPOINT"
        exit 1
    fi
    RESUME_ARG="--resume $RESUME_CHECKPOINT"
    echo "✓ Found checkpoint: $RESUME_CHECKPOINT"
    echo ""
fi

# Build VAE argument
VAE_ARG=""
if [ "$USE_VAE" = "true" ] || [ "$USE_VAE" = "1" ] || [ "$USE_VAE" = "yes" ]; then
    VAE_ARG="--use_vae"
    echo "✓ VAE enabled"
else
    echo "✓ VAE disabled"
fi
echo ""

# Run training
# Note: lr, batch_size, num_workers are now taken from config.py
# Only override if you need different values
if [ $NUM_GPUS -eq 1 ]; then
    echo "Starting training on 1 GPU..."
    python3 train.py \
        --world_size 1 \
        --epochs $NUM_EPOCHS \
        $VAE_ARG \
        $RESUME_ARG
else
    echo "Starting distributed training on $NUM_GPUS GPUs with torchrun..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --standalone \
        train.py \
        --world_size $NUM_GPUS \
        --epochs $NUM_EPOCHS \
        $VAE_ARG \
        $RESUME_ARG
fi

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Checkpoints saved in: ./checkpoints/"
echo "Training logs in: ./logs/"
echo "Sample images in: ./samples/"
