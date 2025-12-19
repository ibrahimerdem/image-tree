# Conditional Image Generator (128→256 Upscaling)

A PyTorch-based conditional image generator that takes 128x128 input images with 10 condition features and generates 256x256 output images (2x upscaling). Optimized for training on 2x36GB GPUs.

## Model Architecture

- **Input**: 128×128 RGB images + 10 condition features
- **Output**: 256×256 RGB images (2x upscaling)
- **Architecture**: Encoder-Decoder with residual blocks
- **Optimization**: Multi-GPU training with DDP, mixed precision (AMP), gradient checkpointing

### Network Structure

```
Encoder (128→16):
- 128×128 → 64×64 (Conv, 64 channels)
- 64×64 → 32×32 (Conv, 128 channels)
- 32×32 → 16×16 (Conv, 256 channels)

Condition Encoder (10 features):
- 10 → 64 → 128 → 256

Decoder (16→256):
- 16×16 → 32×32 (ConvTranspose, 256 channels)
- 32×32 → 64×64 (ConvTranspose, 128 channels)
- 64×64 → 128×128 (ConvTranspose, 64 channels)
- 128×128 → 256×256 (ConvTranspose, 32 channels)
```

## Requirements

```bash
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=9.5.0
scikit-image>=0.20.0
tqdm>=4.65.0
```

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (with CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy pandas Pillow scikit-image tqdm
```

## Dataset Structure

```
data/
├── initial/              # Input images (128×128 recommended)
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── target/               # Target images (256×256 recommended)
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── training_features.csv      # Training set features
├── validation_features.csv    # Validation set features
└── test_features.csv         # Test set features
```

### CSV Format

Each CSV file should contain:
```csv
initial_filename,target_filename,feature_1,feature_2,...,feature_10
img_001.jpg,img_001.jpg,0.5,1.2,0.3,0.8,1.5,0.2,0.9,1.1,0.6,0.4
img_002.jpg,img_002.jpg,0.3,0.9,0.7,1.0,0.5,0.8,0.2,1.3,0.4,0.6
...
```

Or if input and output have different filenames:
```csv
initial_filename,target_filename,feature_1,feature_2,...,feature_10
input_001.jpg,output_001.png,0.5,1.2,0.3,0.8,1.5,0.2,0.9,1.1,0.6,0.4
input_002.jpg,output_002.png,0.3,0.9,0.7,1.0,0.5,0.8,0.2,1.3,0.4,0.6
...
```

## Training

### Single GPU Training
```bash
python3 train.py \
    --data_dir ./data \
    --input_size 128 \
    --output_size 256 \
    --batch_size 32 \
    --epochs 200 \
    --lr 2e-4 \
    --world_size 1
```

### Multi-GPU Training (2 GPUs)
```bash
python3 train.py \
    --data_dir ./data \
    --input_size 128 \
    --output_size 256 \
    --batch_size 32 \
    --epochs 200 \
    --lr 2e-4 \
    --world_size 2
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `./data` | Root data directory |
| `--input_size` | `128` | Input image size |
| `--output_size` | `256` | Output image size |
| `--num_conditions` | `10` | Number of condition features |
| `--batch_size` | `32` | Batch size per GPU |
| `--epochs` | `200` | Number of training epochs |
| `--lr` | `2e-4` | Learning rate |
| `--world_size` | `2` | Number of GPUs |
| `--num_workers` | `8` | DataLoader workers |
| `--checkpoint_dir` | `./checkpoints` | Checkpoint save directory |
| `--log_dir` | `./logs` | Log directory |
| `--resume` | `None` | Path to checkpoint to resume |

## Training Configuration (config.py)

The model is configured for optimal performance on 2×36GB GPUs:

- **Batch Size**: 32 per GPU (effective: 64)
- **Mixed Precision**: Enabled (AMP)
- **Gradient Checkpointing**: Enabled
- **Memory Format**: Channels-last
- **Learning Rate**: 2e-4 with cosine annealing
- **Loss**: MSE + L1 + Perceptual (VGG-based)

## Model Testing

Test the model architecture:
```bash
python3 model.py
```

Expected output:
```
Testing Image Generator Model
============================================================

📊 Model Architecture:
  Input size:  128x128
  Output size: 256x256
  Upscaling factor: 2.0x
  Conditions: 10

🔢 Test Results:
  Input shape:  torch.Size([4, 3, 128, 128])
  Output shape: torch.Size([4, 3, 256, 256])
  Total parameters: ~X,XXX,XXX
  Model memory: XX.XX MB

✅ Model tests passed!
```

## Outputs

During training, the following are generated:

1. **Checkpoints** (`./checkpoints/`):
   - `checkpoint_epoch_XXX.pth` - Model checkpoints
   - Best model based on validation loss

2. **Logs** (`./logs/`):
   - `training_metrics.csv` - Epoch-wise metrics
   - Training and validation losses, PSNR, SSIM

3. **Samples** (`./samples/`):
   - Visual comparisons of input, generated, and target images
   - Saved every epoch

## Memory Optimization Features

1. **Gradient Checkpointing**: Reduces memory by recomputing intermediate activations
2. **Mixed Precision (AMP)**: Uses FP16 for faster training and reduced memory
3. **Channels-Last Memory Format**: Optimizes tensor layout for better GPU performance
4. **Distributed Data Parallel (DDP)**: Efficient multi-GPU training
5. **Synchronized Batch Normalization**: Consistent stats across GPUs

## Performance Metrics

The model tracks:
- **MSE Loss**: Mean squared error
- **L1 Loss**: Mean absolute error
- **Perceptual Loss**: VGG-based perceptual similarity
- **PSNR**: Peak signal-to-noise ratio
- **SSIM**: Structural similarity index

## Advanced Usage

### Resume Training
```bash
python3 train.py \
    --resume ./checkpoints/checkpoint_epoch_100.pth \
    --world_size 2
```

### Adjust Learning Rate
```bash
python3 train.py \
    --lr 1e-4 \
    --world_size 2
```

### Increase Batch Size (if memory allows)
```bash
python3 train.py \
    --batch_size 48 \
    --world_size 2
```

## Troubleshooting

### Out of Memory (OOM)
1. Reduce `BATCH_SIZE` in `config.py`
2. Enable gradient checkpointing (already enabled)
3. Reduce model channels in `ENCODER_CHANNELS`/`DECODER_CHANNELS`

### Training is Too Slow
1. Increase `NUM_WORKERS` for faster data loading
2. Ensure `USE_AMP=True` for mixed precision
3. Use `CHANNELS_LAST=True` for better GPU utilization

### Poor Image Quality
1. Increase `PERCEPTUAL_WEIGHT` in loss
2. Train for more epochs
3. Adjust learning rate schedule
4. Check data normalization

## File Structure

```
├── config.py           # Training configuration
├── model.py           # Model architecture
├── dataset.py         # Dataset and DataLoader
├── train.py           # Training script
├── utils.py           # Utility functions
├── data/              # Dataset
├── checkpoints/       # Saved models
├── logs/              # Training logs
└── samples/           # Generated samples
```

## Citation

If you use this code, please consider citing:

```bibtex
@misc{conditional_image_generator,
  title={Conditional Image Generator with 2x Upscaling},
  author={Your Name},
  year={2025}
}
```

## License

MIT License
