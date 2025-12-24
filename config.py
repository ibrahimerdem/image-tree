
# Data settings
DATA_DIR = './data'
INPUT_SIZE = 128
OUTPUT_SIZE = 512 
NUM_CONDITIONS = 9
INITIAL_DIR = 'initial'  
TARGET_DIR = 'target'    
TRAIN_FEATURES = 'training_features.csv'
VAL_FEATURES = 'validation_features.csv'
TEST_FEATURES = 'test_features.csv'

# Model architecture
# Pretrained encoder: 128 -> 64 -> 32 -> 16 -> 8 -> 4 (5 conv layers, outputs 4x4)
# Decoder: 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512 (7 deconv layers)
LATENT_DIM = 256
ENCODER_CHECKPOINT = 'encoder_epoch_50.pth'  # Pretrained encoder checkpoint (frozen, no training)
ENCODER_CHANNELS = [64, 128, 256, 256, 256]  # (only used for reference, actual encoder is pretrained)
DECODER_CHANNELS = [1024, 512, 256, 128, 64, 32, 16]  # 7 levels for 4x4->512x512 (2^7 = 128x upsampling)
CONDITION_HIDDEN_DIMS = [64, 128, 256]

# GAN generator settings
GAN_NOISE_DIM = 100
GAN_EMBED_DIM = 256
GAN_EMBED_OUT_DIM = 128
GAN_CHANNELS = 3
GAN_USE_INITIAL_IMAGE = True
GAN_ADVERSARIAL_WEIGHT = 1.0
GAN_DISCRIMINATOR_LR = 2e-4

# Training settings
BATCH_SIZE = 20 
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 80
NUM_EPOCHS = 150
LEARNING_RATE = 3e-4 
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 8

# Loss weights - deterministic output with emphasis on sharpness
# Total loss = RECONSTRUCTION_WEIGHT*MSE + L1_WEIGHT*L1 + PERCEPTUAL_WEIGHT*Perceptual + KL (when VAE enabled)
RECONSTRUCTION_WEIGHT = 0.2  # MSE: low weight to reduce blurring tendency
L1_WEIGHT = 2.0  # L1: emphasized for sharp edges and details
PERCEPTUAL_WEIGHT = 0.8  # Perceptual: higher weight for natural textures and coherence

# VAE / latent settings - minimal stochasticity, deterministic output
ENABLE_VAE = True  # Enable conditional VAE (acts as denoising autoencoder with minimal sampling)
VAE_LATENT_DIM = 64  # Channels in spatial latent map (64 x 16 x 16)
VAE_KL_WEIGHT = 0.1  # KL divergence weight: minimal regularization (0.01 -> 0.1) for deterministic output
VAE_KL_ANNEAL_EPOCHS = 5  # Fast annealing: quick focus on reconstruction accuracy
VAE_SPATIAL = True  # Use spatial latent map (preserves local structure vs global vector)

# Optimization - Cosine annealing with warmup for smooth learning rate decay
LR_SCHEDULER = 'cosine'  # 'cosine' or 'plateau'
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
WARMUP_EPOCHS = 5  # Gradual warmup to avoid instability

# Training options
USE_AMP = True  # Mixed precision training - critical for high-res
GRADIENT_CLIP = 5.0  # Increased for more aggressive learning
EARLY_STOP_PATIENCE = 20 

# Multi-GPU settings
USE_DDP = True  # Distributed Data Parallel
SYNC_BATCHNORM = True  # Synchronize batch normalization across GPUs

# Checkpointing
CHECKPOINT_DIR = './checkpoints'
SAVE_FREQUENCY = 5
KEEP_LAST_N = 1

# Logging
LOG_DIR = './logs'
METRICS_FILE = 'training_metrics.csv'
LOG_FREQUENCY = 50  # Log every N batches
VALIDATION_FREQUENCY = 5
SAVE_SAMPLES = True
SAVE_SAMPLE_FREQUENCY = VALIDATION_FREQUENCY
NUM_SAMPLE_IMAGES = 4
SAMPLE_DIR = './samples'

# Testing
TEST_RESULTS_DIR = './test_results'
TEST_METRICS_FILE = 'test_metrics.csv'
TEST_SAVE_ALL_IMAGES = True  # Save all generated images during testing
TEST_BATCH_SIZE = 4

# Device
DEVICE = 'cuda'  # Will use both GPUs with DDP

# Memory optimization
CHANNELS_LAST = True  # Use channels_last memory format for better performance
EMPTY_CACHE_FREQUENCY = 100  # Empty CUDA cache every N batches