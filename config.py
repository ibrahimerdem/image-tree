
# Data settings
DATA_DIR = './data'
INPUT_SIZE = 128  
OUTPUT_SIZE = 512 
NUM_CONDITIONS = 9

# Dataset structure - shared images, separate feature files
INITIAL_DIR = 'initial'  
TARGET_DIR = 'target'    
TRAIN_FEATURES = 'training_features.csv'
VAL_FEATURES = 'validation_features.csv'
TEST_FEATURES = 'test_features.csv'

# Model architecture
# Encoder: 128 -> 64 -> 32 -> 16 (3 downsamples)
# Decoder: 16 -> 32 -> 64 -> 128 -> 256 -> 512 (5 upsamples for 4x resolution increase)
LATENT_DIM = 256
ENCODER_CHANNELS = [64, 128, 256]
DECODER_CHANNELS = [256, 128, 64, 32]
CONDITION_HIDDEN_DIMS = [64, 128, 256]

# Training settings
BATCH_SIZE = 20 
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 80
NUM_EPOCHS = 150
LEARNING_RATE = 3e-3 
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 8

# Loss weights
RECONSTRUCTION_WEIGHT = 0.5  # Reduced - MSE alone can cause blurring
L1_WEIGHT = 2.0  # Increased for sharper edges and details
PERCEPTUAL_WEIGHT = 1.0  # Increased significantly for perceptual quality

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