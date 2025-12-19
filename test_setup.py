#!/usr/bin/env python3
"""
Quick test script to verify data loading and model forward pass
Run this before starting full training
"""

import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from model import ConditionalImageGenerator
from dataset import get_dataloaders

def test_data_loading():
    """Test if data loads correctly"""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)
    
    try:
        # Create dataloaders
        train_loader, val_loader, norm_stats = get_dataloaders(
            data_dir=cfg.DATA_DIR,
            train_features=cfg.TRAIN_FEATURES,
            val_features=cfg.VAL_FEATURES,
            initial_dir=cfg.INITIAL_DIR,
            target_dir=cfg.TARGET_DIR,
            batch_size=4,  # Small batch for testing
            input_size=cfg.INPUT_SIZE,
            output_size=cfg.OUTPUT_SIZE,
            num_workers=0,  # No multiprocessing for test
            pin_memory=False,
            distributed=False,
            rank=0,
            world_size=1
        )
        
        print(f"\n✓ Data loading successful!")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Test one batch
        print(f"\nTesting one batch from training set...")
        initial_images, conditions, target_images, _, _ = next(iter(train_loader))
        
        print(f"  Initial images shape: {initial_images.shape}")
        print(f"  Conditions shape: {conditions.shape}")
        print(f"  Target images shape: {target_images.shape}")
        
        assert initial_images.shape[1:] == (3, cfg.INPUT_SIZE, cfg.INPUT_SIZE), "Wrong input size!"
        assert target_images.shape[1:] == (3, cfg.OUTPUT_SIZE, cfg.OUTPUT_SIZE), "Wrong output size!"
        assert conditions.shape[1] == cfg.NUM_CONDITIONS, "Wrong number of conditions!"
        
        print(f"\n✓ Batch shapes are correct!")
        return True
        
    except Exception as e:
        print(f"\n❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test if model can do forward pass"""
    print("\n" + "="*60)
    print("Testing Model Forward Pass")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ No CUDA GPUs available! Training requires GPU.")
        return False
    
    device = torch.device('cuda:0')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    
    try:
        # Create model
        model = ConditionalImageGenerator(
            num_conditions=cfg.NUM_CONDITIONS,
            input_size=cfg.INPUT_SIZE,
            output_size=cfg.OUTPUT_SIZE,
            encoder_channels=cfg.ENCODER_CHANNELS,
            decoder_channels=cfg.DECODER_CHANNELS,
            condition_hidden_dims=cfg.CONDITION_HIDDEN_DIMS,
            use_checkpoint=False  # Disable for testing
        ).to(device)
        
        print(f"\n✓ Model created successfully!")
        print(f"  Total parameters: {model.get_num_parameters():,}")
        
        # Test forward pass
        batch_size = 2
        dummy_image = torch.randn(batch_size, 3, cfg.INPUT_SIZE, cfg.INPUT_SIZE, device=device)
        dummy_conditions = torch.randn(batch_size, cfg.NUM_CONDITIONS, device=device)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_image, dummy_conditions)
        
        print(f"\n✓ Forward pass successful!")
        print(f"  Input: {dummy_image.shape}")
        print(f"  Output: {output.shape}")
        print(f"  Expected output: ({batch_size}, 3, {cfg.OUTPUT_SIZE}, {cfg.OUTPUT_SIZE})")
        
        assert output.shape == (batch_size, 3, cfg.OUTPUT_SIZE, cfg.OUTPUT_SIZE), "Wrong output shape!"
        
        # Check GPU memory
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"\n  GPU memory allocated: {allocated:.2f} GB")
        print(f"  GPU memory reserved: {reserved:.2f} GB")
        
        # Cleanup
        del model, dummy_image, dummy_conditions, output
        torch.cuda.empty_cache()
        
        print(f"\n✓ Model test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("Pre-Training System Check")
    print("="*60)
    
    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / (1024**3):.1f} GB")
    
    # Check data directory
    print(f"\nData directory: {cfg.DATA_DIR}")
    print(f"  Exists: {os.path.exists(cfg.DATA_DIR)}")
    if os.path.exists(cfg.DATA_DIR):
        print(f"  Initial images: {len(os.listdir(os.path.join(cfg.DATA_DIR, cfg.INITIAL_DIR)))} files")
        print(f"  Target images: {len(os.listdir(os.path.join(cfg.DATA_DIR, cfg.TARGET_DIR)))} files")
    
    # Run tests
    success = True
    success &= test_data_loading()
    success &= test_model()
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("✅ All tests passed! Ready for training.")
        print("="*60)
        print("\nTo start training:")
        print("  Single GPU:  ./run_training.sh 1")
        print("  Two GPUs:    ./run_training.sh 2")
        print("  Quick test:  ./run_training.sh 1 1")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before training.")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
