"""
Unified Dataset and DataLoader for Train/Val/Test
Uses shared image folders with separate feature CSV files
Supports different filenames for initial and target images
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, Literal
import numpy as np


class ConditionalImageDataset(Dataset):
    """
    Unified dataset for train/validation/test
    Uses shared image folders with separate feature files
    Supports different filenames for initial and target images
    
    Directory structure:
        data/
            initial/              # Shared initial images
                input_001.jpg
                input_002.jpg
                ...
            target/               # Shared target images
                output_001.png
                output_002.png
                ...
            training_features.csv     # Training features
            validation_features.csv   # Validation features
            test_features.csv         # Test features
    
    CSV format (with different filenames):
        initial_filename,target_filename,feature_1,feature_2,...
        input_001.jpg,output_001.png,0.5,1.2,...
        input_002.jpg,output_002.png,0.3,0.9,... 
    """
    
    def __init__(
        self,
        data_dir: str,
        features_file: str,
        initial_dir: str = 'initial',
        target_dir: str = 'target',
        input_size: int = 128,
        output_size: int = 256,
        normalize_conditions: bool = True,
        condition_stats: Optional[dict] = None,
        transform_input: Optional[transforms.Compose] = None,
        transform_output: Optional[transforms.Compose] = None,
        mode:  Literal['train', 'val', 'test'] = 'train'
    ):
        """
        Args:
            data_dir:  Root data directory containing image folders and feature files
            features_file: Name of the CSV file with features
            initial_dir: Name of initial images folder (relative to data_dir)
            target_dir: Name of target images folder (relative to data_dir)
            input_size: Size to resize input images to (default: 128)
            output_size: Size to resize output images to (default: 256)
            normalize_conditions: Whether to normalize condition features
            condition_stats: Pre-computed statistics for normalization (for test set)
            transform_input: Optional custom transforms for input images
            transform_output: Optional custom transforms for output images
            mode: 'train', 'val', or 'test' - affects augmentation and behavior
        """
        self.data_dir = data_dir
        self.input_size = input_size
        self.output_size = output_size
        self.mode = mode

        self.initial_dir = os.path.join(data_dir, initial_dir)
        self.target_dir = os. path.join(data_dir, target_dir)

        self.features_file = os.path.join(data_dir, features_file)
        
        # Validate paths
        if not os.path. exists(self.initial_dir):
            raise ValueError(f"Initial images directory not found: {self.initial_dir}")
        if not os.path.exists(self.target_dir):
            raise ValueError(f"Target images directory not found: {self.target_dir}")
        if not os.path.exists(self.features_file):
            raise ValueError(f"Features file not found: {self.features_file}")
        
        # Load features
        self.features_df = pd.read_csv(self.features_file)
        
        # Check CSV format - determine if it has separate initial/target columns
        columns = self.features_df.columns.tolist()
        
        if 'initial_filename' in columns and 'target_filename' in columns:
            self.initial_filenames = self.features_df['initial_filename'].tolist()
            self.target_filenames = self.features_df['target_filename'].tolist()
            self.paired_format = True
            condition_cols = [col for col in columns if col not in ['initial_filename', 'target_filename']]
        else:
            raise ValueError(
                "CSV must contain either:\n"
                "  1. 'initial_filename' and 'target_filename' columns, OR\n"
                "  2. 'filename' column (assumes same name for both initial and target)"
            )
        
        print(f"[{mode.upper()}] Loaded {len(self.initial_filenames)} samples from {features_file}")
        
        # Extract condition values
        self.conditions = self.features_df[condition_cols].values.astype(np.float32)
        self.num_conditions = len(condition_cols)
        
        print(f"[{mode.upper()}] Number of condition features: {self.num_conditions}")
        
        # Normalize conditions
        if normalize_conditions:
            if condition_stats is not None:
                # Use provided statistics (for test set)
                self.condition_mean = condition_stats['mean']
                self.condition_std = condition_stats['std']
                print(f"[{mode.upper()}] Using provided normalization statistics")
            else:
                # Compute statistics from this dataset (for train/val)
                self.condition_mean = self.conditions.mean(axis=0)
                self.condition_std = self.conditions.std(axis=0) + 1e-8
                print(f"[{mode.upper()}] Computed normalization statistics")
            
            self.conditions = (self.conditions - self.condition_mean) / self.condition_std
        else:
            self.condition_mean = None
            self.condition_std = None
        
        # Image transforms - separate for input and output (NO AUGMENTATION)
        if transform_input is None:
            # Same transforms for train/val/test - no augmentation
            self.transform_input = transforms.Compose([
                transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform_input = transform_input
        
        if transform_output is None:
            # Same transforms for train/val/test - no augmentation
            self.transform_output = transforms.Compose([
                transforms.Resize((output_size, output_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform_output = transform_output
        
        print(f"[{mode.upper()}] Input: {input_size}x{input_size}, Output: {output_size}x{output_size}")
    
    def __len__(self) -> int:
        return len(self.initial_filenames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        """
        Returns:
            initial_image: Initial input image [3, H, W]
            conditions: Condition features [num_conditions]
            target_image: Target output image [3, H, W]
            initial_filename: Initial image filename
            target_filename: Target image filename
        """
        initial_filename = self.initial_filenames[idx]
        target_filename = self.target_filenames[idx]
        
        # Load images with error handling
        try:
            initial_path = os.path.join(self.initial_dir, initial_filename)
            target_path = os.path.join(self.target_dir, target_filename)
            
            # Check if files exist
            if not os.path.exists(initial_path):
                raise FileNotFoundError(f"Initial image not found: {initial_path}")
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Target image not found:  {target_path}")
            
            initial_image = Image.open(initial_path).convert('RGB')
            target_image = Image.open(target_path).convert('RGB')
            
            # Apply separate transforms for input and output
            initial_image = self.transform_input(initial_image)
            target_image = self.transform_output(target_image)
            
            # Get conditions
            conditions = torch.from_numpy(self.conditions[idx])
            
            return initial_image, conditions, target_image, initial_filename, target_filename
        
        except Exception as e: 
            print(f"Error loading pair ({initial_filename}, {target_filename}): {e}")
            # Return a random valid sample instead
            random_idx = np.random.randint(0, len(self))
            if random_idx == idx:  # Avoid infinite loop
                random_idx = (idx + 1) % len(self)
            return self.__getitem__(random_idx)
    
    def denormalize_conditions(self, conditions: np.ndarray) -> np.ndarray:
        """Denormalize conditions back to original scale"""
        if self.condition_mean is not None:
            return conditions * self.condition_std + self.condition_mean
        return conditions
    
    def get_statistics(self):
        """Get dataset statistics"""
        return {
            'num_samples': len(self),
            'num_conditions': self.num_conditions,
            'condition_mean': self.condition_mean,
            'condition_std':  self.condition_std,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'mode': self.mode,
            'paired_format': self.paired_format
        }
    
    def get_normalization_stats(self):
        """Get normalization statistics for sharing with test set"""
        return {
            'mean': self.condition_mean,
            'std': self.condition_std
        }


def get_dataloader(
    data_dir:  str,
    features_file: str,
    initial_dir:  str = 'initial',
    target_dir: str = 'target',
    batch_size: int = 8,
    input_size: int = 128,
    output_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    mode: Literal['train', 'val', 'test'] = 'train',
    condition_stats: Optional[dict] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
) -> DataLoader:
    """
    Create a single dataloader for train/val/test
    
    Args:
        data_dir:  Root data directory
        features_file: Features CSV filename
        initial_dir: Initial images folder name
        target_dir:  Target images folder name
        batch_size: Batch size per GPU
        input_size: Input image size (default: 128)
        output_size: Output image size (default: 256)
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        mode: 'train', 'val', or 'test'
        condition_stats: Pre-computed normalization statistics (for test)
        distributed: Whether to use distributed training
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        dataloader
    """
    dataset = ConditionalImageDataset(
        data_dir=data_dir,
        features_file=features_file,
        initial_dir=initial_dir,
        target_dir=target_dir,
        input_size=input_size,
        output_size=output_size,
        mode=mode,
        condition_stats=condition_stats
    )
    
    # Create sampler for distributed training
    sampler = None
    shuffle = (mode == 'train')
    
    if distributed and mode == 'train':
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        shuffle = False  # Sampler handles shuffling
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(mode == 'train'),
        persistent_workers=True if num_workers > 0 else False
    )
    
    return dataloader


def get_dataloaders(
    data_dir: str,
    train_features:  str,
    val_features:  str,
    initial_dir: str = 'initial',
    target_dir: str = 'target',
    batch_size: int = 8,
    input_size: int = 128,
    output_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Create train and validation dataloaders with shared image folders
    Returns normalization statistics for use with test set
    
    Returns:
        train_loader, val_loader, normalization_stats
    """
    if rank == 0:
        print(f"\nCreating datasets from:")
        print(f"  Data directory: {data_dir}")
        print(f"  Initial images: {os.path.join(data_dir, initial_dir)}")
        print(f"  Target images:  {os.path.join(data_dir, target_dir)}")
        print(f"  Training features: {train_features}")
        print(f"  Validation features: {val_features}")
    
    # Create training dataset first to get normalization statistics
    train_dataset = ConditionalImageDataset(
        data_dir=data_dir,
        features_file=train_features,
        initial_dir=initial_dir,
        target_dir=target_dir,
        input_size=input_size,
        output_size=output_size,
        mode='train'
    )
    
    # Get normalization statistics from training set
    normalization_stats = train_dataset.get_normalization_stats()
    
    # Create validation dataset with training statistics
    val_dataset = ConditionalImageDataset(
        data_dir=data_dir,
        features_file=val_features,
        initial_dir=initial_dir,
        target_dir=target_dir,
        input_size=input_size,
        output_size=output_size,
        mode='val',
        condition_stats=normalization_stats
    )
    
    if rank == 0:
        print(f"\nDataset statistics:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Number of conditions: {train_dataset.num_conditions}")
    
    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    shuffle_train = True
    
    if distributed: 
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        shuffle_train = False
        
        if rank == 0:
            print(f"\nUsing DistributedSampler:")
            print(f"  World size: {world_size}")
            print(f"  Samples per GPU (train): {len(train_dataset) // world_size}")
            print(f"  Samples per GPU (val): {len(val_dataset) // world_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    if rank == 0:
        print(f"\nDataLoader configuration:")
        print(f"  Batch size per GPU: {batch_size}")
        print(f"  Effective batch size: {batch_size * world_size}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Num workers: {num_workers}")
    
    return train_loader, val_loader, normalization_stats


def verify_dataset(data_dir: str, train_features: str, val_features: str, test_features: str = None):
    """
    Verify dataset structure and print statistics
    """
    print("="*80)
    print("DATASET VERIFICATION")
    print("="*80)
    
    # Check directories
    initial_dir = os.path.join(data_dir, 'initial')
    target_dir = os.path.join(data_dir, 'target')
    train_path = os.path.join(data_dir, train_features)
    val_path = os.path.join(data_dir, val_features)
    
    print(f"\n1. Checking directory structure:")
    print(f"   {'Success' if os.path.exists(data_dir) else 'Failure'} Data directory:  {data_dir}")
    print(f"   {'Success' if os.path.exists(initial_dir) else 'Failure'} Initial images: {initial_dir}")
    print(f"   {'Success' if os.path.exists(target_dir) else 'Failure'} Target images:  {target_dir}")
    print(f"   {'Success' if os.path.exists(train_path) else 'Failure'} Training features: {train_path}")
    print(f"   {'Success' if os.path.exists(val_path) else 'Failure'} Validation features: {val_path}")
    
    if test_features:
        test_path = os.path.join(data_dir, test_features)
        print(f"   {'Success' if os. path.exists(test_path) else 'Failure'} Test features: {test_path}")
    
    required_paths = [data_dir, initial_dir, target_dir, train_path, val_path]
    if not all([os.path.exists(p) for p in required_paths]):
        print("\n   Failure: Missing required files or directories!")
        return False
    
    # Count images
    initial_images = set(os.listdir(initial_dir))
    target_images = set(os.listdir(target_dir))
    
    print(f"\n2. Image counts:")
    print(f"   Initial images: {len(initial_images)}")
    print(f"   Target images: {len(target_images)}")
    
    # Load feature files
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"\n3. Feature file statistics:")
    print(f"   Training samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")
    
    # Detect CSV format
    train_cols = train_df.columns.tolist()
    if 'initial_filename' in train_cols and 'target_filename' in train_cols:
        print(f"   CSV format:  Paired (initial_filename, target_filename)")
        initial_col = 'initial_filename'
        target_col = 'target_filename'
        num_conditions = len(train_cols) - 2
    elif 'filename' in train_cols: 
        print(f"   CSV format: Single filename")
        initial_col = 'filename'
        target_col = 'filename'
        num_conditions = len(train_cols) - 1
    else:
        print(f"   Invalid CSV format!")
        return False
    
    print(f"   Number of conditions: {num_conditions}")
    
    if test_features and os.path.exists(os.path.join(data_dir, test_features)):
        test_df = pd.read_csv(os.path.join(data_dir, test_features))
        print(f"   Test samples: {len(test_df)}")
    
    # Check for missing images
    print(f"\n4. Validating image pairs:")
    
    def check_missing(df, dataset_name):
        initial_files = set(df[initial_col].tolist())
        target_files = set(df[target_col]. tolist())
        
        missing_initial = [f for f in initial_files if f not in initial_images]
        missing_target = [f for f in target_files if f not in target_images]
        
        print(f"   {dataset_name}:")
        print(f"     Missing initial images: {len(missing_initial)}")
        print(f"     Missing target images: {len(missing_target)}")
        
        if missing_initial:
            print(f"       First missing initial:  {missing_initial[0]}")
        if missing_target: 
            print(f"       First missing target: {missing_target[0]}")
        
        return len(missing_initial) == 0 and len(missing_target) == 0
    
    train_valid = check_missing(train_df, "Training")
    val_valid = check_missing(val_df, "Validation")
    
    if test_features and os.path.exists(os.path.join(data_dir, test_features)):
        test_valid = check_missing(test_df, "Test")
    
    # Check for overlapping samples (if using paired format)
    if initial_col == 'initial_filename': 
        print(f"\n5. Data split verification:")
        
        def get_pair_set(df):
            return set(zip(df['initial_filename'], df['target_filename']))
        
        train_pairs = get_pair_set(train_df)
        val_pairs = get_pair_set(val_df)
        
        overlap_train_val = train_pairs.intersection(val_pairs)
        print(f"   Train/Val overlap: {len(overlap_train_val)} pairs")
        if len(overlap_train_val) > 0:
            print(f"   Warning: {len(overlap_train_val)} pairs appear in both train and val!")
        
        if test_features and os.path.exists(os.path.join(data_dir, test_features)):
            test_pairs = get_pair_set(test_df)
            overlap_train_test = train_pairs.intersection(test_pairs)
            overlap_val_test = val_pairs.intersection(test_pairs)
            print(f"   Train/Test overlap: {len(overlap_train_test)} pairs")
            print(f"   Val/Test overlap: {len(overlap_val_test)} pairs")
            if len(overlap_train_test) > 0 or len(overlap_val_test) > 0:
                print(f"   Warning: Test set overlaps with train/val!")
    
    # Sample statistics
    print(f"\n6.  Condition statistics (training):")
    condition_cols = [col for col in train_df.columns if col not in [initial_col, target_col]]
    for col in condition_cols[: 5]:  # Show first 5 conditions
        values = train_df[col].values
        print(f"   {col}: mean={values.mean():.3f}, std={values.std():.3f}, "
              f"min={values. min():.3f}, max={values.max():.3f}")
    if len(condition_cols) > 5:
        print(f"   ... and {len(condition_cols) - 5} more conditions")
    
    print("\n" + "="*80)
    if train_valid and val_valid:
        print("Success: Dataset verification complete!  All images found.")
    else:
        print("Warning: Dataset verification complete with warnings.  Some images are missing.")
    print("="*80 + "\n")
    
    return train_valid and val_valid


if __name__ == "__main__": 
    # Example verification
    print("Unified Dataset module loaded successfully!\n")
    print("Supports two CSV formats:")
    print("\n1. Paired format (different filenames for initial and target):")
    print("   initial_filename,target_filename,feature_1,feature_2,...")
    print("   input_001. jpg,output_001.png,0.5,1.2,...")
    print("   input_002.jpg,output_002.png,0.3,0.9,...")
    print("\n2. Single filename format (same filename for both):")
    print("   filename,feature_1,feature_2,...")
    print("   img_001.jpg,0.5,1.2,...")
    print("   img_002.jpg,0.3,0.9,...")
    print("\nTo verify your dataset, run:")
    print("  python dataset. py --data_dir ./data")
    
    import sys
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', type=str, default='./data')
        parser.add_argument('--train_features', type=str, default='training_features.csv')
        parser.add_argument('--val_features', type=str, default='validation_features.csv')
        parser.add_argument('--test_features', type=str, default='test_features.csv')
        args = parser.parse_args()
        
        verify_dataset(args.data_dir, args. train_features, args.val_features, args.test_features)