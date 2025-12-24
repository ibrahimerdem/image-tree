"""
Training script for Integrated Conditional Generator
Demonstrates training with both GAN-style and conditional.py approaches
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.integrated_conditional_generator import (
    IntegratedConditionalGenerator, 
    TrainingModule
)
import config as cfg


class DummyDataset(Dataset):
    """Dummy dataset for testing - replace with actual data loading"""
    
    def __init__(self, num_samples: int = 100, num_conditions: int = 9):
        self.num_samples = num_samples
        self.num_conditions = num_conditions
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        initial_image = torch.randn(3, cfg.INPUT_SIZE, cfg.INPUT_SIZE)
        target_image = torch.randn(3, cfg.OUTPUT_SIZE, cfg.OUTPUT_SIZE)
        conditions = torch.randn(self.num_conditions)
        
        return {
            'initial_image': initial_image,
            'target_image': target_image,
            'conditions': conditions
        }


class Trainer:
    """Training loop for IntegratedConditionalGenerator"""
    
    def __init__(
        self,
        model: IntegratedConditionalGenerator,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        device: str = 'cuda:0',
        learning_rate: float = 3e-4,
        num_epochs: int = 100,
        checkpoint_dir: str = './checkpoints'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training module with losses
        self.trainer = TrainingModule(
            model,
            reconstruction_weight=cfg.RECONSTRUCTION_WEIGHT,
            l1_weight=cfg.L1_WEIGHT,
            perceptual_weight=cfg.PERCEPTUAL_WEIGHT,
            vae_kl_weight=cfg.VAE_KL_WEIGHT,
            device=device
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=cfg.WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'mse': 0.0,
            'l1': 0.0,
            'perceptual': 0.0,
            'kl': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            conditions = batch['conditions'].to(self.device)
            target = batch['target_image'].to(self.device)
            initial_image = batch['initial_image'].to(self.device)
            
            # Forward pass
            losses = self.trainer(
                conditions,
                target,
                initial_image=initial_image
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'l1': losses['l1'].item()
            })
        
        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def val_epoch(self) -> dict:
        """Validate for one epoch"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        epoch_losses = {
            'total': 0.0,
            'mse': 0.0,
            'l1': 0.0,
            'perceptual': 0.0,
            'kl': 0.0
        }
        
        pbar = tqdm(self.val_loader, desc="Validation")
        for batch in pbar:
            conditions = batch['conditions'].to(self.device)
            target = batch['target_image'].to(self.device)
            initial_image = batch['initial_image'].to(self.device)
            
            # Forward pass
            losses = self.trainer(
                conditions,
                target,
                initial_image=initial_image
            )
            
            # Track losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
        
        # Average losses
        num_batches = len(self.val_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train(self):
        """Full training loop"""
        print("\n" + "="*60)
        print("Starting Training - Integrated Conditional Generator")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Num epochs: {self.num_epochs}")
        print(f"Batch size: {cfg.BATCH_SIZE}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Model parameters: {self.model.get_num_parameters()}")
        print("="*60 + "\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Training
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            # Validation
            val_losses = self.val_epoch()
            if val_losses:
                self.val_losses.append(val_losses)
                best_val_loss = min(best_val_loss, val_losses['total'])
            
            # Learning rate step
            self.scheduler.step()
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            if train_losses.get('l1'):
                print(f"    - L1: {train_losses['l1']:.4f}")
            if train_losses.get('mse'):
                print(f"    - MSE: {train_losses['mse']:.4f}")
            if train_losses.get('kl'):
                print(f"    - KL: {train_losses['kl']:.4f}")
            
            if val_losses:
                print(f"  Val Loss: {val_losses['total']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch+1}.pt"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")
        
        print("\nTraining completed!")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Integrated Conditional Generator")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--batch-size', type=int, default=cfg.BATCH_SIZE)
    parser.add_argument('--num-epochs', type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument('--learning-rate', type=float, default=cfg.LEARNING_RATE)
    parser.add_argument('--num-workers', type=int, default=cfg.NUM_WORKERS)
    parser.add_argument('--use-vae', action='store_true', default=cfg.ENABLE_VAE)
    parser.add_argument('--use-initial-image', action='store_true', default=False)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--num-samples', type=int, default=100, help='Num samples for dummy dataset')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = IntegratedConditionalGenerator(
        num_conditions=cfg.NUM_CONDITIONS,
        input_size=cfg.INPUT_SIZE,
        output_size=cfg.OUTPUT_SIZE,
        noise_dim=100,
        embed_dim=256,
        embed_out_dim=128,
        latent_dim=cfg.LATENT_DIM,
        decoder_channels=cfg.DECODER_CHANNELS,
        condition_hidden_dims=cfg.CONDITION_HIDDEN_DIMS,
        use_vae=args.use_vae,
        initial_image=args.use_initial_image,
        device=device
    ).to(device)
    
    # Create datasets (replace with actual data)
    train_dataset = DummyDataset(num_samples=args.num_samples, num_conditions=cfg.NUM_CONDITIONS)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for dummy data
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        device=device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
