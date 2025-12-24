#!/usr/bin/env python3
"""
Training Script for Conditional Image Generator (conditional.py)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
import csv
from datetime import datetime
from pathlib import Path

# Import models
from models.conditional import ConditionalImageGenerator, PerceptualLoss
from dataset import get_dataloaders
from utils import (
    save_checkpoint, load_checkpoint, calculate_psnr, calculate_ssim,
    visualize_results, AverageMeter, EarlyStopping, count_parameters
)
import config as cfg


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Conditional Image Generator (128->512)')
    
    # Data
    parser.add_argument('--data_dir', type=str, default=cfg.DATA_DIR)
    parser.add_argument('--input_size', type=int, default=cfg.INPUT_SIZE)
    parser.add_argument('--output_size', type=int, default=cfg.OUTPUT_SIZE)
    parser.add_argument('--num_conditions', type=int, default=cfg.NUM_CONDITIONS)
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=cfg.LATENT_DIM)
    parser.add_argument('--use_vae', action='store_true', default=cfg.ENABLE_VAE,
                        help='Enable VAE with KL loss')
    parser.add_argument('--use_perceptual', action='store_true', default=True,
                        help='Use perceptual loss')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=cfg.LEARNING_RATE)
    parser.add_argument('--num_workers', type=int, default=cfg.NUM_WORKERS)
    parser.add_argument('--gradient_accumulation', type=int, default=cfg.GRADIENT_ACCUMULATION_STEPS)
    
    # Multi-GPU
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    
    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default=cfg.CHECKPOINT_DIR)
    parser.add_argument('--log_dir', type=str, default=cfg.LOG_DIR)
    parser.add_argument('--sample_dir', type=str, default=cfg.SAMPLE_DIR)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    return parser.parse_args()


class MetricsLogger:
    """CSV-based metrics logger"""
    def __init__(self, log_dir: str, filename: str = 'metrics.csv'):
        self.log_dir = log_dir
        self.filepath = os.path.join(log_dir, filename)
        self.fieldnames = [
            'epoch', 'train_loss', 'train_mse', 'train_l1', 'train_perceptual',
            'val_loss', 'val_mse', 'val_l1', 'val_perceptual', 
            'val_psnr', 'val_ssim', 'learning_rate', 'timestamp'
        ]
        
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def log(self, metrics: dict):
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics)


def setup_ddp(rank: int, world_size: int):
    """Setup Distributed Data Parallel"""
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion_mse: nn.Module,
    criterion_l1: nn.Module,
    criterion_perceptual: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    scaler: GradScaler,
    epoch: int,
    gradient_accumulation_steps: int,
    rank: int,
    use_perceptual: bool = False,
    use_vae: bool = False,
) -> dict:
    """Train for one epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    mse_meter = AverageMeter()
    l1_meter = AverageMeter()
    perceptual_meter = AverageMeter()
    
    if rank == 0:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    else:
        pbar = train_loader
    
    optimizer.zero_grad()
    
    for batch_idx, batch_data in enumerate(pbar):
        initial_images, conditions, target_images = batch_data[0], batch_data[1], batch_data[2]
        
        initial_images = initial_images.to(device)
        conditions = conditions.to(device)
        target_images = target_images.to(device)
        
        with autocast(device_type='cuda', enabled=cfg.USE_AMP):
            # Forward pass
            if use_vae:
                out = model(initial_images, conditions)
                if isinstance(out, tuple):
                    generated_images, mu, logvar = out
                else:
                    generated_images = out
                    mu = None
                    logvar = None
            else:
                generated_images = model(initial_images, conditions)
                mu = None
                logvar = None

            # Compute losses
            mse_loss = criterion_mse(generated_images, target_images)
            l1_loss = criterion_l1(generated_images, target_images)
            total_loss = cfg.RECONSTRUCTION_WEIGHT * mse_loss + cfg.L1_WEIGHT * l1_loss

            perceptual_loss = torch.tensor(0.0).to(device)
            if use_perceptual and criterion_perceptual is not None:
                perceptual_loss = criterion_perceptual(generated_images, target_images)
                total_loss += cfg.PERCEPTUAL_WEIGHT * perceptual_loss

            if use_vae and mu is not None and logvar is not None:
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
                kl_weight = cfg.VAE_KL_WEIGHT * min(1.0, (epoch + 1) / max(1, cfg.VAE_KL_ANNEAL_EPOCHS))
                total_loss += kl_weight * kl_loss

            total_loss = total_loss / gradient_accumulation_steps
        
        scaler.scale(total_loss).backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Update meters
        loss_meter.update(total_loss.item() * gradient_accumulation_steps, initial_images.size(0))
        mse_meter.update(mse_loss.item(), initial_images.size(0))
        l1_meter.update(l1_loss.item(), initial_images.size(0))
        if use_perceptual:
            perceptual_meter.update(perceptual_loss.item(), initial_images.size(0))
        
        if batch_idx % cfg.EMPTY_CACHE_FREQUENCY == 0:
            torch.cuda.empty_cache()
        
        if rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'mse': f'{mse_meter.avg:.4f}',
                'l1': f'{l1_meter.avg:.4f}',
                'perc': f'{perceptual_meter.avg:.4f}' if use_perceptual else 'N/A'
            })
    
    return {
        'loss': loss_meter.avg,
        'mse': mse_meter.avg,
        'l1': l1_meter.avg,
        'perceptual': perceptual_meter.avg if use_perceptual else 0.0
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion_mse: nn.Module,
    criterion_l1: nn.Module,
    criterion_perceptual: nn.Module,
    device: str,
    epoch: int,
    rank: int,
    use_perceptual: bool = True,
    use_vae: bool = False,
) -> dict:
    """Validate the model"""
    model.eval()
    
    loss_meter = AverageMeter()
    mse_meter = AverageMeter()
    l1_meter = AverageMeter()
    perceptual_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    if rank == 0:
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
    else:
        pbar = val_loader
    
    sample_initials = []
    sample_generated = []
    sample_targets = []
    
    for batch_idx, batch_data in enumerate(pbar):
        initial_images, conditions, target_images = batch_data[0], batch_data[1], batch_data[2]
        
        initial_images = initial_images.to(device)
        conditions = conditions.to(device)
        target_images = target_images.to(device)
        
        # Forward pass
        out = model(initial_images, conditions)
        if use_vae and isinstance(out, tuple):
            generated_images, mu, logvar = out
        else:
            generated_images = out

        # Compute losses
        mse_loss = criterion_mse(generated_images, target_images)
        l1_loss = criterion_l1(generated_images, target_images)
        total_loss = cfg.RECONSTRUCTION_WEIGHT * mse_loss + cfg.L1_WEIGHT * l1_loss

        perceptual_loss = torch.tensor(0.0).to(device)
        if use_perceptual and criterion_perceptual is not None:
            perceptual_loss = criterion_perceptual(generated_images, target_images)
            total_loss += cfg.PERCEPTUAL_WEIGHT * perceptual_loss
        
        # Compute metrics
        psnr_val = calculate_psnr(generated_images, target_images)
        ssim_val = calculate_ssim(generated_images, target_images)
        
        loss_meter.update(total_loss.item(), initial_images.size(0))
        mse_meter.update(mse_loss.item(), initial_images.size(0))
        l1_meter.update(l1_loss.item(), initial_images.size(0))
        if use_perceptual:
            perceptual_meter.update(perceptual_loss.item(), initial_images.size(0))
        psnr_meter.update(psnr_val, initial_images.size(0))
        ssim_meter.update(ssim_val, initial_images.size(0))
        
        if batch_idx == 0 and rank == 0:
            sample_initials = initial_images[:cfg.NUM_SAMPLE_IMAGES].cpu()
            sample_generated = generated_images[:cfg.NUM_SAMPLE_IMAGES].cpu()
            sample_targets = target_images[:cfg.NUM_SAMPLE_IMAGES].cpu()
        
        if rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'psnr': f'{psnr_meter.avg:.2f}',
                'ssim': f'{ssim_meter.avg:.4f}'
            })
    
    return {
        'loss': loss_meter.avg,
        'mse': mse_meter.avg,
        'l1': l1_meter.avg,
        'perceptual': perceptual_meter.avg if use_perceptual else 0.0,
        'psnr': psnr_meter.avg,
        'ssim': ssim_meter.avg,
        'samples': (sample_initials, sample_generated, sample_targets)
    }


def train_worker(rank: int, world_size: int, args):
    """Training worker for each GPU"""
    
    args.use_vae = bool(args.use_vae)

    if world_size > 1:
        setup_ddp(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    
    # Create directories
    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.sample_dir, exist_ok=True)
        print(f"\n{'='*80}")
        print(f"Training Configuration")
        print(f"{'='*80}")
        print(f"GPUs: {world_size}")
        print(f"Input size: {args.input_size}x{args.input_size}")
        print(f"Output size: {args.output_size}x{args.output_size}")
        print(f"Upscaling: {args.output_size/args.input_size:.0f}x")
        print(f"Use VAE: {args.use_vae}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation}")
        print(f"{'='*80}\n")
    
    # Load data
    if rank == 0:
        print("Loading datasets...")
    
    train_loader, val_loader, norm_stats = get_dataloaders(
        data_dir=args.data_dir,
        train_features=cfg.TRAIN_FEATURES,
        val_features=cfg.VAL_FEATURES,
        initial_dir=cfg.INITIAL_DIR,
        target_dir=cfg.TARGET_DIR,
        batch_size=args.batch_size,
        input_size=args.input_size,
        output_size=args.output_size,
        num_workers=args.num_workers,
        pin_memory=True,
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size
    )
    
    if rank == 0:
        print(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    if rank == 0:
        print("Creating model...")
    
    model = ConditionalImageGenerator(
        num_conditions=args.num_conditions,
        input_size=args.input_size,
        output_size=args.output_size,
        latent_dim=args.latent_dim,
        encoder_channels=cfg.ENCODER_CHANNELS,
        decoder_channels=cfg.DECODER_CHANNELS,
        condition_hidden_dims=cfg.CONDITION_HIDDEN_DIMS,
        use_vae=args.use_vae,
        use_checkpoint=True,
        encoder_checkpoint=cfg.ENCODER_CHECKPOINT,
        device=f'cuda:{rank}'
    ).to(device)
    
    # Sync batch norm
    if world_size > 1 and cfg.SYNC_BATCHNORM:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # DDP wrap
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    if rank == 0:
        param_count = count_parameters(model.module if world_size > 1 else model)
        print(f"✓ Total parameters: {param_count:,}")
    
    # Loss functions
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = None
    
    if args.use_perceptual:
        criterion_perceptual = PerceptualLoss().to(device)
        if rank == 0:
            print("✓ Using perceptual loss")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch < cfg.WARMUP_EPOCHS:
            return (epoch + 1) / cfg.WARMUP_EPOCHS
        return 0.5 * (1 + np.cos(np.pi * (epoch - cfg.WARMUP_EPOCHS) / (args.epochs - cfg.WARMUP_EPOCHS)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(device='cuda', enabled=cfg.USE_AMP)
    early_stopping = EarlyStopping(patience=cfg.EARLY_STOP_PATIENCE)
    
    # Logger
    if rank == 0:
        metrics_logger = MetricsLogger(args.log_dir, cfg.METRICS_FILE)
        print(f"✓ Metrics saved to: {os.path.join(args.log_dir, cfg.METRICS_FILE)}")
    
    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"✓ Resuming from: {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, 
            model.module if world_size > 1 else model, 
            optimizer, 
            scheduler
        )
        start_epoch += 1
    
    # Training loop
    if rank == 0:
        print("\nStarting training...\n")
    
    last_val_loss = best_val_loss
    last_trained_epoch = start_epoch - 1
    
    for epoch in range(start_epoch, args.epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        train_metrics = train_epoch(
            model, train_loader, criterion_mse, criterion_l1, criterion_perceptual,
            optimizer, device, scaler, epoch, args.gradient_accumulation,
            rank, args.use_perceptual, args.use_vae
        )
        
        should_validate = (epoch + 1) % cfg.VALIDATION_FREQUENCY == 0 or (epoch + 1) == args.epochs
        
        if should_validate:
            val_metrics = validate(
                model, val_loader, criterion_mse, criterion_l1, criterion_perceptual,
                device, epoch, rank, args.use_perceptual, args.use_vae
            )
        else:
            val_metrics = None
        
        scheduler.step()
        
        # Logging
        if rank == 0:
            if should_validate:
                metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'train_mse': train_metrics['mse'],
                    'train_l1': train_metrics['l1'],
                    'train_perceptual': train_metrics['perceptual'],
                    'val_loss': val_metrics['loss'],
                    'val_mse': val_metrics['mse'],
                    'val_l1': val_metrics['l1'],
                    'val_perceptual': val_metrics['perceptual'],
                    'val_psnr': val_metrics['psnr'],
                    'val_ssim': val_metrics['ssim'],
                    'learning_rate': optimizer.param_groups[0]['lr']
                }
                metrics_logger.log(metrics)
                
                if cfg.SAVE_SAMPLES:
                    sample_initials, sample_generated, sample_targets = val_metrics['samples']
                    sample_path = os.path.join(args.sample_dir, f'epoch_{epoch+1:04d}.png')
                    visualize_results(
                        sample_initials, sample_generated, sample_targets,
                        num_samples=cfg.NUM_SAMPLE_IMAGES, save_path=sample_path
                    )
                
                print(f"\n{'='*80}")
                print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
                print(f"PSNR: {val_metrics['psnr']:.2f} dB | SSIM: {val_metrics['ssim']:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
                print(f"{'='*80}\n")
                
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                    save_checkpoint(
                        model.module if world_size > 1 else model,
                        optimizer, epoch, val_metrics['loss'],
                        best_model_path, scheduler
                    )
                    print(f"✓ Best model saved")
                
                last_val_loss = val_metrics['loss']
                
                if early_stopping(val_metrics['loss']):
                    print(f"\n✓ Early stopping at epoch {epoch + 1}")
                    break
            else:
                print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_metrics['loss']:.4f}")
            
            if (epoch + 1) % cfg.SAVE_FREQUENCY == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1:04d}.pth')
                loss_to_save = val_metrics['loss'] if should_validate else train_metrics['loss']
                save_checkpoint(
                    model.module if world_size > 1 else model,
                    optimizer, epoch, loss_to_save,
                    checkpoint_path, scheduler
                )
                print(f"✓ Checkpoint saved")
        
        if world_size > 1:
            dist.barrier()
        
        torch.cuda.empty_cache()
        last_trained_epoch = epoch
    
    # Save final model
    if rank == 0:
        final_epoch = last_trained_epoch if last_trained_epoch >= 0 else max(start_epoch - 1, 0)
        final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
        save_checkpoint(
            model.module if world_size > 1 else model,
            optimizer, final_epoch, last_val_loss,
            final_model_path, scheduler
        )
        print("\n✓ Training completed!")
        print(f"✓ Final model saved: {final_model_path}")
        print(f"✓ Best validation loss: {best_val_loss:.4f}")
    
    if world_size > 1:
        cleanup_ddp()


def main():
    args = get_args()
    
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"Using torchrun: rank={local_rank}, world_size={world_size}")
        train_worker(local_rank, world_size, args)
    elif args.world_size > 1:
        mp.spawn(train_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        train_worker(0, 1, args)


if __name__ == "__main__":
    main()
