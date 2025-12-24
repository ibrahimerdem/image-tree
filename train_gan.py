#!/usr/bin/env python3
"""Training script for the GAN-based generator (conditional_gan.Generator)."""

import os
import argparse
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np

from models.conditional_gan import Generator as GANGenerator, Discriminator as GANDiscriminator
from dataset import get_dataloaders
from utils import (
    save_checkpoint, load_checkpoint, calculate_psnr, calculate_ssim,
    visualize_results, AverageMeter, EarlyStopping, count_parameters
)
import config as cfg


def get_args():
    parser = argparse.ArgumentParser(description='Train GAN-based Conditional Image Generator (128->512)')

    # Data
    parser.add_argument('--data_dir', type=str, default=cfg.DATA_DIR)
    parser.add_argument('--input_size', type=int, default=cfg.INPUT_SIZE)
    parser.add_argument('--output_size', type=int, default=cfg.OUTPUT_SIZE)
    parser.add_argument('--num_conditions', type=int, default=cfg.NUM_CONDITIONS)

    # Generator parameters
    parser.add_argument('--noise_dim', type=int, default=cfg.GAN_NOISE_DIM)
    parser.add_argument('--embed_dim', type=int, default=cfg.GAN_EMBED_DIM)
    parser.add_argument('--embed_out_dim', type=int, default=cfg.GAN_EMBED_OUT_DIM)
    parser.add_argument('--channels', type=int, default=cfg.GAN_CHANNELS)
    parser.add_argument('--no_initial_image', action='store_true',
                        help='Disable usage of initial images as generator input')
    parser.add_argument('--use_perceptual', action='store_true', default=True,
                        help='Use perceptual loss')

    # Training
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=cfg.LEARNING_RATE)
    parser.add_argument('--disc_lr', type=float, default=cfg.GAN_DISCRIMINATOR_LR)
    parser.add_argument('--num_workers', type=int, default=cfg.NUM_WORKERS)
    parser.add_argument('--gradient_accumulation', type=int, default=cfg.GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument('--adv_weight', type=float, default=cfg.GAN_ADVERSARIAL_WEIGHT)

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

    def __init__(self, log_dir: str, filename: str = 'gan_metrics.csv'):
        self.log_dir = log_dir
        self.filepath = os.path.join(log_dir, filename)
        self.fieldnames = [
            'epoch', 'train_loss', 'train_mse', 'train_l1', 'train_perceptual', 'train_adv', 'disc_loss',
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


class GANTrainingWrapper(nn.Module):
    """Adapts GANGenerator to accept (initial_images, conditions)."""

    def __init__(self, generator: nn.Module, noise_dim: int, use_initial_image: bool = True):
        super().__init__()
        self.generator = generator
        self.noise_dim = noise_dim
        self.use_initial_image = use_initial_image

    def forward(self, initial_images: torch.Tensor, conditions: torch.Tensor):
        batch_size = conditions.size(0)
        noise = torch.randn(batch_size, self.noise_dim, device=conditions.device, dtype=conditions.dtype)
        init_img = initial_images if self.use_initial_image else None
        return self.generator(noise, conditions, init_img)


def save_gan_checkpoint(
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    epoch: int,
    loss: float,
    filename: str,
    scheduler_g=None,
    scheduler_d=None
):
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict()
    }

    if scheduler_g is not None:
        checkpoint['scheduler_g_state_dict'] = scheduler_g.state_dict()
    if scheduler_d is not None:
        checkpoint['scheduler_d_state_dict'] = scheduler_d.state_dict()

    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def load_gan_checkpoint(
    filename: str,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: optim.Optimizer = None,
    optimizer_d: optim.Optimizer = None,
    scheduler_g=None,
    scheduler_d=None
):
    checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint['model_state_dict'])

    if 'discriminator_state_dict' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    else:
        print("Warning: discriminator state not found in checkpoint; using current weights.")

    if optimizer_g is not None:
        key = 'optimizer_g_state_dict' if 'optimizer_g_state_dict' in checkpoint else 'optimizer_state_dict'
        if key in checkpoint:
            optimizer_g.load_state_dict(checkpoint[key])
    if optimizer_d is not None:
        if 'optimizer_d_state_dict' in checkpoint:
            optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

    if scheduler_g is not None and 'scheduler_g_state_dict' in checkpoint:
        scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
    if scheduler_d is not None and 'scheduler_d_state_dict' in checkpoint:
        scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)

    print(f"Checkpoint loaded: {filename} (epoch {epoch})")
    return epoch, loss


def setup_ddp(rank: int, world_size: int):
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def train_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader,
    criterion_mse: nn.Module,
    criterion_l1: nn.Module,
    criterion_perceptual: nn.Module,
    criterion_adv: nn.Module,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    device: str,
    scaler_g: GradScaler,
    scaler_d: GradScaler,
    epoch: int,
    gradient_accumulation_steps: int,
    rank: int,
    use_perceptual: bool = True,
    adv_weight: float = 1.0,
):
    generator.train()
    discriminator.train()

    loss_meter = AverageMeter()
    mse_meter = AverageMeter()
    l1_meter = AverageMeter()
    perceptual_meter = AverageMeter()
    adv_meter = AverageMeter()
    disc_meter = AverageMeter()

    total_batches = len(train_loader)

    if rank == 0:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    else:
        pbar = train_loader

    optimizer_g.zero_grad()
    optimizer_d.zero_grad()

    for batch_idx, batch_data in enumerate(pbar):
        initial_images, conditions, target_images = batch_data[0], batch_data[1], batch_data[2]

        initial_images = initial_images.to(device)
        conditions = conditions.to(device)
        target_images = target_images.to(device)

        # -------------------------
        # Train Discriminator
        # -------------------------
        with autocast(device_type='cuda', enabled=cfg.USE_AMP):
            with torch.no_grad():
                fake_images = generator(initial_images, conditions)

            real_preds, _ = discriminator(target_images, conditions)
            fake_preds, _ = discriminator(fake_images.detach(), conditions)

            real_labels = torch.ones_like(real_preds)
            fake_labels = torch.zeros_like(fake_preds)

            d_loss_real = criterion_adv(real_preds, real_labels)
            d_loss_fake = criterion_adv(fake_preds, fake_labels)
            disc_loss = 0.5 * (d_loss_real + d_loss_fake)
            disc_loss = disc_loss / gradient_accumulation_steps

        scaler_d.scale(disc_loss).backward()

        step_discriminator = ((batch_idx + 1) % gradient_accumulation_steps == 0) or ((batch_idx + 1) == total_batches)
        if step_discriminator:
            scaler_d.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), cfg.GRADIENT_CLIP)
            scaler_d.step(optimizer_d)
            scaler_d.update()
            optimizer_d.zero_grad()

        # -------------------------
        # Train Generator
        # -------------------------
        with autocast(device_type='cuda', enabled=cfg.USE_AMP):
            generated_images = generator(initial_images, conditions)

            mse_loss = criterion_mse(generated_images, target_images)
            l1_loss = criterion_l1(generated_images, target_images)
            recon_loss = cfg.RECONSTRUCTION_WEIGHT * mse_loss + cfg.L1_WEIGHT * l1_loss

            perceptual_loss = torch.tensor(0.0, device=device)
            if use_perceptual and criterion_perceptual is not None:
                perceptual_loss = criterion_perceptual(generated_images, target_images)
                recon_loss += cfg.PERCEPTUAL_WEIGHT * perceptual_loss

            adv_preds, _ = discriminator(generated_images, conditions)
            adv_labels = torch.ones_like(adv_preds)
            adv_loss = criterion_adv(adv_preds, adv_labels)

            total_loss = recon_loss + adv_weight * adv_loss
            total_loss = total_loss / gradient_accumulation_steps

        scaler_g.scale(total_loss).backward()

        step_generator = ((batch_idx + 1) % gradient_accumulation_steps == 0) or ((batch_idx + 1) == total_batches)
        if step_generator:
            scaler_g.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg.GRADIENT_CLIP)
            scaler_g.step(optimizer_g)
            scaler_g.update()
            optimizer_g.zero_grad()

        batch_size = initial_images.size(0)
        loss_meter.update(total_loss.item() * gradient_accumulation_steps, batch_size)
        mse_meter.update(mse_loss.item(), batch_size)
        l1_meter.update(l1_loss.item(), batch_size)
        adv_meter.update(adv_loss.item(), batch_size)
        disc_meter.update(disc_loss.item() * gradient_accumulation_steps, batch_size)
        if use_perceptual:
            perceptual_meter.update(perceptual_loss.item(), batch_size)

        if batch_idx % cfg.EMPTY_CACHE_FREQUENCY == 0:
            torch.cuda.empty_cache()

        if rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix({
                'g_loss': f'{loss_meter.avg:.4f}',
                'd_loss': f'{disc_meter.avg:.4f}',
                'adv': f'{adv_meter.avg:.4f}',
                'mse': f'{mse_meter.avg:.4f}',
                'l1': f'{l1_meter.avg:.4f}'
            })

    return {
        'loss': loss_meter.avg,
        'mse': mse_meter.avg,
        'l1': l1_meter.avg,
        'perceptual': perceptual_meter.avg if use_perceptual else 0.0,
        'adv': adv_meter.avg,
        'disc_loss': disc_meter.avg
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
):
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

        generated_images = model(initial_images, conditions)

        mse_loss = criterion_mse(generated_images, target_images)
        l1_loss = criterion_l1(generated_images, target_images)
        total_loss = cfg.RECONSTRUCTION_WEIGHT * mse_loss + cfg.L1_WEIGHT * l1_loss

        perceptual_loss = torch.tensor(0.0, device=device)
        if use_perceptual and criterion_perceptual is not None:
            perceptual_loss = criterion_perceptual(generated_images, target_images)
            total_loss += cfg.PERCEPTUAL_WEIGHT * perceptual_loss

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
    use_initial_image = cfg.GAN_USE_INITIAL_IMAGE and (not args.no_initial_image)

    if world_size > 1:
        setup_ddp(rank, world_size)

    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.sample_dir, exist_ok=True)
        print(f"\n{'='*80}")
        print("GAN Training Configuration")
        print(f"{'='*80}")
        print(f"GPUs: {world_size}")
        print(f"Input size: {args.input_size}x{args.input_size}")
        print(f"Output size: {args.output_size}x{args.output_size}")
        print(f"Noise dim: {args.noise_dim}")
        print(f"Use initial image: {use_initial_image}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation}")
        print(f"Adv. weight: {args.adv_weight}")
        print(f"LR (G/D): {args.lr} / {args.disc_lr}")
        print(f"{'='*80}\n")

    if rank == 0:
        print("Loading datasets...")

    train_loader, val_loader, _ = get_dataloaders(
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
        print("Creating model...")

    generator = GANGenerator(
        num_conditions=args.num_conditions,
        noise_dim=args.noise_dim,
        embed_dim=args.embed_dim,
        embed_out_dim=args.embed_out_dim,
        channels=args.channels,
        use_initial_image=use_initial_image,
        encoder_checkpoint=cfg.ENCODER_CHECKPOINT,
        freeze_encoder=True,
        input_size=args.input_size,
        device=f'cuda:{rank}'
    ).to(device)

    model = GANTrainingWrapper(generator, args.noise_dim, use_initial_image).to(device)
    discriminator = GANDiscriminator(
        num_conditions=args.num_conditions,
        channels=args.channels,
        embed_dim=args.embed_dim,
        embed_out_dim=args.embed_out_dim
    ).to(device)

    if world_size > 1 and cfg.SYNC_BATCHNORM:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        discriminator = DDP(discriminator, device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        gen_params = count_parameters(model.module if world_size > 1 else model)
        disc_params = count_parameters(discriminator.module if world_size > 1 else discriminator)
        print(f"✓ Generator parameters: {gen_params:,}")
        print(f"✓ Discriminator parameters: {disc_params:,}")

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = None
    criterion_adv = nn.BCEWithLogitsLoss()

    if args.use_perceptual:
        from models.conditional import PerceptualLoss
        criterion_perceptual = PerceptualLoss().to(device)
        if rank == 0:
            print("✓ Using perceptual loss")

    optimizer_g = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    optimizer_d = optim.AdamW(
        discriminator.parameters(),
        lr=args.disc_lr,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    def lr_lambda(epoch):
        if epoch < cfg.WARMUP_EPOCHS:
            return (epoch + 1) / cfg.WARMUP_EPOCHS
        return 0.5 * (1 + np.cos(np.pi * (epoch - cfg.WARMUP_EPOCHS) / max(1, args.epochs - cfg.WARMUP_EPOCHS)))

    scheduler_g = optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda)
    scheduler_d = optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda)
    scaler_g = GradScaler(device='cuda', enabled=cfg.USE_AMP)
    scaler_d = GradScaler(device='cuda', enabled=cfg.USE_AMP)
    early_stopping = EarlyStopping(patience=cfg.EARLY_STOP_PATIENCE)

    if rank == 0:
        metrics_logger = MetricsLogger(args.log_dir, cfg.METRICS_FILE)
        print(f"✓ Metrics saved to: {os.path.join(args.log_dir, cfg.METRICS_FILE)}")

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"✓ Resuming from: {args.resume}")
        start_epoch, best_val_loss = load_gan_checkpoint(
            args.resume,
            model.module if world_size > 1 else model,
            discriminator.module if world_size > 1 else discriminator,
            optimizer_g,
            optimizer_d,
            scheduler_g,
            scheduler_d
        )
        start_epoch += 1

    if rank == 0:
        print("\nStarting GAN training...\n")

    last_val_loss = best_val_loss
    last_trained_epoch = start_epoch - 1

    for epoch in range(start_epoch, args.epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        train_metrics = train_epoch(
            model, discriminator, train_loader, criterion_mse, criterion_l1, criterion_perceptual,
            criterion_adv, optimizer_g, optimizer_d, device, scaler_g, scaler_d,
            epoch, args.gradient_accumulation, rank, args.use_perceptual, args.adv_weight
        )

        should_validate = (epoch + 1) % cfg.VALIDATION_FREQUENCY == 0 or (epoch + 1) == args.epochs

        if should_validate:
            val_metrics = validate(
                model, val_loader, criterion_mse, criterion_l1, criterion_perceptual,
                device, epoch, rank, args.use_perceptual
            )
        else:
            val_metrics = None

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            if should_validate:
                metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'train_mse': train_metrics['mse'],
                    'train_l1': train_metrics['l1'],
                    'train_perceptual': train_metrics['perceptual'],
                    'train_adv': train_metrics['adv'],
                    'disc_loss': train_metrics['disc_loss'],
                    'val_loss': val_metrics['loss'],
                    'val_mse': val_metrics['mse'],
                    'val_l1': val_metrics['l1'],
                    'val_perceptual': val_metrics['perceptual'],
                    'val_psnr': val_metrics['psnr'],
                    'val_ssim': val_metrics['ssim'],
                    'learning_rate': optimizer_g.param_groups[0]['lr']
                }
                metrics_logger.log(metrics)

                if cfg.SAVE_SAMPLES:
                    sample_initials, sample_generated, sample_targets = val_metrics['samples']
                    sample_path = os.path.join(args.sample_dir, f'gan_epoch_{epoch+1:04d}.png')
                    visualize_results(
                        sample_initials, sample_generated, sample_targets,
                        num_samples=cfg.NUM_SAMPLE_IMAGES, save_path=sample_path
                    )

                print(f"\n{'='*80}")
                print(
                    f"Epoch {epoch+1}/{args.epochs} - "
                    f"G Loss: {train_metrics['loss']:.4f} | D Loss: {train_metrics['disc_loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f}"
                )
                print(
                    f"PSNR: {val_metrics['psnr']:.2f} dB | SSIM: {val_metrics['ssim']:.4f} | "
                    f"LR (G/D): {optimizer_g.param_groups[0]['lr']:.6f} / {optimizer_d.param_groups[0]['lr']:.6f}"
                )
                print(f"{'='*80}\n")

                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_model_path = os.path.join(args.checkpoint_dir, 'gan_best_model.pth')
                    save_gan_checkpoint(
                        model.module if world_size > 1 else model,
                        discriminator.module if world_size > 1 else discriminator,
                        optimizer_g, optimizer_d, epoch, val_metrics['loss'],
                        best_model_path, scheduler_g, scheduler_d
                    )
                    print("✓ Best GAN model saved")

                last_val_loss = val_metrics['loss']

                if early_stopping(val_metrics['loss']):
                    print(f"\n✓ Early stopping at epoch {epoch + 1}")
                    break
            else:
                last_val_loss = train_metrics['loss']
                print(
                    f"Epoch {epoch+1}/{args.epochs} - "
                    f"G Loss: {train_metrics['loss']:.4f} | D Loss: {train_metrics['disc_loss']:.4f}"
                )

            if (epoch + 1) % cfg.SAVE_FREQUENCY == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f'gan_checkpoint_epoch_{epoch+1:04d}.pth')
                loss_to_save = val_metrics['loss'] if should_validate else train_metrics['loss']
                save_gan_checkpoint(
                    model.module if world_size > 1 else model,
                    discriminator.module if world_size > 1 else discriminator,
                    optimizer_g, optimizer_d, epoch, loss_to_save,
                    checkpoint_path, scheduler_g, scheduler_d
                )
                print("✓ GAN checkpoint saved")

        if world_size > 1:
            dist.barrier()

        torch.cuda.empty_cache()
        last_trained_epoch = epoch

    if rank == 0:
        final_epoch = last_trained_epoch if last_trained_epoch >= 0 else max(start_epoch - 1, 0)
        final_model_path = os.path.join(args.checkpoint_dir, 'gan_final_model.pth')
        save_gan_checkpoint(
            model.module if world_size > 1 else model,
            discriminator.module if world_size > 1 else discriminator,
            optimizer_g, optimizer_d, final_epoch, last_val_loss,
            final_model_path, scheduler_g, scheduler_d
        )
        print("\n✓ GAN training completed!")
        print(f"✓ Final GAN model saved: {final_model_path}")
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
