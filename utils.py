"""
Utility functions for training and evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from typing import Dict, List, Tuple


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filename: str,
    scheduler=None
) -> None:
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict':  model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved:  {filename}")


def load_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None
) -> Tuple[int, float]:
    """Load model checkpoint"""
    checkpoint = torch.load(filename, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: {filename} (epoch {epoch})")
    return epoch, loss

def save_gan_checkpoint(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
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
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer = None,
    optimizer_d: torch.optim.Optimizer = None,
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

def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Denormalize image from [-1, 1] to [0, 1]"""
    return (image + 1) / 2


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate PSNR between two images"""
    img1 = denormalize_image(img1).cpu().numpy()
    img2 = denormalize_image(img2).cpu().numpy()
    
    # Convert from [B, C, H, W] to [B, H, W, C]
    img1 = np.transpose(img1, (0, 2, 3, 1))
    img2 = np.transpose(img2, (0, 2, 3, 1))
    
    psnr_values = []
    for i in range(img1.shape[0]):
        psnr_val = psnr(img1[i], img2[i], data_range=1.0)
        psnr_values.append(psnr_val)
    
    return np.mean(psnr_values)


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate SSIM between two images"""
    img1 = denormalize_image(img1).cpu().numpy()
    img2 = denormalize_image(img2).cpu().numpy()
    
    # Convert from [B, C, H, W] to [B, H, W, C]
    img1 = np.transpose(img1, (0, 2, 3, 1))
    img2 = np.transpose(img2, (0, 2, 3, 1))
    
    ssim_values = []
    for i in range(img1.shape[0]):
        ssim_val = ssim(img1[i], img2[i], data_range=1.0, channel_axis=2)
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)


def visualize_results(
    initial_images: torch.Tensor,
    generated_images: torch.Tensor,
    target_images: torch.Tensor,
    num_samples: int = 4,
    save_path: str = None
) -> None:
    """
    Visualize results:  initial, generated, and target images
    
    Args:
        initial_images: Initial input images [B, 3, H, W]
        generated_images: Generated output images [B, 3, H, W]
        target_images: Target ground truth images [B, 3, H, W]
        num_samples: Number of samples to visualize
        save_path: Path to save the figure (if None, display instead)
    """
    num_samples = min(num_samples, initial_images.size(0))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize images
        initial = denormalize_image(initial_images[i]).cpu().permute(1, 2, 0).numpy()
        generated = denormalize_image(generated_images[i]).cpu().permute(1, 2, 0).numpy()
        target = denormalize_image(target_images[i]).cpu().permute(1, 2, 0).numpy()
        
        # Clip to [0, 1]
        initial = np.clip(initial, 0, 1)
        generated = np.clip(generated, 0, 1)
        target = np.clip(target, 0, 1)
        
        # Plot
        axes[i, 0]. imshow(initial)
        axes[i, 0].set_title('Initial Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(generated)
        axes[i, 1].set_title('Generated Image')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(target)
        axes[i, 2]. set_title('Target Image')
        axes[i, 2]. axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self. sum += val * n
        self.count += n
        self. avg = self.sum / self. count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self. best_loss is None: 
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self. min_delta:
            self. counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self. best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__": 
    print("Utils module loaded successfully!")