"""
Conditional Image Generator Model - Uses Pretrained Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
import config as cfg
from pretrained_encoder import PretrainedEncoderWrapper
import os


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn. Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ImageEncoder(nn.Module):
    """
    Wrapper for pretrained encoder (encoder_epoch_50.pth)
    Takes 128x128 images and outputs features
    """
    
    def __init__(self, checkpoint_path: str = 'encoder_epoch_50.pth', device: str = 'cuda:0'):
        super().__init__()
        self.encoder_wrapper = PretrainedEncoderWrapper(checkpoint_path, device)
        self.device = device
        
        # Freeze encoder (no training)
        for param in self.encoder_wrapper.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through pretrained encoder
        
        Args:
            x: Input images [B, 3, 128, 128]
            
        Returns:
            Dict with global and local features
        """
        # Ensure input is on correct device
        if x.device != self.device:
            x = x.to(self.device)
        
        with torch.no_grad():
            features = self.encoder_wrapper.encode(x)
        
        return features


class ConditionEncoder(nn.Module):
    """Lightweight condition encoder"""
    
    def __init__(self, num_conditions: int, hidden_dims: List[int] = [64, 128, 256]):
        super().__init__()
        
        layers = []
        prev_dim = num_conditions
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn. Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SelfAttention(nn.Module):
    """Self-Attention mechanism for capturing long-range dependencies"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Output projection
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable weight
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            out: Self-attention output [B, C, H, W]
        """
        batch_size, C, H, W = x.size()
        
        # Query: [B, C', H, W] -> [B, C', H*W] -> [B, H*W, C']
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        
        # Key: [B, C', H, W] -> [B, C', H*W]
        key = self.key(x).view(batch_size, -1, H * W)
        
        # Attention map: [B, H*W, H*W]
        attention = self.softmax(torch.bmm(query, key))
        
        # Value: [B, C, H, W] -> [B, C, H*W]
        value = self.value(x).view(batch_size, -1, H * W)
        
        # Attended features: [B, C, H*W] -> [B, C, H, W]
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


class Decoder(nn.Module):
    """Deep decoder with ConvTranspose2d upsampling, 7 levels from 4x4 to 512x512"""
    
    def __init__(self, in_channels: int, channels: List[int] = [1024, 512, 256, 128, 64, 32, 16], out_channels: int = 3, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        
        # Decoder progression: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256 -> 512x512
        # 7 levels of deconv (2^7 = 128x total upsampling)
        self.layers = nn.ModuleList()
        prev_channels = in_channels

        for i, out_ch in enumerate(channels):
            # ConvTranspose2d: kernel_size=4, stride=2, padding=1 for 2x spatial upsampling
            deconv_block = nn.Sequential(
                nn.ConvTranspose2d(prev_channels, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.layers.append(deconv_block)

            # Add self-attention after 3rd upsampling (at 32x32 resolution)
            if use_attention and i == 2:  # After 3rd deconv layer
                self.layers.append(SelfAttention(out_ch))

            # Add residual block for all layers to maintain detail and prevent artifacts
            self.layers.append(ResidualBlock(out_ch))

            prev_channels = out_ch
        
        # Final layer to output resolution (512x512) with Conv2d
        self.final_layer = nn.Sequential(
            nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features for better image quality"""
    
    def __init__(self):
        super().__init__()
        # Use VGG16 features with modern weights API
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features)[:9]).eval()
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(generated.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(generated.device)
        
        generated = (generated + 1) / 2  # [-1, 1] -> [0, 1]
        target = (target + 1) / 2
        
        generated = (generated - mean) / std
        target = (target - mean) / std
        
        gen_features = self.features(generated)
        target_features = self.features(target)
        
        return F.mse_loss(gen_features, target_features)


class ConditionalImageGenerator(nn.Module):
    """
    Conditional image generator with 4x upscaling
    Input: 128x128, Output: 512x512 (4x upscaling)
    Optimized for 2x36GB GPUs
    """
    
    def __init__(
        self,
        num_conditions: int,
        input_size: int = 128,
        output_size: int = 512,
        latent_dim: int = 256,
        encoder_channels: List[int] = [64, 128, 256, 256, 256],
        decoder_channels: List[int] = [1024, 512, 256, 128, 64, 32, 16],
        condition_hidden_dims:  List[int] = [64, 128, 256],
        use_vae: bool = False,
        use_checkpoint: bool = False,
        encoder_checkpoint: str = 'encoder_epoch_50.pth',
        device: str = 'cuda:0'
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.latent_dim = latent_dim
        self.num_conditions = num_conditions
        self.use_checkpoint = use_checkpoint
        self.use_vae = use_vae
        self.vae_latent_dim = latent_dim
        self.device = device
        
        # Pretrained encoder takes 128x128 images
        # If input_size > 128, we'll resize during forward pass
        self.encoder_input_size = 128
        
        # Load pretrained encoder (frozen, no gradients)
        self.image_encoder = PretrainedEncoderWrapper(encoder_checkpoint, device=device)
        
        # Condition encoder (trainable)
        self.condition_encoder = ConditionEncoder(
            num_conditions=num_conditions,
            hidden_dims=condition_hidden_dims
        )
        
        # Spatial size after pretrained encoder: 128 -> 4x4 local features
        # Global features: 512D vector
        self.spatial_size = 4
        
        # Fusion layer: combine global features with conditions
        # Global features: 512D, Conditions: 256D
        concat_channels = 512 + condition_hidden_dims[-1]
        self.fusion_global = nn.Sequential(
            nn.Linear(concat_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
        
        # Expand global features to 4x4 spatial map for decoder
        self.expand_spatial = nn.Sequential(
            nn.Linear(512, 256 * 4 * 4),
            nn.ReLU(inplace=True)
        )

        # VAE heads (spatial latent map) - optional
        if self.use_vae:
            self.mu_conv = nn.Conv2d(256, self.vae_latent_dim, kernel_size=1)
            self.logvar_conv = nn.Conv2d(256, self.vae_latent_dim, kernel_size=1)
            decoder_in_channels = 256 + self.vae_latent_dim
        else:
            decoder_in_channels = 256
        
        # Decoder with self-attention
        self.decoder = Decoder(
            in_channels=decoder_in_channels,
            channels=decoder_channels,
            out_channels=3,
            use_attention=True
        )
        
    def forward(self, image: torch.Tensor, conditions: torch.Tensor) -> Tuple:
        """
        Forward pass using pretrained encoder
        
        Args:
            image: Input image tensor [B, 3, H, W] (will be resized to 128x128 for encoder)
            conditions: Condition features [B, num_conditions]
            
        Returns:
            Generated image [B, 3, 512, 512]
            If VAE: (image, mu, logvar)
        """
        batch_size = image.size(0)
        
        # Resize image to 128x128 if needed (pretrained encoder expects 128x128)
        if image.shape[-1] != self.encoder_input_size or image.shape[-2] != self.encoder_input_size:
            image = F.interpolate(image, size=(self.encoder_input_size, self.encoder_input_size), 
                                mode='bilinear', align_corners=False)
        
        # Encode image with pretrained encoder (no gradients)
        encoded_features = self.image_encoder(image)
        
        # Extract global and local features
        global_feat = encoded_features['global']  # [B, 512]
        local_feat = encoded_features['local']    # [B, 512, 4, 4]
        
        # Encode conditions
        encoded_conditions = self.condition_encoder(conditions)  # [B, 256]
        
        # Fuse global features with conditions
        combined = torch.cat([global_feat, encoded_conditions], dim=1)  # [B, 768]
        fused_global = self.fusion_global(combined)  # [B, 512]
        
        # Expand to spatial map for decoder input
        spatial_feat = self.expand_spatial(fused_global)  # [B, 256*16]
        spatial_feat = spatial_feat.view(batch_size, 256, 4, 4)  # [B, 256, 4, 4]

        # If using VAE, compute mu/logvar and sample spatial latent
        if self.use_vae and self.training:
            mu = self.mu_conv(spatial_feat)
            logvar = self.logvar_conv(spatial_feat)
            z = self.reparameterize(mu, logvar)
            decoder_input = torch.cat([spatial_feat, z], dim=1)
        elif self.use_vae and not self.training:
            mu = self.mu_conv(spatial_feat)
            logvar = self.logvar_conv(spatial_feat)
            z = mu  # Deterministic in eval mode
            decoder_input = torch.cat([spatial_feat, z], dim=1)
        else:
            decoder_input = spatial_feat
            mu = None
            logvar = None

        # Decode to 512x512
        output_image = self.decoder(decoder_input)

        # If VAE enabled, return (output, mu, logvar) for KL computation
        if self.use_vae:
            return output_image, mu, logvar

        return output_image

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling z from mu/logvar (spatial map)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """Compute KL divergence between N(mu, sigma) and N(0, I).

        Returns a scalar tensor.
        """
        # KL per element
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if reduction == 'sum':
            return kl.sum()
        elif reduction == 'mean':
            return kl.sum(dim=[1, 2, 3]).mean()
        else:
            return kl
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """Test model with 256->512 upscaling - GPU safe, no CPU usage"""
    print(f"\nTesting Image Generator Model")
    print("="*60)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPUs available! This test requires GPU.")
        return
    
    device = torch.device('cuda:0')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    
    input_size = 128
    output_size = 512  # 4x upscaling
    batch_size = 2
    num_conditions = 10
    
    # Create model and move to GPU immediately
    model = ConditionalImageGenerator(
        num_conditions=num_conditions,
        input_size=input_size,
        output_size=output_size,
        encoder_channels=[64, 128, 256, 256, 256],
        decoder_channels=[1024, 512, 256, 128, 64, 32, 16],
        condition_hidden_dims=[64, 128, 256],
        use_checkpoint=False
    ).to(device)
    
    
    # Create dummy inputs directly on GPU
    with torch.no_grad():
        dummy_image = torch.randn(batch_size, 3, input_size, input_size, device=device)
        dummy_conditions = torch.randn(batch_size, num_conditions, device=device)
        
        # Forward pass
        output = model(dummy_image, dummy_conditions)
        if isinstance(output, tuple):
            output_image = output[0]
        else:
            output_image = output
    
    print(f"\nModel Architecture:")
    print(f"  Input size:  {input_size}x{input_size}")
    print(f"  Output size: {output_size}x{output_size}")
    print(f"  Upscaling factor: {output_size/input_size}x")
    print(f"  Conditions: {num_conditions}")
    
    print(f"\nTest Results:")
    print(f"  Input shape:  {tuple(dummy_image.shape)}")
    print(f"  Output shape: {tuple(output_image.shape)}")
    print(f"  Total parameters: {model.get_num_parameters():,}")
    
    # Calculate memory usage
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    print(f"  Model memory: {param_memory:.2f} MB")
    
    # GPU memory info
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    print(f"  GPU memory allocated: {allocated:.2f} GB")
    print(f"  GPU memory reserved: {reserved:.2f} GB")
    
    # Verify output shape
    expected_shape = (batch_size, 3, output_size, output_size)
    assert output_image.shape == expected_shape, f"Output shape mismatch! Expected {expected_shape}, got {output_image.shape}"
    
    # Cleanup
    del model, dummy_image, dummy_conditions, output, output_image
    torch.cuda.empty_cache()
    
    print("\nModel tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_model()