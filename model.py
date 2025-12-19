"""
Conditional Image Generator Model - Optimized for High Resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


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
    """Memory-efficient image encoder for high-resolution images"""
    
    def __init__(self, in_channels: int = 3, channels: List[int] = [64, 128, 256, 256]):
        super().__init__()
        
        layers = []
        prev_channels = in_channels
        
        for i, out_channels in enumerate(channels):
            # Downsampling layer
            layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            # Add residual block for deeper layers
            if i >= 2:  # Add residual blocks from 3rd layer onwards
                layers.append(ResidualBlock(out_channels))
            
            prev_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


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
    """Memory-efficient decoder with residual connections and self-attention"""
    
    def __init__(self, in_channels: int, channels: List[int] = [256, 256, 128, 64], out_channels: int = 3, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        
        # Build decoder layers
        self.layers = nn.ModuleList()
        prev_channels = in_channels
        
        for i, out_ch in enumerate(channels):
            # Upsampling layer
            upsample_block = nn.Sequential(
                nn.ConvTranspose2d(prev_channels, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.layers.append(upsample_block)
            
            # Add self-attention after first upsampling (at 32x32 or 64x64 resolution)
            if use_attention and i == 1:  # Add attention after 2nd layer
                self.layers.append(SelfAttention(out_ch))
            
            # Add residual block for earlier layers
            if i < len(channels) - 2: 
                self.layers.append(ResidualBlock(out_ch))
            
            prev_channels = out_ch
        
        # Final upsampling to output resolution
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(prev_channels, out_channels, kernel_size=4, stride=2, padding=1),
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
        encoder_channels: List[int] = [64, 128, 256],
        decoder_channels: List[int] = [256, 128, 64, 32],
        condition_hidden_dims:  List[int] = [64, 128, 256],
        use_checkpoint: bool = False
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.latent_dim = latent_dim
        self.num_conditions = num_conditions
        self.use_checkpoint = use_checkpoint
        
        # Calculate spatial size after encoding (128 -> 64 -> 32 -> 16)
        self.spatial_size = input_size // (2 ** len(encoder_channels))
        
        # Encoders
        self.image_encoder = ImageEncoder(in_channels=3, channels=encoder_channels)
        self.condition_encoder = ConditionEncoder(
            num_conditions=num_conditions,
            hidden_dims=condition_hidden_dims
        )
        
        # Fusion layer to reduce concatenated channels
        concat_channels = encoder_channels[-1] + condition_hidden_dims[-1]
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, encoder_channels[-1], 1),
            nn.BatchNorm2d(encoder_channels[-1]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with self-attention
        self.decoder = Decoder(
            in_channels=encoder_channels[-1],
            channels=decoder_channels,
            out_channels=3,
            use_attention=True  # Enable self-attention
        )
        
    def forward(self, image: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            image: Input image tensor [B, 3, 128, 128]
            conditions: Condition features [B, num_conditions]
            
        Returns:
            Generated image [B, 3, 512, 512] (4x upscaled)
        """
        batch_size = image.size(0)
        
        # Encode image with gradient checkpointing if enabled
        if self.use_checkpoint and self.training:
            encoded_image = torch.utils.checkpoint.checkpoint(
                self.image_encoder, image, use_reentrant=False, preserve_rng_state=True
            )
        else:
            encoded_image = self.image_encoder(image)
        
        # Encode conditions
        encoded_conditions = self.condition_encoder(conditions)
        
        # Reshape conditions to match spatial dimensions
        encoded_conditions = encoded_conditions.view(
            batch_size, -1, 1, 1
        ).expand(
            batch_size, -1, self.spatial_size, self.spatial_size
        )
        
        # Concatenate and fuse
        combined = torch.cat([encoded_image, encoded_conditions], dim=1)
        fused = self.fusion(combined)
        
        # Decode with gradient checkpointing if enabled
        if self.use_checkpoint and self.training:
            output_image = torch.utils.checkpoint.checkpoint(
                self.decoder, fused, use_reentrant=False, preserve_rng_state=True
            )
        else:
            output_image = self.decoder(fused)
        
        return output_image
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """Test model with 128->256 upscaling - GPU safe, no CPU usage"""
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
        encoder_channels=[64, 128, 256],
        decoder_channels=[256, 128, 64, 32],
        condition_hidden_dims=[64, 128, 256],
        use_checkpoint=False
    ).to(device)
    
    
    # Create dummy inputs directly on GPU
    with torch.no_grad():
        dummy_image = torch.randn(batch_size, 3, input_size, input_size, device=device)
        dummy_conditions = torch.randn(batch_size, num_conditions, device=device)
        
        # Forward pass
        output = model(dummy_image, dummy_conditions)
    
    print(f"\nModel Architecture:")
    print(f"  Input size:  {input_size}x{input_size}")
    print(f"  Output size: {output_size}x{output_size}")
    print(f"  Upscaling factor: {output_size/input_size}x")
    print(f"  Conditions: {num_conditions}")
    
    print(f"\nTest Results:")
    print(f"  Input shape:  {tuple(dummy_image.shape)}")
    print(f"  Output shape: {tuple(output.shape)}")
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
    assert output.shape == expected_shape, f"Output shape mismatch! Expected {expected_shape}, got {output.shape}"
    
    # Cleanup
    del model, dummy_image, dummy_conditions, output
    torch.cuda.empty_cache()
    
    print("\nModel tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_model()