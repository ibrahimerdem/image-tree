#!/usr/bin/env python3
"""
Pretrained Encoder - Load and use encoder_epoch_50.pth
Encodes 128x128 images into feature representations
Outputs: global features (512D) and local features (4x4 spatial map)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import os


class CustomEncoder(nn.Module):
    """Pretrained encoder for feature extraction"""

    def __init__(self, input_channels=3, feature_dim=512):
        super(CustomEncoder, self).__init__()
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(input_channels, 64, 4, 2, 1)  # 128 -> 64
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)  # 64 -> 32
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)  # 32 -> 16
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)  # 16 -> 8
        self.bn4 = nn.BatchNorm2d(512)
        
        self.conv5 = nn.Conv2d(512, 1024, 4, 2, 1)  # 8 -> 4
        self.bn5 = nn.BatchNorm2d(1024)
        
        # Global and local feature extractors
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Linear(1024, feature_dim)
        self.local_conv = nn.Conv2d(1024, feature_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder
        
        Args:
            x: Input image tensor [B, 3, 128, 128]
            
        Returns:
            Dict with:
            - 'global': Global features [B, 512]
            - 'local': Local feature map [B, 512, 4, 4]
            - 'skip_64': Skip connection [B, 64, 64, 64]
            - 'skip_32': Skip connection [B, 128, 32, 32]
            - 'skip_16': Skip connection [B, 256, 16, 16]
            - 'skip_8': Skip connection [B, 512, 8, 8]
        """
        # Encoder forward pass
        x1 = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)  # 64x64x64
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)), 0.2)  # 32x32x128
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)), 0.2)  # 16x16x256
        x4 = F.leaky_relu(self.bn4(self.conv4(x3)), 0.2)  # 8x8x512
        x5 = F.leaky_relu(self.bn5(self.conv5(x4)), 0.2)  # 4x4x1024
        
        # Global features
        global_feat = self.global_pool(x5).view(x5.size(0), -1)
        global_feat = self.global_fc(global_feat)
        
        # Local features
        local_feat = self.local_conv(x5)
        
        return {
            'global': global_feat,
            'local': local_feat,
            'skip_64': x1,
            'skip_32': x2,
            'skip_16': x3,
            'skip_8': x4
        }


class PretrainedEncoderWrapper(nn.Module):
    """Wrapper to load and use pretrained encoder"""
    
    def __init__(self, checkpoint_path: str = 'encoder_epoch_50.pth', device: str = 'cuda:0'):
        """
        Initialize pretrained encoder
        
        Args:
            checkpoint_path: Path to encoder_epoch_50.pth
            device: Device to load model on
        """
        super().__init__()
        self.device = torch.device(device)
        self.encoder = CustomEncoder(input_channels=3, feature_dim=512)
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            print(f"Loading pretrained encoder from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.encoder.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'encoder' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder'])
            else:
                self.encoder.load_state_dict(checkpoint)
            
            print("✓ Encoder loaded successfully")
        else:
            print(f"⚠ Warning: {checkpoint_path} not found")
            print("  Will initialize with random weights")
        
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        
        # Freeze encoder (no training)
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass - encode images using pretrained encoder
        
        Args:
            images: Tensor of shape [B, 3, 128, 128]
            
        Returns:
            Dict with encoded features
        """
        if images.device != self.device:
            images = images.to(self.device)
        
        with torch.no_grad():
            return self.encoder(images)
    
    def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode images using pretrained encoder (alias for forward)
        
        Args:
            images: Tensor of shape [B, 3, 128, 128]
            
        Returns:
            Dict with encoded features
        """
        return self.forward(images)
    
    def get_feature_dim(self) -> int:
        """Get feature dimension (512)"""
        return self.encoder.feature_dim
    
    def get_num_parameters(self) -> int:
        """Get total parameters"""
        return sum(p.numel() for p in self.encoder.parameters())


def test_pretrained_encoder():
    """Test pretrained encoder loading and inference"""
    print("\n" + "="*70)
    print("PRETRAINED ENCODER TEST")
    print("="*70)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Initialize wrapper
    encoder_wrapper = PretrainedEncoderWrapper(
        checkpoint_path='encoder_epoch_50.pth',
        device=device
    )
    
    print(f"\nEncoder Configuration:")
    print(f"  Input size: 128×128")
    print(f"  Feature dimension: {encoder_wrapper.get_feature_dim()}")
    print(f"  Total parameters: {encoder_wrapper.get_num_parameters():,}")
    print(f"  Device: {device}")
    
    # Test encoding
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 128, 128, device=device)
    
    print(f"\nTest Encoding (batch size {batch_size}):")
    features = encoder_wrapper.encode(dummy_images)
    
    print(f"  Global features shape: {features['global'].shape}")
    print(f"  Local features shape: {features['local'].shape}")
    print(f"  Skip features available: {list(k for k in features.keys() if 'skip' in k)}")
    
    # Verify shapes
    assert features['global'].shape == (batch_size, 512), "Global features incorrect shape"
    assert features['local'].shape == (batch_size, 512, 4, 4), "Local features incorrect shape"
    
    print(f"\n✓ TEST PASSED: Pretrained encoder working correctly")
    print("="*70)
    
    return encoder_wrapper


if __name__ == "__main__":
    encoder = test_pretrained_encoder()
