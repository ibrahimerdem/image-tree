"""
Integrated Conditional Generator Model
Combines conditional_gan.py Generator with conditional.py ConditionalImageGenerator
Without discriminator, optimized for training both architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from .pretrained_encoder import PretrainedEncoderWrapper


class SelfAttention(nn.Module):
    """Self-Attention mechanism for capturing long-range dependencies"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H * W)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        value = self.value(x).view(B, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


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
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    """Deep decoder with ConvTranspose2d upsampling, 7 levels from 4x4 to 512x512"""
    
    def __init__(self, in_channels: int, channels: List[int] = [1024, 512, 256, 128, 64, 32, 16], 
                 out_channels: int = 3, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        
        self.layers = nn.ModuleList()
        prev_channels = in_channels

        for i, out_ch in enumerate(channels):
            deconv_block = nn.Sequential(
                nn.ConvTranspose2d(prev_channels, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.layers.append(deconv_block)

            if use_attention and i == 2:
                self.layers.append(SelfAttention(out_ch))

            self.layers.append(ResidualBlock(out_ch))
            prev_channels = out_ch
        
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
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features)[:9]).eval()
        
        for param in self.features.parameters():
            param.requires_grad = False
        
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(generated.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(generated.device)
        
        generated = (generated + 1) / 2
        target = (target + 1) / 2
        
        generated = (generated - mean) / std
        target = (target - mean) / std
        
        gen_features = self.features(generated)
        target_features = self.features(target)
        
        return F.mse_loss(gen_features, target_features)


class IntegratedConditionalGenerator(nn.Module):
    """
    Integrated Conditional Generator
    Combines GAN generator architecture with pretrained encoder approach
    Supports both noise-based and image-based conditioning
    
    Features:
    - No discriminator (generator-only)
    - Flexible conditioning: noise, text/features, or initial images
    - Pretrained encoder for initial image features
    - Self-attention mechanisms
    - Optional VAE latent space
    - Appropriate for supervised training without adversarial loss
    """
    
    def __init__(
        self,
        conf: Dict = None,
        num_conditions: int = 9,
        input_size: int = 128,
        output_size: int = 512,
        noise_dim: int = 100,
        embed_dim: int = 256,
        embed_out_dim: int = 128,
        latent_dim: int = 256,
        decoder_channels: List[int] = None,
        condition_hidden_dims: List[int] = None,
        use_vae: bool = True,
        initial_image: bool = False,
        image_encoder_path: str = None,
        device: str = 'cuda:0'
    ):
        """
        Args:
            conf: Configuration dictionary (optional, for compatibility with conditional_gan)
            num_conditions: Number of conditional features
            input_size: Input image size (for pretrained encoder)
            output_size: Output image size
            noise_dim: Noise vector dimension
            embed_dim: Initial embedding dimension
            embed_out_dim: Output embedding dimension
            latent_dim: Latent space dimension
            decoder_channels: Channels for decoder upsampling
            condition_hidden_dims: Hidden dimensions for condition encoder
            use_vae: Whether to use VAE latent space
            initial_image: Whether to use initial images as input
            image_encoder_path: Path to pretrained image encoder
            device: Device to use ('cuda:0', 'cpu', etc.)
        """
        super().__init__()
        
        if conf is not None:
            num_conditions = len(conf.get("dataset", {}).get("input_features", "").split(","))
            output_size = int(conf.get("image", {}).get("output_size", 512))
            noise_dim = int(conf.get("model", {}).get("noise_dim", 100))
            embed_dim = int(conf.get("model", {}).get("embed_dim", 256))
            embed_out_dim = int(conf.get("model", {}).get("embed_out_dim", 128))
        
        self.num_conditions = num_conditions
        self.input_size = input_size
        self.output_size = output_size
        self.noise_dim = noise_dim
        self.embed_dim = embed_dim
        self.embed_out_dim = embed_out_dim
        self.latent_dim = latent_dim
        self.use_vae = use_vae
        self.device = device
        
        if decoder_channels is None:
            decoder_channels = [1024, 512, 256, 128, 64, 32, 16]
        if condition_hidden_dims is None:
            condition_hidden_dims = [64, 128, 256]
        
        self.decoder_channels = decoder_channels
        self.condition_hidden_dims = condition_hidden_dims
        
        # ============ Condition Processing ============
        # Text/feature embedding (from conditional_gan)
        self.text_embedding = nn.Sequential(
            nn.Linear(num_conditions, embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_out_dim),
            nn.BatchNorm1d(embed_out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Condition encoder (from conditional.py)
        self.condition_encoder = ConditionEncoder(
            num_conditions=num_conditions,
            hidden_dims=condition_hidden_dims
        )
        
        # ============ Image Encoder (Optional) ============
        self.image_encoder = None
        self.initial_image = initial_image
        
        if initial_image:
            if image_encoder_path:
                try:
                    self.image_encoder = PretrainedEncoderWrapper(image_encoder_path, device=device)
                except:
                    print(f"Warning: Could not load pretrained encoder from {image_encoder_path}")
                    self.image_encoder = None
        
        # ============ Noise Processing ============
        # Different FC layers for with/without image features
        self.fc_no_image = nn.Linear(noise_dim + embed_out_dim, 1024 * 4 * 4)
        
        if self.image_encoder:
            # 512 is typical for pretrained encoders
            self.fc_with_image = nn.Linear(noise_dim + embed_out_dim + 512, 1024 * 4 * 4)
        
        # ============ Decoder (7 levels: 4x4 -> 512x512) ============
        self.decoder = Decoder(
            in_channels=1024,
            channels=decoder_channels,
            out_channels=3,
            use_attention=True
        )
        
        # ============ VAE Components ============
        if self.use_vae:
            self.vae_latent_dim = latent_dim
            self.mu_fc = nn.Linear(1024 * 4 * 4, latent_dim)
            self.logvar_fc = nn.Linear(1024 * 4 * 4, latent_dim)
            # Remap latent back to spatial
            self.vae_to_spatial = nn.Linear(latent_dim, 1024 * 4 * 4)
    
    def freeze_image_encoder(self):
        """Freeze image encoder parameters"""
        if self.image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            print("Image encoder frozen")
    
    def unfreeze_image_encoder(self):
        """Unfreeze image encoder for fine-tuning"""
        if self.image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = True
            print("Image encoder unfrozen for fine-tuning")
    
    def forward(
        self, 
        conditions: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        initial_image: Optional[torch.Tensor] = None,
        return_vae_params: bool = False
    ) -> torch.Tensor:
        """
        Forward pass supporting multiple conditioning modes
        
        Args:
            conditions: Conditional features [B, num_conditions]
            noise: Noise vector [B, noise_dim]. If None, random noise is generated
            initial_image: Initial image [B, 3, H, W] (optional)
            return_vae_params: Whether to return VAE parameters (mu, logvar)
            
        Returns:
            Generated image [B, 3, output_size, output_size]
            If return_vae_params: (image, mu, logvar)
        """
        batch_size = conditions.shape[0]
        
        # Generate random noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=conditions.device)
        
        # Embed conditions using text embedding (GAN-style)
        text_emb = self.text_embedding(conditions)  # [B, embed_out_dim]
        noise_flat = noise.view(batch_size, -1)  # [B, noise_dim]
        
        # Handle initial image features if provided
        image_features = None
        if initial_image is not None and self.image_encoder is not None:
            # Resize if needed
            if initial_image.shape[-1] != self.input_size or initial_image.shape[-2] != self.input_size:
                initial_image = F.interpolate(
                    initial_image, 
                    size=(self.input_size, self.input_size),
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Get image features (frozen encoder)
            encoded = self.image_encoder.encode(initial_image)
            global_feat = encoded.get('global', encoded) if isinstance(encoded, dict) else encoded
            if isinstance(global_feat, dict):
                global_feat = global_feat.get('global', global_feat)
            
            image_features = global_feat
            combined_features = torch.cat([noise_flat, text_emb, image_features], dim=1)
            z = self.fc_with_image(combined_features)
        else:
            combined_features = torch.cat([noise_flat, text_emb], dim=1)
            z = self.fc_no_image(combined_features)
        
        # Reshape to spatial
        z = z.view(batch_size, 1024, 4, 4)  # [B, 1024, 4, 4]
        
        # ============ VAE Reparameterization (Optional) ============
        mu = None
        logvar = None
        if self.use_vae:
            z_flat = z.view(batch_size, -1)  # [B, 16384]
            mu = self.mu_fc(z_flat)  # [B, latent_dim]
            logvar = self.logvar_fc(z_flat)  # [B, latent_dim]
            
            if self.training:
                # Reparameterize during training
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z_sample = mu + eps * std
            else:
                # Deterministic during inference
                z_sample = mu
            
            # Map back to spatial
            z = self.vae_to_spatial(z_sample)
            z = z.view(batch_size, 1024, 4, 4)
        
        # ============ Decode to output size ============
        output = self.decoder(z)
        
        if return_vae_params and self.use_vae:
            return output, mu, logvar
        
        return output
    
    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """Compute KL divergence between N(mu, sigma) and N(0, I)"""
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if reduction == 'sum':
            return kl.sum()
        elif reduction == 'mean':
            return kl.mean()
        else:
            return kl
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen
        }


class TrainingModule(nn.Module):
    """
    Training wrapper for IntegratedConditionalGenerator
    Handles loss computation and training logic for both architectures
    """
    
    def __init__(
        self,
        generator: IntegratedConditionalGenerator,
        reconstruction_weight: float = 0.2,
        l1_weight: float = 2.0,
        perceptual_weight: float = 0.8,
        vae_kl_weight: float = 0.1,
        device: str = 'cuda:0'
    ):
        """
        Args:
            generator: IntegratedConditionalGenerator instance
            reconstruction_weight: MSE loss weight
            l1_weight: L1 loss weight
            perceptual_weight: Perceptual loss weight
            vae_kl_weight: VAE KL divergence weight
            device: Device to use
        """
        super().__init__()
        self.generator = generator
        self.device = device
        
        self.reconstruction_weight = reconstruction_weight
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.vae_kl_weight = vae_kl_weight
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Perceptual loss (only if weight > 0)
        self.perceptual_loss = None
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss().to(device)
    
    def compute_loss(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss combining reconstruction, L1, perceptual, and KL losses
        
        Args:
            generated: Generated images [B, 3, H, W]
            target: Target images [B, 3, H, W]
            mu: VAE mean (optional)
            logvar: VAE logvar (optional)
            
        Returns:
            Dict with loss components and total loss
        """
        losses = {}
        
        # Reconstruction loss (MSE)
        mse = self.mse_loss(generated, target)
        losses['mse'] = mse
        total_loss = self.reconstruction_weight * mse
        
        # L1 loss
        l1 = self.l1_loss(generated, target)
        losses['l1'] = l1
        total_loss += self.l1_weight * l1
        
        # Perceptual loss
        if self.perceptual_loss is not None:
            perc = self.perceptual_loss(generated, target)
            losses['perceptual'] = perc
            total_loss += self.perceptual_weight * perc
        
        # VAE KL divergence
        if mu is not None and logvar is not None:
            kl = IntegratedConditionalGenerator.kl_divergence(mu, logvar, reduction='mean')
            losses['kl'] = kl
            total_loss += self.vae_kl_weight * kl
        
        losses['total'] = total_loss
        return losses
    
    def forward(
        self,
        conditions: torch.Tensor,
        target: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        initial_image: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass
        
        Args:
            conditions: Conditional features [B, num_conditions]
            target: Target images [B, 3, H, W]
            noise: Noise vector [B, noise_dim] (optional)
            initial_image: Initial images [B, 3, H, W] (optional)
            
        Returns:
            Loss dictionary
        """
        # Generate
        if self.generator.use_vae:
            generated, mu, logvar = self.generator(
                conditions, 
                noise, 
                initial_image,
                return_vae_params=True
            )
        else:
            generated = self.generator(conditions, noise, initial_image)
            mu = None
            logvar = None
        
        # Compute losses
        losses = self.compute_loss(generated, target, mu, logvar)
        
        return losses


def test_model():
    """Test integrated model"""
    print("\nTesting Integrated Conditional Generator")
    print("=" * 60)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size = 2
    num_conditions = 9
    noise_dim = 100
    
    # Create model
    model = IntegratedConditionalGenerator(
        num_conditions=num_conditions,
        noise_dim=noise_dim,
        output_size=512,
        use_vae=True,
        initial_image=False,
        device=device
    ).to(device)
    
    # Create training module
    trainer = TrainingModule(model, device=device)
    
    # Dummy data
    conditions = torch.randn(batch_size, num_conditions, device=device)
    target = torch.randn(batch_size, 3, 512, 512, device=device)
    
    # Forward pass
    losses = trainer(conditions, target)
    
    print(f"\nModel Architecture:")
    print(f"  Conditions: {num_conditions}")
    print(f"  Noise dim: {noise_dim}")
    print(f"  Output size: 512x512")
    print(f"  Use VAE: {model.use_vae}")
    
    print(f"\nParameter counts: {model.get_num_parameters()}")
    
    print(f"\nLosses:")
    for name, loss in losses.items():
        print(f"  {name}: {loss.item():.4f}")
    
    print("\nModel tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
