import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights, VGG19_Weights, EfficientNet_B4_Weights
import clip


class PretrainedImageEncoder(nn.Module):
    """
    Wrapper for various pretrained image encoders
    Supports: ResNet, VGG, EfficientNet, CLIP, Custom pretrained
    """
    def __init__(self, encoder_type='resnet50', feature_dim=512, freeze_backbone=True):
        super(PretrainedImageEncoder, self).__init__()
        self.encoder_type = encoder_type
        self.feature_dim = feature_dim
        self.freeze_backbone = freeze_backbone
        
        if encoder_type == 'resnet50':
            self.backbone = self._create_resnet50_encoder()
        elif encoder_type == 'vgg19':
            self.backbone = self._create_vgg19_encoder()
        elif encoder_type == 'efficientnet':
            self.backbone = self._create_efficientnet_encoder()
        elif encoder_type == 'clip':
            self.backbone = self._create_clip_encoder()
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        # Feature adaptation layers
        self.setup_feature_adaptation()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone_weights()
    
    def _create_resnet50_encoder(self):
        """Create ResNet50-based encoder with skip connections"""
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Extract intermediate layers for skip connections
        self.layer1 = resnet.layer1  # 64 channels, /4 resolution
        self.layer2 = resnet.layer2  # 128 channels, /8 resolution  
        self.layer3 = resnet.layer3  # 256 channels, /16 resolution
        self.layer4 = resnet.layer4  # 512 channels, /32 resolution
        
        # Initial convolution layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Global average pooling and classifier
        self.avgpool = resnet.avgpool
        
        return resnet
    
    def _create_vgg19_encoder(self):
        """Create VGG19-based encoder"""
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        features = vgg.features
        
        # Extract different layers for skip connections
        # VGG19 layer indices for different resolutions
        self.vgg_layers = {
            'layer1': features[:5],   # First conv block
            'layer2': features[:10],  # Second conv block  
            'layer3': features[:19],  # Third conv block
            'layer4': features[:28],  # Fourth conv block
            'layer5': features,       # Full features
        }
        
        return features
    
    def _create_efficientnet_encoder(self):
        """Create EfficientNet-based encoder"""
        efficientnet = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        
        # Extract feature layers
        self.efficientnet_features = efficientnet.features
        self.efficientnet_avgpool = efficientnet.avgpool
        
        return efficientnet
    
    def _create_clip_encoder(self):
        """Create CLIP-based encoder"""
        try:
            clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
            self.clip_model = clip_model.visual
            self.clip_preprocess = preprocess
            return self.clip_model
        except:
            raise RuntimeError("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
    
    def setup_feature_adaptation(self):
        """Setup feature adaptation layers for different encoders"""
        if self.encoder_type == 'resnet50':
            # ResNet50 outputs 2048 features
            self.global_adapter = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(1024, self.feature_dim)
            )
            
            # Skip connection adapters
            self.skip_adapters = nn.ModuleDict({
                'skip_64': nn.Conv2d(64, 32, 1),     # layer1 output
                'skip_32': nn.Conv2d(128, 64, 1),    # layer2 output  
                'skip_16': nn.Conv2d(256, 128, 1),   # layer3 output
                'skip_8': nn.Conv2d(512, 256, 1),    # layer4 output
            })
            
        elif self.encoder_type == 'vgg19':
            # VGG19 features need global pooling
            self.global_adapter = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(1024, self.feature_dim)
            )
            
            # VGG skip adapters
            self.skip_adapters = nn.ModuleDict({
                'skip_64': nn.Conv2d(64, 32, 1),
                'skip_32': nn.Conv2d(128, 64, 1),
                'skip_16': nn.Conv2d(256, 128, 1),
                'skip_8': nn.Conv2d(512, 256, 1),
            })
            
        elif self.encoder_type == 'efficientnet':
            # EfficientNet-B4 outputs vary, typically 1792
            self.global_adapter = nn.Sequential(
                nn.Linear(1792, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(1024, self.feature_dim)
            )
            
        elif self.encoder_type == 'clip':
            # CLIP ViT-B/32 outputs 512 features
            self.global_adapter = nn.Sequential(
                nn.Linear(512, self.feature_dim),
                nn.ReLU(inplace=True)
            )
    
    def freeze_backbone_weights(self):
        """Freeze pretrained backbone weights"""
        if hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Also freeze individual layers for ResNet
        if self.encoder_type == 'resnet50':
            for layer in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def unfreeze_backbone_weights(self):
        """Unfreeze backbone for fine-tuning"""
        if hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = True
        
        if self.encoder_type == 'resnet50':
            for layer in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def forward_resnet50(self, x):
        """Forward pass for ResNet50"""
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Extract skip connections
        skip_64 = self.layer1(x)  # 1/4 resolution
        skip_32 = self.layer2(skip_64)  # 1/8 resolution
        skip_16 = self.layer3(skip_32)  # 1/16 resolution
        skip_8 = self.layer4(skip_16)   # 1/32 resolution
        
        # Global features
        global_feat = self.avgpool(skip_8)
        global_feat = torch.flatten(global_feat, 1)
        global_feat = self.global_adapter(global_feat)
        
        # Adapt skip connections
        adapted_skips = {}
        for name, adapter in self.skip_adapters.items():
            if name == 'skip_64':
                adapted_skips[name] = adapter(skip_64)
            elif name == 'skip_32':
                adapted_skips[name] = adapter(skip_32)
            elif name == 'skip_16':
                adapted_skips[name] = adapter(skip_16)
            elif name == 'skip_8':
                adapted_skips[name] = adapter(skip_8)
        
        return {
            'global': global_feat,
            'local': adapted_skips['skip_8'],  # Use lowest resolution as local
            **adapted_skips
        }
    
    def forward_vgg19(self, x):
        """Forward pass for VGG19"""
        skip_features = {}
        
        # Extract features at different depths
        for name, layer_module in self.vgg_layers.items():
            x_temp = layer_module(x)
            if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                # Downsample to appropriate skip sizes
                if name == 'layer1':
                    skip_features['skip_64'] = self.skip_adapters['skip_64'](x_temp)
                elif name == 'layer2':  
                    skip_features['skip_32'] = self.skip_adapters['skip_32'](x_temp)
                elif name == 'layer3':
                    skip_features['skip_16'] = self.skip_adapters['skip_16'](x_temp)
                elif name == 'layer4':
                    skip_features['skip_8'] = self.skip_adapters['skip_8'](x_temp)
        
        # Global features from final layer
        final_features = self.vgg_layers['layer5'](x)
        global_feat = self.global_adapter(final_features)
        
        return {
            'global': global_feat,
            'local': skip_features['skip_8'],
            **skip_features
        }
    
    def forward_efficientnet(self, x):
        """Forward pass for EfficientNet"""
        features = self.efficientnet_features(x)
        global_feat = self.efficientnet_avgpool(features)
        global_feat = torch.flatten(global_feat, 1)
        global_feat = self.global_adapter(global_feat)
        
        # EfficientNet is more complex for skip connections
        # For simplicity, we'll just use global features
        return {
            'global': global_feat,
            'local': features,  # Use final features as local
        }
    
    def forward_clip(self, x):
        """Forward pass for CLIP"""
        # Ensure input is in correct range for CLIP
        if x.min() < -0.5:  # Assume input is in [-1, 1]
            x = (x + 1) / 2  # Convert to [0, 1]
        
        # CLIP expects specific normalization
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                       std=[0.26862954, 0.26130258, 0.27577711])
        x = normalize(x)
        
        features = self.clip_model(x)
        global_feat = self.global_adapter(features)
        
        return {
            'global': global_feat,
            'local': features.unsqueeze(-1).unsqueeze(-1),  # Make it spatial-like
        }
    
    def forward(self, x):
        """Main forward pass"""
        if self.encoder_type == 'resnet50':
            return self.forward_resnet50(x)
        elif self.encoder_type == 'vgg19':
            return self.forward_vgg19(x)
        elif self.encoder_type == 'efficientnet':
            return self.forward_efficientnet(x)
        elif self.encoder_type == 'clip':
            return self.forward_clip(x)
        

class CustomEncoder(nn.Module):

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
        
    def forward(self, x):
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


def load_trained_encoder(encoder_path, device='cpu', feature_dim=512):

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder checkpoint not found at: {encoder_path}")

    encoder = CustomEncoder(input_channels=3, feature_dim=feature_dim)

    try:
        checkpoint = torch.load(encoder_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'encoder_state_dict' in checkpoint:
                # If saved with additional metadata
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                print(f"Loaded encoder from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'loss' in checkpoint:
                    print(f"Training loss: {checkpoint['loss']:.6f}")
            elif 'model_state_dict' in checkpoint:
                # Generic model state dict format
                encoder.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Assume it's just the state dict
                encoder.load_state_dict(checkpoint)
        else:
            # Direct state dict
            encoder.load_state_dict(checkpoint)
        
        encoder.to(device)
        encoder.eval()
        
        print(f"Successfully loaded trained encoder from: {encoder_path}")
        return encoder
        
    except Exception as e:
        raise RuntimeError(f"Failed to load encoder: {str(e)}")