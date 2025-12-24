import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

    
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scalar

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//8)
        key = self.key(x).view(B, -1, H * W)  # (B, C//8, HW)
        attention = torch.bmm(query, key)  # (B, HW, HW)
        attention = F.softmax(attention, dim=-1)

        value = self.value(x).view(B, -1, H * W)  # (B, C, HW)
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

class Generator(nn.Module):
    def __init__(
        self,
        conf: Optional[Dict] = None,
        num_conditions: int = 9,
        noise_dim: int = 100,
        embed_dim: int = 256,
        embed_out_dim: int = 128,
        channels: int = 3,
        use_initial_image: bool = True,
        image_encoder: Optional[nn.Module] = None,
        encoder_checkpoint: Optional[str] = None,
        freeze_encoder: bool = True,
        input_size: int = 128,
        device: str = 'cuda:0'
    ):
        super().__init__()

        if conf is not None:
            channels = int(conf.get("image", {}).get("n_channels", channels))
            noise_dim = int(conf.get("model", {}).get("noise_dim", noise_dim))
            embed_dim = int(conf.get("model", {}).get("embed_dim", embed_dim))
            embed_out_dim = int(conf.get("model", {}).get("embed_out_dim", embed_out_dim))
            input_features = conf.get("dataset", {}).get("input_features", "")
            if input_features:
                num_conditions = len(input_features.split(","))
            encoder_cfg = conf.get("model", {}).get("image_encoder", {})
            use_initial_image = encoder_cfg.get("encoder_type") is not None or use_initial_image
            encoder_checkpoint = encoder_cfg.get("encoder_path", encoder_checkpoint)
            freeze_encoder = encoder_cfg.get("freeze_encoder", freeze_encoder)

        self.channels = channels
        self.noise_dim = noise_dim
        self.embed_dim = embed_dim
        self.embed_out_dim = embed_out_dim
        self.input_dim = num_conditions
        self.input_size = input_size
        self.use_initial_image = use_initial_image
        self.device = device

        self.image_encoder = image_encoder if self.use_initial_image else None
        if self.use_initial_image and self.image_encoder is None and encoder_checkpoint:
            try:
                from .pretrained_encoder import PretrainedEncoderWrapper
                self.image_encoder = PretrainedEncoderWrapper(encoder_checkpoint, device=device)
            except Exception as exc:
                print(f"Warning: Failed to load pretrained encoder from {encoder_checkpoint}: {exc}")
                self.image_encoder = None
        if self.use_initial_image and self.image_encoder is not None and freeze_encoder:
            self.freeze_image_encoder()

        if self.use_initial_image and encoder_checkpoint:
            print(f"Using pretrained encoder from: {encoder_checkpoint}")

        self.text_embedding = nn.Sequential(
            nn.Linear(self.input_dim , self.embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim, self.embed_out_dim),
            nn.BatchNorm1d(self.embed_out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc_no_image = nn.Linear(self.noise_dim + self.embed_out_dim, 1024 * 4 * 4)
        self.fc_with_image = nn.Linear(self.noise_dim + self.embed_out_dim + 512, 1024 * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.attn = SelfAttention(64)

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)

        self.deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(16)

        self.deconv7 = nn.ConvTranspose2d(16, self.channels, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.tanh = nn.Tanh()


    def freeze_image_encoder(self):
        if self.image_encoder is None:
            return
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        print("Image encoder frozen")

    def unfreeze_image_encoder(self):
        if self.image_encoder is None:
            return
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        print("Image encoder unfrozen for fine-tuning")

    def forward(self, noise, conditions, initial_image=None):
        text_emb = self.text_embedding(conditions)
        noise_flat = noise.view(noise.shape[0], -1)

        image_features = None
        if self.use_initial_image and self.image_encoder is not None and initial_image is not None:
            if initial_image.shape[-1] != self.input_size or initial_image.shape[-2] != self.input_size:
                initial_image = F.interpolate(
                    initial_image,
                    size=(self.input_size, self.input_size),
                    mode='bilinear',
                    align_corners=False
                )
            encoded = self.image_encoder(initial_image)
            if isinstance(encoded, dict):
                image_features = encoded.get('global', encoded)
            else:
                image_features = encoded

        if image_features is not None:
            combined_features = torch.cat([noise_flat, text_emb, image_features], dim=1)
            z = self.fc_with_image(combined_features)
        else:
            combined_features = torch.cat([noise_flat, text_emb], dim=1)
            z = self.fc_no_image(combined_features)

        z = z.view(z.shape[0], 1024, 4, 4)

        z = F.relu(self.bn1(self.deconv1(z)))  # 8x8x512
        z = F.relu(self.bn2(self.deconv2(z)))  # 16x16x256
        z = F.relu(self.bn3(self.deconv3(z)))  # 32x32x128      
        z = F.relu(self.bn4(self.deconv4(z)))  # 64x64x64      
        z = self.attn(z)
        z = F.relu(self.bn5(self.deconv5(z)))  # 128x128x32
        z = F.relu(self.bn6(self.deconv6(z)))  # 256x256x16
        z = self.tanh(self.deconv7(z))  # 512x512x3

        return z

class Discriminator(nn.Module):
    def __init__(
        self,
        num_conditions: int = 9,
        channels: int = 3,
        embed_dim: int = 256,
        embed_out_dim: int = 128
    ):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.embed_dim = embed_dim
        self.embed_out_dim = embed_out_dim
        self.input_dim = num_conditions

        self.conv1 = nn.Conv2d(self.channels, 32, 4, 2, 1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.attn = SelfAttention(128)

        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.text_embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, self.embed_out_dim),
            nn.BatchNorm1d(self.embed_out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.output = nn.Conv2d(512 + self.embed_out_dim, 1, 4, 1, 0, bias=False)

    def forward(self, x, text):
        
        x_out = self.relu1(self.conv1(x))
        x_out = self.relu2(self.bn2(self.conv2(x_out)))
        x_out = self.relu3(self.bn3(self.conv3(x_out)))
        x_out = self.attn(x_out)
        x_out = self.relu4(self.bn4(self.conv4(x_out)))
        x_out = self.relu5(self.bn5(self.conv5(x_out)))

        batch_size, _, height, width = x_out.size()

        text_emb = self.text_embedding(text)
        text_emb = text_emb.view(text_emb.size(0), text_emb.size(1), 1, 1)
        text_emb = text_emb.expand(-1, -1, height, width)
        combined = torch.cat([x_out, text_emb], dim=1)

        out = self.output(combined)

        return out.squeeze(), x_out
    