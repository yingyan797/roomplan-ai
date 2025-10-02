import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class CrossAttention(nn.Module):
    """
    Cross-attention module with Q, K, V, and projection.
    Applied between encoder (down) and decoder (up) features.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        # Q from decoder (up), K and V from encoder (down)
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, dec_feat, enc_feat):
        """
        Args:
            dec_feat: decoder features (upsampled) [B, C, H, W]
            enc_feat: encoder features (skip connection) [B, C, H, W]
        """
        B, C, H, W = dec_feat.shape
        
        # Generate Q, K, V
        q = self.q_proj(dec_feat)  # Query from decoder
        k = self.k_proj(enc_feat)  # Key from encoder
        v = self.v_proj(enc_feat)  # Value from encoder
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # [B, heads, HW, head_dim]
        k = k.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # [B, heads, HW, head_dim]
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)  # [B, heads, HW, head_dim]
        
        # Attention computation
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, HW, HW]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, heads, HW, head_dim]
        
        # Reshape back
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        out = self.out_proj(out)
        
        return out


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=6, base_ch=64):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.enc4 = ConvBlock(base_ch*4, base_ch*8)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4


class UNetDecoder(nn.Module):
    def __init__(self, base_ch=64, out_channels=1, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)
        self.out = nn.Conv2d(base_ch, out_channels, 1)
        
        if use_attention:
            self.attn3 = CrossAttention(base_ch*4, num_heads=4)
            self.attn2 = CrossAttention(base_ch*2, num_heads=4)
            self.attn1 = CrossAttention(base_ch, num_heads=4)
        
    def forward(self, e1, e2, e3, e4, apply_attention=False):
        # Decoder level 3
        d3 = self.up3(e4)
        
        # Handle odd dimension mismatches (e.g., 65->32->64 instead of 65)
        # This only happens with odd input dimensions after pooling/upsampling
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        
        if apply_attention and self.use_attention:
            e3_attn = self.attn3(d3, e3)
            d3 = torch.cat([d3, e3_attn], dim=1)
        else:
            d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        # Decoder level 2
        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        
        if apply_attention and self.use_attention:
            e2_attn = self.attn2(d2, e2)
            d2 = torch.cat([d2, e2_attn], dim=1)
        else:
            d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        # Decoder level 1
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        
        if apply_attention and self.use_attention:
            e1_attn = self.attn1(d1, e1)
            d1 = torch.cat([d1, e1_attn], dim=1)
        else:
            d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)


class UNetWithAttention(nn.Module):
    """UNet with cross-attention between encoder and decoder"""
    def __init__(self, in_channels=6, base_ch=64):
        super().__init__()
        self.encoder = UNetEncoder(in_channels, base_ch)
        self.decoder = UNetDecoder(base_ch, out_channels=1, use_attention=True)
        
    def forward(self, x, use_attention=True):
        e1, e2, e3, e4 = self.encoder(x)
        out = self.decoder(e1, e2, e3, e4, apply_attention=use_attention)
        return out


class FusionModel(nn.Module):
    """Simple fusion model to combine predictions from both routes"""
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
        
    def forward(self, pred_a, pred_b):
        # Handle size mismatch if any
        if pred_a.shape != pred_b.shape:
            pred_b = F.interpolate(pred_b, size=pred_a.shape[2:], mode='bilinear', align_corners=False)
        
        combined = torch.cat([pred_a, pred_b], dim=1)
        return self.fusion(combined)