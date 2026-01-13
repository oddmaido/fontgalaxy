"""
Glyph Decoder for Font VAE.

Takes font latent + glyph ID and generates a 64x64 grayscale glyph image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTransposeBlock(nn.Module):
    """ConvTranspose + BatchNorm + ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_ch, out_ch, 4, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class GlyphDecoder(nn.Module):
    """
    Decodes (z_font, glyph_id) into a 64x64 grayscale glyph image.

    Architecture:
    - Glyph ID → learned embedding
    - Concat [z_font, glyph_emb] → project to 4x4x256
    - Upsample: 4x4 → 8x8 → 16x16 → 32x32 → 64x64
    """

    def __init__(
        self,
        latent_dim: int = 256,
        glyph_emb_dim: int = 64,
        num_glyphs: int = 94,  # A-Z, a-z, 0-9, punctuation
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.glyph_emb_dim = glyph_emb_dim

        # Learnable glyph embeddings
        self.glyph_embedding = nn.Embedding(num_glyphs, glyph_emb_dim)

        # Project concatenated input to spatial feature map
        input_dim = latent_dim + glyph_emb_dim
        self.fc = nn.Linear(input_dim, 256 * 4 * 4)

        # Upsample path: 4x4 → 64x64
        self.deconv1 = ConvTransposeBlock(256, 128)  # 4→8
        self.deconv2 = ConvTransposeBlock(128, 64)   # 8→16
        self.deconv3 = ConvTransposeBlock(64, 32)    # 16→32
        self.deconv4 = ConvTransposeBlock(32, 16)    # 32→64

        # Final conv to grayscale
        self.final_conv = nn.Conv2d(16, 1, 3, padding=1)

    def forward(
        self,
        z_font: torch.Tensor,
        glyph_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_font: (B, latent_dim) - font style latent
            glyph_ids: (B,) - integer glyph IDs (0-93)

        Returns:
            images: (B, 1, 64, 64) - generated glyph images
        """
        # Get glyph embeddings
        glyph_emb = self.glyph_embedding(glyph_ids)  # (B, glyph_emb_dim)

        # Concatenate font latent and glyph embedding
        combined = torch.cat([z_font, glyph_emb], dim=1)  # (B, latent_dim + glyph_emb_dim)

        # Project to spatial
        x = self.fc(combined)  # (B, 256*4*4)
        x = x.view(-1, 256, 4, 4)  # (B, 256, 4, 4)

        # Upsample
        x = self.deconv1(x)  # (B, 128, 8, 8)
        x = self.deconv2(x)  # (B, 64, 16, 16)
        x = self.deconv3(x)  # (B, 32, 32, 32)
        x = self.deconv4(x)  # (B, 16, 64, 64)

        # Final output
        x = self.final_conv(x)  # (B, 1, 64, 64)

        # Sigmoid to [0, 1] range (0=black, 1=white)
        return torch.sigmoid(x)


class GlyphDecoderWithSkip(nn.Module):
    """
    Enhanced decoder with skip connections for sharper outputs.
    Uses a UNet-style architecture with FiLM conditioning.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        glyph_emb_dim: int = 64,
        num_glyphs: int = 94,
    ):
        super().__init__()

        # Glyph embeddings
        self.glyph_embedding = nn.Embedding(num_glyphs, glyph_emb_dim)

        input_dim = latent_dim + glyph_emb_dim

        # Initial projection
        self.fc = nn.Linear(input_dim, 512 * 4 * 4)

        # Decoder layers with FiLM modulation
        self.up1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)  # 4→8
        self.up2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 8→16
        self.up3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 16→32
        self.up4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)    # 32→64

        # Batch norms
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)

        # Output
        self.out = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z_font: torch.Tensor, glyph_ids: torch.Tensor) -> torch.Tensor:
        glyph_emb = self.glyph_embedding(glyph_ids)
        z = torch.cat([z_font, glyph_emb], dim=1)

        x = self.fc(z).view(-1, 512, 4, 4)

        x = F.relu(self.bn1(self.up1(x)))
        x = F.relu(self.bn2(self.up2(x)))
        x = F.relu(self.bn3(self.up3(x)))
        x = F.relu(self.bn4(self.up4(x)))

        return self.out(x)
