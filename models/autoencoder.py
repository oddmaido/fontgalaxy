"""
Simple per-glyph autoencoder for font embedding and blending.

Architecture:
- Encoder: 64x64 glyph → 256d latent
- Decoder: 256d latent → 64x64 glyph

Usage:
- Clustering: encode all glyphs of a font, average → font embedding
- Blending: interpolate z_glyph from font1 and font2, decode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block."""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return x + h


class Encoder(nn.Module):
    """Encodes 64x64 glyph to latent vector."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()

        # 64x64x1 → 32x32x64
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResBlock(64),
        )

        # 32x32x64 → 16x16x128
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResBlock(128),
        )

        # 16x16x128 → 8x8x256
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            ResBlock(256),
        )

        # 8x8x256 → 4x4x512
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            ResBlock(512),
        )

        # 4x4x512 = 8192 → latent_dim
        self.fc = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = x.flatten(1)
        return self.fc(x)


class Decoder(nn.Module):
    """Decodes latent vector to 64x64 glyph."""

    def __init__(self, latent_dim: int = 256):
        super().__init__()

        # latent_dim → 4x4x512
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        # 4x4x512 → 8x8x256
        self.up1 = nn.Sequential(
            ResBlock(512),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
        )

        # 8x8x256 → 16x16x128
        self.up2 = nn.Sequential(
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )

        # 16x16x128 → 32x32x64
        self.up3 = nn.Sequential(
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        # 32x32x64 → 64x64x1
        self.up4 = nn.Sequential(
            ResBlock(64),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x


class GlyphAutoencoder(nn.Module):
    """
    Simple autoencoder for font glyphs.

    Training: encode glyph → z → decode → reconstruct same glyph

    Inference:
    - Embedding: encode all glyphs, average z → font embedding
    - Blending: z_blend = lerp(z1, z2, alpha), decode
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x):
        """Encode glyph(s) to latent."""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent to glyph(s)."""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def get_font_embedding(self, glyphs):
        """
        Get font-level embedding by averaging glyph embeddings.

        Args:
            glyphs: (N, 1, 64, 64) - all glyphs from one font
        Returns:
            z_font: (latent_dim,) - averaged embedding
        """
        z_glyphs = self.encode(glyphs)  # (N, latent_dim)
        return z_glyphs.mean(dim=0)

    def blend(self, z1, z2, alpha=0.5):
        """Interpolate between two latents."""
        return (1 - alpha) * z1 + alpha * z2


if __name__ == "__main__":
    model = GlyphAutoencoder(latent_dim=256)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test
    x = torch.randn(4, 1, 64, 64)
    recon, z = model(x)
    print(f"Input: {x.shape}")
    print(f"Latent: {z.shape}")
    print(f"Recon: {recon.shape}")
