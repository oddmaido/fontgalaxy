"""
ResNet-based Font VAE with proper residual connections.

Adapted from resnet.py for:
- 1-channel grayscale input (64x64)
- Glyph-conditioned decoder
- VAE latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block for encoder."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResidualBlockUp(nn.Module):
    """Residual block with upsampling for decoder."""

    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        self.upsample = upsample

        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.shortcut_bn = nn.BatchNorm2d(out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut_bn(self.shortcut(x)) if isinstance(self.shortcut, nn.Conv2d) else x
        return F.relu(out + shortcut)


class ResNetGlyphEncoder(nn.Module):
    """
    ResNet-style encoder for individual glyphs.
    64x64 → 256-dim token
    """

    def __init__(self, token_dim: int = 256):
        super().__init__()

        # Initial conv: 64x64x1 → 64x64x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Residual blocks with downsampling
        self.layer1 = self._make_layer(64, 64, 2, stride=2)    # 64→32
        self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 32→16
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 16→8
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 8→4

        # Global pool + project to token
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, token_dim)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, 64, 64)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)  # (B, token_dim)


class ResNetFontEncoder(nn.Module):
    """
    Encodes a set of glyphs into font latent space.
    Uses attention pooling over glyph tokens.
    """

    def __init__(self, token_dim: int = 256, latent_dim: int = 256):
        super().__init__()

        self.glyph_encoder = ResNetGlyphEncoder(token_dim)

        # Attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads=4, batch_first=True)

        # VAE projections
        self.fc_mu = nn.Linear(token_dim, latent_dim)
        self.fc_logvar = nn.Linear(token_dim, latent_dim)

    def forward(self, glyphs):
        """
        Args:
            glyphs: (B, N, 1, 64, 64) - N glyphs per font
        Returns:
            z, mu, logvar
        """
        B, N, C, H, W = glyphs.shape

        # Encode each glyph
        glyphs_flat = glyphs.view(B * N, C, H, W)
        tokens = self.glyph_encoder(glyphs_flat)  # (B*N, token_dim)
        tokens = tokens.view(B, N, -1)  # (B, N, token_dim)

        # Attention pooling
        query = self.query.expand(B, -1, -1)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)
        pooled, _ = self.attn(query, k, v)
        pooled = pooled.squeeze(1)  # (B, token_dim)

        # VAE latent
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)

        return mu, logvar


class ResNetGlyphDecoder(nn.Module):
    """
    ResNet-style decoder with glyph conditioning.
    (z_font, glyph_id) → 64x64 glyph
    """

    def __init__(self, latent_dim: int = 256, glyph_emb_dim: int = 64, num_glyphs: int = 94):
        super().__init__()

        self.glyph_embedding = nn.Embedding(num_glyphs, glyph_emb_dim)

        input_dim = latent_dim + glyph_emb_dim

        # Project to spatial: 4x4x512
        self.fc = nn.Linear(input_dim, 512 * 4 * 4)

        # Residual upsampling blocks
        self.layer1 = ResidualBlockUp(512, 256, upsample=True)   # 4→8
        self.layer2 = ResidualBlockUp(256, 128, upsample=True)   # 8→16
        self.layer3 = ResidualBlockUp(128, 64, upsample=True)    # 16→32
        self.layer4 = ResidualBlockUp(64, 32, upsample=True)     # 32→64

        # Output conv
        self.out = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z_font, glyph_ids):
        """
        Args:
            z_font: (B, latent_dim)
            glyph_ids: (B,)
        Returns:
            (B, 1, 64, 64)
        """
        glyph_emb = self.glyph_embedding(glyph_ids)
        z = torch.cat([z_font, glyph_emb], dim=1)

        x = self.fc(z).view(-1, 512, 4, 4)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return self.out(x)


class ResNetFontVAE(nn.Module):
    """
    Complete ResNet-based Font VAE.
    """

    def __init__(
        self,
        token_dim: int = 256,
        latent_dim: int = 256,
        glyph_emb_dim: int = 64,
        num_glyphs: int = 94,
        beta: float = 0.0001,  # Very low KL weight
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = ResNetFontEncoder(token_dim, latent_dim)
        self.decoder = ResNetGlyphDecoder(latent_dim, glyph_emb_dim, num_glyphs)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, glyphs):
        mu, logvar = self.encoder(glyphs)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z_font, glyph_ids):
        return self.decoder(z_font, glyph_ids)

    def forward(self, encoder_glyphs, decoder_glyph_ids, decoder_targets):
        """
        Args:
            encoder_glyphs: (B, K, 1, 64, 64)
            decoder_glyph_ids: (B, M)
            decoder_targets: (B, M, 1, 64, 64)
        """
        B, M = decoder_glyph_ids.shape

        # Encode
        z, mu, logvar = self.encode(encoder_glyphs)

        # Decode each target glyph
        z_expanded = z.unsqueeze(1).expand(-1, M, -1).reshape(B * M, -1)
        glyph_ids_flat = decoder_glyph_ids.reshape(B * M)

        reconstructions = self.decode(z_expanded, glyph_ids_flat)
        reconstructions = reconstructions.view(B, M, 1, 64, 64)

        # Losses
        recon_loss = self.reconstruction_loss(reconstructions, decoder_targets)
        kl_loss = self.kl_loss(mu, logvar)

        loss = recon_loss + self.beta * kl_loss

        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'reconstructions': reconstructions,
            'z': z,
            'mu': mu,
        }

    def reconstruction_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def kl_loss(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


if __name__ == "__main__":
    # Test
    model = ResNetFontVAE()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    enc_glyphs = torch.randn(2, 16, 1, 64, 64)
    dec_ids = torch.randint(0, 94, (2, 8))
    dec_targets = torch.randn(2, 8, 1, 64, 64).sigmoid()

    out = model(enc_glyphs, dec_ids, dec_targets)
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"Recon shape: {out['reconstructions'].shape}")
