"""
Glyph and Font Encoders for Font VAE.

Architecture:
- GlyphEncoder: Single glyph image → 256d token
- FontEncoder: Set of glyph tokens → font latent (μ, σ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv + BatchNorm + ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class GlyphEncoder(nn.Module):
    """
    Encodes a single 64x64 grayscale glyph image into a 256d token.

    Architecture: Simple ConvNet (not ResNet to keep it fast)
    64x64 → 32x32 → 16x16 → 8x8 → 4x4 → flatten → 256d
    """

    def __init__(self, token_dim: int = 256):
        super().__init__()
        self.token_dim = token_dim

        # 64x64x1 → 32x32x32
        self.conv1 = ConvBlock(1, 32, stride=2)
        # 32x32x32 → 16x16x64
        self.conv2 = ConvBlock(32, 64, stride=2)
        # 16x16x64 → 8x8x128
        self.conv3 = ConvBlock(64, 128, stride=2)
        # 8x8x128 → 4x4x256
        self.conv4 = ConvBlock(128, 256, stride=2)

        # 4x4x256 = 4096 → token_dim
        self.fc = nn.Linear(256 * 4 * 4, token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 64, 64) grayscale glyph images

        Returns:
            tokens: (B, token_dim) glyph tokens
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(1)
        return self.fc(x)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over a set of tokens.
    Uses a learned query with proper key/value projections.
    """

    def __init__(self, token_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Learned query token
        self.query = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

        # Projections
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        self.out_proj = nn.Linear(token_dim, token_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            tokens: (B, K, token_dim) - K glyph tokens per font
            mask: (B, K) - True for valid tokens, False for padding

        Returns:
            pooled: (B, token_dim) - single font representation
        """
        B, K, D = tokens.shape

        # Project keys and values
        keys = self.k_proj(tokens)    # (B, K, D)
        values = self.v_proj(tokens)  # (B, K, D)

        # Expand query for batch
        query = self.query.expand(B, -1, -1)  # (B, 1, D)

        # Reshape for multi-head attention
        # (B, K, D) -> (B, num_heads, K, head_dim)
        keys = keys.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        query = query.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, num_heads, 1, K)
        scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask: (B, K) -> (B, 1, 1, K)
            scores = scores.masked_fill(~mask.view(B, 1, 1, K), float('-inf'))

        weights = F.softmax(scores, dim=-1)  # (B, num_heads, 1, K)

        # Weighted sum: (B, num_heads, 1, head_dim)
        attended = torch.matmul(weights, values)

        # Reshape back: (B, 1, D)
        attended = attended.transpose(1, 2).contiguous().view(B, 1, D)

        # Output projection
        pooled = self.out_proj(attended.squeeze(1))  # (B, D)

        return pooled


class FontEncoder(nn.Module):
    """
    Encodes a set of glyph images from one font into a VAE latent (μ, σ).

    Pipeline:
    1. Encode each glyph independently with shared GlyphEncoder
    2. Pool glyph tokens with attention
    3. Project to VAE parameters (μ, log_σ)
    """

    def __init__(self, token_dim: int = 256, latent_dim: int = 256):
        super().__init__()
        self.glyph_encoder = GlyphEncoder(token_dim)
        self.attention_pool = AttentionPooling(token_dim)

        # VAE projection heads
        self.fc_mu = nn.Linear(token_dim, latent_dim)
        self.fc_logvar = nn.Linear(token_dim, latent_dim)

    def encode_glyphs(self, glyphs: torch.Tensor) -> torch.Tensor:
        """
        Encode batch of glyphs.

        Args:
            glyphs: (B, K, 1, 64, 64) - K glyphs per font

        Returns:
            tokens: (B, K, token_dim)
        """
        B, K, C, H, W = glyphs.shape
        # Flatten batch and glyph dims
        glyphs_flat = glyphs.view(B * K, C, H, W)
        tokens_flat = self.glyph_encoder(glyphs_flat)
        return tokens_flat.view(B, K, -1)

    def forward(
        self,
        glyphs: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            glyphs: (B, K, 1, 64, 64) - K glyph images per font
            mask: (B, K) - True for valid glyphs

        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        tokens = self.encode_glyphs(glyphs)
        pooled = self.attention_pool(tokens, mask)

        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)

        return mu, logvar
