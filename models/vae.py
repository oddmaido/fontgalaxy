"""
Font VAE - Complete model with encoding, decoding, and losses.

Architecture:
- FontEncoder: Set of glyphs → (μ, σ) → z_font (256d)
- GlyphDecoder: (z_font, glyph_id) → 64x64 glyph

Losses:
- Reconstruction (L1 + perceptual)
- KL divergence
- Same-font consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import FontEncoder
from .decoder import GlyphDecoder, GlyphDecoderWithSkip


class FontVAE(nn.Module):
    """
    Variational Autoencoder for font style representation.

    Training:
    1. Sample K glyphs from a font for encoding
    2. Sample M glyphs from the same font for reconstruction
    3. Encode → sample z → decode
    4. Compute losses

    Inference:
    - Embedding: encode all glyphs → get μ (deterministic)
    - Generation: provide z + glyph_id → decode
    - Blending: interpolate z_A and z_B → decode
    """

    def __init__(
        self,
        token_dim: int = 256,
        latent_dim: int = 256,
        glyph_emb_dim: int = 64,
        num_glyphs: int = 94,
        beta: float = 0.001,  # KL weight (very small to preserve style)
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = FontEncoder(token_dim, latent_dim)
        self.decoder = GlyphDecoderWithSkip(latent_dim, glyph_emb_dim, num_glyphs)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Deterministic at inference

    def encode(
        self,
        glyphs: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a set of glyphs to font latent.

        Args:
            glyphs: (B, K, 1, 64, 64) - K glyphs per font
            mask: (B, K) - valid glyph mask

        Returns:
            z: (B, latent_dim) - sampled latent
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        mu, logvar = self.encoder(glyphs, mask)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor, glyph_ids: torch.Tensor) -> torch.Tensor:
        """
        Decode font latent + glyph IDs to images.

        Args:
            z: (B, latent_dim) - font latent
            glyph_ids: (B,) - glyph indices

        Returns:
            images: (B, 1, 64, 64)
        """
        return self.decoder(z, glyph_ids)

    def forward(
        self,
        encoder_glyphs: torch.Tensor,
        decoder_glyph_ids: torch.Tensor,
        decoder_targets: torch.Tensor,
        encoder_mask: torch.Tensor = None,
    ) -> dict:
        """
        Full forward pass with loss computation.

        Args:
            encoder_glyphs: (B, K, 1, 64, 64) - glyphs for encoding
            decoder_glyph_ids: (B, M) - glyph IDs to reconstruct
            decoder_targets: (B, M, 1, 64, 64) - target images
            encoder_mask: (B, K) - valid encoder glyphs

        Returns:
            dict with 'loss', 'recon_loss', 'kl_loss', 'reconstructions'
        """
        B, M = decoder_glyph_ids.shape

        # Encode
        z, mu, logvar = self.encode(encoder_glyphs, encoder_mask)

        # Decode each target glyph
        # Expand z for each glyph: (B, latent_dim) → (B*M, latent_dim)
        z_expanded = z.unsqueeze(1).expand(-1, M, -1).reshape(B * M, -1)
        glyph_ids_flat = decoder_glyph_ids.reshape(B * M)

        reconstructions = self.decode(z_expanded, glyph_ids_flat)
        reconstructions = reconstructions.view(B, M, 1, 64, 64)

        # Losses
        recon_loss = self.reconstruction_loss(reconstructions, decoder_targets)
        kl_loss = self.kl_loss(mu, logvar)

        total_loss = recon_loss + self.beta * kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'reconstructions': reconstructions,
            'z': z,
            'mu': mu,
        }

    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence from N(0, I)."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def consistency_loss(
        self,
        glyphs1: torch.Tensor,
        glyphs2: torch.Tensor,
        mask1: torch.Tensor = None,
        mask2: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Same-font consistency: μ from different glyph subsets should match.

        Args:
            glyphs1, glyphs2: (B, K, 1, 64, 64) - different glyph subsets from same fonts
            mask1, mask2: (B, K) - valid glyph masks

        Returns:
            MSE between the two μ vectors
        """
        mu1, _ = self.encoder(glyphs1, mask1)
        mu2, _ = self.encoder(glyphs2, mask2)
        return F.mse_loss(mu1, mu2)

    @torch.no_grad()
    def get_embedding(
        self,
        glyphs: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Get deterministic font embedding (μ only, no sampling).

        Args:
            glyphs: (B, K, 1, 64, 64)
            mask: (B, K)

        Returns:
            embeddings: (B, latent_dim)
        """
        self.eval()
        mu, _ = self.encoder(glyphs, mask)
        return mu

    @torch.no_grad()
    def interpolate(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        glyph_ids: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Interpolate between two font latents and decode.

        Args:
            z1, z2: (latent_dim,) - two font latents
            glyph_ids: (N,) - glyph IDs to generate
            alpha: interpolation weight (0=z1, 1=z2)

        Returns:
            images: (N, 1, 64, 64)
        """
        self.eval()
        z_mix = (1 - alpha) * z1 + alpha * z2
        z_mix = z_mix.unsqueeze(0).expand(len(glyph_ids), -1)
        return self.decode(z_mix, glyph_ids)


def create_model(
    latent_dim: int = 256,
    beta: float = 0.1,
    num_glyphs: int = 94,
) -> FontVAE:
    """Factory function to create a FontVAE with default settings."""
    return FontVAE(
        token_dim=256,
        latent_dim=latent_dim,
        glyph_emb_dim=64,
        num_glyphs=num_glyphs,
        beta=beta,
    )
