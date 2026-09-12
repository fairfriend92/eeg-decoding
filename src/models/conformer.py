"""EEG Conformer-style transformer encoder for motor imagery classification.

Tokenizes each trial into non-overlapping temporal patches (patch embedding
via a strided temporal convolution), applies a single shared spatial filter
across channels, then processes the resulting token sequence with a
standard transformer encoder. Positional information is injected via fixed
sinusoidal encodings rather than a learned embedding table.
"""

import math

import torch
import torch.nn as nn

from config import ConformerConfig


class PatchEmbedding(nn.Module):
    """Tokenizes EEG epochs via temporal patches, followed by a shared
    spatial filter across channels.

    The temporal convolution's kernel and stride both equal `patch_size`,
    so it performs windowing and feature extraction in a single step,
    producing one token per non-overlapping time window. The spatial
    convolution then collapses the channel dimension, mixing across all
    channels identically for every token.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    embed_dim : int
        Token embedding dimension.
    patch_size : int
        Number of time samples per patch (token). Trailing samples beyond
        the last full patch are dropped.
    """

    def __init__(self, n_channels, embed_dim, patch_size):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, embed_dim, (1, patch_size), stride=(1, patch_size), bias=False)
        self.spatial_conv  = nn.Conv2d(embed_dim, embed_dim, (n_channels, 1), bias=False)
        self.norm          = nn.BatchNorm2d(embed_dim)
        self.activation    = nn.ELU()

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_channels, n_times)

        Returns
        -------
        Tensor, shape (batch, n_patches, embed_dim)
        """
        x = x.unsqueeze(1)                  # (batch, 1, channels, times)
        x = self.temporal_conv(x)           # (batch, embed_dim, channels, n_patches)
        x = self.spatial_conv(x)            # (batch, embed_dim, 1, n_patches)
        x = self.activation(self.norm(x))
        x = x.squeeze(2)                    # (batch, embed_dim, n_patches)
        return x.transpose(1, 2)            # (batch, n_patches, embed_dim)


def sinusoidal_positional_encoding(seq_len, embed_dim, device):
    """Computes fixed sinusoidal position encodings.

    Parameters
    ----------
    seq_len : int
        Number of token positions.
    embed_dim : int
        Token embedding dimension.
    device : torch.device
        Device on which to allocate the encoding.

    Returns
    -------
    Tensor, shape (seq_len, embed_dim)
    """
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, device=device) * (-math.log(10000.0) / embed_dim)
    )
    pe          = torch.zeros(seq_len, embed_dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class EEGConformer(nn.Module):
    """Transformer encoder for EEG motor imagery classification.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_times : int
        Number of time samples per epoch.
    n_classes : int
        Number of output classes.
    cfg : ConformerConfig
        Architecture hyperparameters.
    """

    def __init__(self, n_channels, n_times, n_classes, cfg: ConformerConfig):
        super().__init__()

        self.patch_embed = PatchEmbedding(n_channels, cfg.embed_dim, cfg.patch_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        self.dropout    = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.embed_dim, n_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_channels, n_times)

        Returns
        -------
        Tensor, shape (batch, n_classes)
        """
        tokens = self.patch_embed(x)                                     # (batch, n_patches, embed_dim)
        pe     = sinusoidal_positional_encoding(tokens.shape[1], tokens.shape[2], tokens.device)
        tokens = tokens + pe.unsqueeze(0)
        tokens = self.encoder(tokens)
        pooled = tokens.mean(dim=1)                                      # average over patch tokens
        return self.classifier(self.dropout(pooled))
