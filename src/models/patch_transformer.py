"""Patch-token transformer encoder for motor imagery classification.

Tokenizes each trial into temporal patches, overlapping when the stride is
below the patch size (patch embedding via a strided temporal convolution), applies a spatial
filter bank, then processes the resulting token sequence with a standard
transformer encoder. By default the spatial filters collapse all channels into
one token per window. With channel_groups, each group of channels is collapsed
separately, giving one token per group and window. Time positions are fixed
sinusoidal encodings, and group identity is a learned embedding.
"""

import math

import torch
import torch.nn as nn

from config import PatchTransformerConfig


class PatchEmbedding(nn.Module):
    """Tokenizes EEG epochs via temporal patches, followed by a shared
    spatial filter across channels.

    The temporal convolution's kernel is `patch_size` and its stride is
    `patch_stride`, so it performs windowing and feature extraction in a
    single step, producing one token per time window. A stride below the
    patch size gives overlapping windows. The spatial convolution then
    collapses the channel dimension, mixing across all channels
    identically for every token. With factorized_spatial, the full
    embed_dim x embed_dim x n_channels spatial kernel is replaced by a
    depthwise spatial convolution (one channel filter per feature map)
    followed by a 1x1 convolution, which cuts its weights by about 10x
    at 41 channels. With channel_gate_reduction, a gate first
    rescales each channel per window from the channels' relative temporal-filter
    power, so the spatial mixing adapts to the window. With channel_groups, one spatial
    convolution per group collapses only that group's channels, so each
    window yields one token per group.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    embed_dim : int
        Token embedding dimension.
    patch_size : int
        Number of time samples per patch (token).
    patch_stride : int, optional
        Step between patch starts, in samples. Defaults to patch_size.
        Trailing samples beyond the last full patch are dropped.
    channel_groups : list of list of int, optional
        Channel indices of each group. None collapses all channels into a
        single token per window. Groups need not cover every channel.
    channel_gate_reduction : int, optional
        Bottleneck reduction of the per-window channel gate. 0 disables it.
    factorized_spatial : bool, optional
        Replaces the full spatial convolution by a depthwise one followed
        by a 1x1 convolution. Incompatible with channel_groups.

    Raises
    ------
    ValueError
        If a group is empty or names a channel index outside n_channels, or
        if factorized_spatial is combined with channel_groups.
    """

    def __init__(self, n_channels, embed_dim, patch_size, patch_stride=None, channel_groups=None, channel_gate_reduction=0, factorized_spatial=False):
        super().__init__()
        patch_stride = patch_stride or patch_size
        self.temporal_conv = nn.Conv2d(1, embed_dim, (1, patch_size), stride=(1, patch_stride), bias=False)
        self.channel_groups = channel_groups
        self.channel_gate = nn.Sequential(
            nn.Linear(n_channels, n_channels // channel_gate_reduction), nn.ReLU(),
            nn.Linear(n_channels // channel_gate_reduction, n_channels),
        ) if channel_gate_reduction else None
        if factorized_spatial and channel_groups is not None:
            raise ValueError("factorized_spatial cannot be combined with channel_groups")
        if factorized_spatial:
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, (n_channels, 1), groups=embed_dim, bias=False),
                nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            )
        elif channel_groups is None:
            self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (n_channels, 1), bias=False)
        else:
            if any(not g or min(g) < 0 or max(g) >= n_channels for g in channel_groups):
                raise ValueError(f"channel_groups must be non-empty index lists below n_channels={n_channels}")
            self.spatial_conv = nn.ModuleList(
                [nn.Conv2d(embed_dim, embed_dim, (len(g), 1), bias=False) for g in channel_groups]
            )
        self.norm          = nn.BatchNorm2d(embed_dim)
        self.activation    = nn.ELU()

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_channels, n_times)

        Returns
        -------
        Tensor, shape (batch, n_patches * n_groups, embed_dim)
            Tokens are ordered by window, then by group. n_groups is 1
            without channel_groups.
        """
        x = x.unsqueeze(1)                  # (batch, 1, channels, times)
        x = self.temporal_conv(x)           # (batch, embed_dim, channels, n_patches)
        if self.channel_gate is not None:
            power = x.pow(2).mean(dim=1)                                  # (batch, channels, n_patches)
            power = power / (power.mean(dim=1, keepdim=True) + 1e-8)      # scale-free relative power
            gate  = torch.sigmoid(self.channel_gate(power.transpose(1, 2)))
            x     = x * gate.transpose(1, 2).unsqueeze(1)
        if self.channel_groups is None:
            x = self.spatial_conv(x)        # (batch, embed_dim, 1, n_patches)
        else:
            x = torch.cat([conv(x[:, :, g, :]) for conv, g in zip(self.spatial_conv, self.channel_groups)], dim=2)
        x = self.activation(self.norm(x))   # (batch, embed_dim, n_groups, n_patches)
        return x.permute(0, 3, 2, 1).flatten(1, 2)


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


class PatchTransformer(nn.Module):
    """Transformer encoder for EEG motor imagery classification.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_times : int
        Number of time samples per epoch.
    n_classes : int
        Number of output classes.
    cfg : PatchTransformerConfig
        Architecture hyperparameters.
    """

    def __init__(self, n_channels, n_times, n_classes, cfg: PatchTransformerConfig):
        super().__init__()

        patch_stride = cfg.patch_stride or cfg.patch_size
        if n_times < cfg.patch_size:
            raise ValueError(f"n_times={n_times} is shorter than patch_size={cfg.patch_size}")
        self.n_patches   = (n_times - cfg.patch_size) // patch_stride + 1
        self.patch_embed = PatchEmbedding(n_channels, cfg.embed_dim, cfg.patch_size, patch_stride, cfg.channel_groups, cfg.channel_gate_reduction, cfg.factorized_spatial)
        self.n_groups    = len(cfg.channel_groups) if cfg.channel_groups else 1
        self.region_embed = nn.Parameter(torch.randn(self.n_groups, cfg.embed_dim) * 0.02) if cfg.channel_groups else None

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

    def forward(self, x, channel_positions=None, ch_names=None):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_channels, n_times)
        channel_positions : Tensor, shape (batch, n_channels, 3), optional
            Unused. Accepted for compatibility with NeuralBench's
            check_forward contract (see scripts/neuralbench_eval.py).
        ch_names : list of str, optional
            Unused, same reason as channel_positions.

        Returns
        -------
        Tensor, shape (batch, n_classes)
        """
        tokens = self.patch_embed(x)                                     # (batch, n_patches * n_groups, embed_dim)
        pe     = sinusoidal_positional_encoding(tokens.shape[1] // self.n_groups, tokens.shape[2], tokens.device)
        pe     = pe.repeat_interleave(self.n_groups, dim=0)              # one time position per window, shared by its groups
        if self.region_embed is not None:
            pe = pe + self.region_embed.repeat(tokens.shape[1] // self.n_groups, 1)
        tokens = tokens + pe.unsqueeze(0)
        tokens = self.encoder(tokens)
        pooled = tokens.mean(dim=1)                                      # average over patch tokens
        return self.classifier(self.dropout(pooled))
