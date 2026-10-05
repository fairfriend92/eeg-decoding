"""EEG Conformer: convolutional front end plus transformer encoder (Song et al., 2023).

A shallow ConvNet-style convolution module (temporal convolution, spatial
convolution across channels, batch norm, average pooling) turns each trial
into a short token sequence. A pre-norm transformer encoder then models
global dependencies between tokens, and a fully connected head classifies
the flattened encoder output.
"""

import torch
import torch.nn as nn

from config import ConformerConfig


class ConvModule(nn.Module):
    """Shallow ConvNet-style convolution module producing the token sequence.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    cfg : ConformerConfig
        Architecture hyperparameters.
    """

    def __init__(self, n_channels, cfg: ConformerConfig):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, cfg.n_filters, (1, cfg.kernel_length))
        self.spatial_conv  = nn.Conv2d(cfg.n_filters, cfg.n_filters, (n_channels, 1), bias=False)
        self.norm          = nn.BatchNorm2d(cfg.n_filters)
        self.activation    = nn.ELU()
        self.pool          = nn.AvgPool2d((1, cfg.pool_size), (1, cfg.pool_stride))
        self.dropout       = nn.Dropout(cfg.dropout)
        self.projection    = nn.Conv2d(cfg.n_filters, cfg.embed_dim, (1, 1))

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_channels, n_times)

        Returns
        -------
        Tensor, shape (batch, n_tokens, embed_dim)
        """
        x = x.unsqueeze(1)                          # (batch, 1, channels, times)
        x = self.temporal_conv(x)                   # (batch, n_filters, channels, times')
        x = self.spatial_conv(x)                    # (batch, n_filters, 1, times')
        x = self.dropout(self.pool(self.activation(self.norm(x))))
        x = self.projection(x)                      # (batch, embed_dim, 1, n_tokens)
        return x.squeeze(2).transpose(1, 2)         # (batch, n_tokens, embed_dim)


class EEGConformer(nn.Module):
    """Convolution module, transformer encoder, and MLP classification head.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_times : int
        Number of time samples per epoch. Fixes the classification head's
        input size.
    n_classes : int
        Number of output classes.
    cfg : ConformerConfig
        Architecture hyperparameters.
    """

    def __init__(self, n_channels, n_times, n_classes, cfg: ConformerConfig):
        super().__init__()

        self.patch_embed = ConvModule(n_channels, cfg)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_expansion * cfg.embed_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers, enable_nested_tensor=False)

        n_tokens = (n_times - cfg.kernel_length + 1 - cfg.pool_size) // cfg.pool_stride + 1
        if n_tokens < 1:
            raise ValueError(
                f"n_times={n_times} is too short for kernel_length={cfg.kernel_length} "
                f"and pool_size={cfg.pool_size}"
            )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_tokens * cfg.embed_dim, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

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
        tokens = self.patch_embed(x)
        tokens = self.encoder(tokens)
        return self.classifier(tokens)
