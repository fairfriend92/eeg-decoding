"""EEGNet: compact CNN baseline for EEG classification (Lawhern et al., 2018).

Factorizes convolution into three stages: a temporal filter, a depthwise
spatial filter across channels, and a separable convolution that mixes the
resulting feature maps.
"""

import torch
import torch.nn as nn

from config import EEGNetConfig


class EEGNet(nn.Module):
    """Compact CNN baseline factorizing convolution into temporal, spatial,
    and separable stages.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (input width).
    n_times : int
        Number of time samples per epoch (input length).
    n_classes : int
        Number of output classes.
    cfg : EEGNetConfig
        Architecture hyperparameters.
    """

    def __init__(self, n_channels, n_times, n_classes, cfg: EEGNetConfig):
        super().__init__()

        f1 = cfg.f1
        d  = cfg.depth_multiplier
        f2 = cfg.f2
        k  = cfg.kernel_length

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, f1, (1, k), padding=(0, k // 2), bias=False),
            nn.BatchNorm2d(f1),
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(cfg.dropout),
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(cfg.dropout),
        )

        flat_dim        = self._infer_flat_dim(n_channels, n_times)
        self.classifier = nn.Linear(flat_dim, n_classes)

    def _infer_flat_dim(self, n_channels, n_times):
        """Computes the flattened feature size via a dummy forward pass,
        avoiding manually tracked arithmetic through the pooling stages."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out   = self._features(dummy)
        return out.shape[1]

    def _features(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        return x.flatten(start_dim=1)

    def forward(self, x):
        """Runs the forward pass.

        Parameters
        ----------
        x : Tensor, shape (batch, n_channels, n_times)
            Batch of EEG epochs.

        Returns
        -------
        Tensor, shape (batch, n_classes)
            Class logits.
        """
        x = x.unsqueeze(1)    # (batch, channels, times) -> (batch, 1, channels, times)
        x = self._features(x)
        return self.classifier(x)
