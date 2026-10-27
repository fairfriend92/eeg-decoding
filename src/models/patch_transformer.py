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
import torch.nn.functional as F

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
    band_kernel : int, optional
        Length in samples of the stride-1 temporal filters that feed the
        band power tokens. 0 keeps the amplitude tokens.

    Raises
    ------
    ValueError
        If a group is empty or names a channel index outside n_channels, or
        if factorized_spatial is combined with channel_groups.
    """

    def __init__(self, n_channels, embed_dim, patch_size, patch_stride=None, channel_groups=None, channel_gate_reduction=0, factorized_spatial=False, band_kernel=0, multiband_kernels=None):
        super().__init__()
        patch_stride = patch_stride or patch_size
        self.band_kernel = band_kernel
        self.patch_size, self.patch_stride = patch_size, patch_stride
        if multiband_kernels and band_kernel:
            raise ValueError("multiband_kernels cannot be combined with band_kernel")
        if multiband_kernels:
            if any(k < patch_size or (k - patch_size) % 2 for k in multiband_kernels):
                raise ValueError(f"multiband_kernels must exceed patch_size={patch_size} by an even number of samples")
            widths = [embed_dim // len(multiband_kernels)] * len(multiband_kernels)
            widths[-1] += embed_dim - sum(widths)
            self.temporal_conv = nn.ModuleList([
                nn.Conv2d(1, w, (1, k), stride=(1, patch_stride), padding=(0, (k - patch_size) // 2), bias=False)
                for w, k in zip(widths, multiband_kernels)
            ])
        elif band_kernel:
            self.temporal_conv = nn.Conv2d(1, embed_dim, (1, band_kernel), bias=False)
        else:
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
        if isinstance(self.temporal_conv, nn.ModuleList):
            x = torch.cat([conv(x) for conv in self.temporal_conv], dim=1)
        else:
            x = self.temporal_conv(x)       # (batch, embed_dim, channels, n_patches)
        if self.channel_gate is not None:
            power = x.pow(2).mean(dim=1)                                  # (batch, channels, n_patches)
            power = power / (power.mean(dim=1, keepdim=True) + 1e-8)      # scale-free relative power
            gate  = torch.sigmoid(self.channel_gate(power.transpose(1, 2)))
            x     = x * gate.transpose(1, 2).unsqueeze(1)
        if self.channel_groups is None:
            x = self.spatial_conv(x)        # (batch, embed_dim, 1, n_patches)
        else:
            x = torch.cat([conv(x[:, :, g, :]) for conv, g in zip(self.spatial_conv, self.channel_groups)], dim=2)
        x = self.norm(x)
        if self.band_kernel:
            x = F.avg_pool2d(x.pow(2), (1, self.patch_size), (1, self.patch_stride))
            x = torch.log(x.clamp(min=1e-6))
        else:
            x = self.activation(x)          # (batch, embed_dim, n_groups, n_patches)
        return x.permute(0, 3, 2, 1).flatten(1, 2)


class CovarianceEmbedding(nn.Module):
    """Tokenizes EEG epochs via log-Euclidean covariance features of sliding windows.

    Each window's channel covariance is shrunk toward a scaled identity to
    keep it positive definite, mapped to the matrix logarithm, and its upper
    triangle (off-diagonals scaled by sqrt(2)) is projected linearly to the
    token dimension. The logarithm is taken at the identity, which is the
    natural reference once the trials are aligned. The covariance and its
    logarithm contain no learned weights, so they are computed without gradients in
    float32.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    embed_dim : int
        Token embedding dimension.
    window : int
        Number of time samples per covariance window.
    stride : int
        Step between window starts, in samples.
    shrinkage : float
        Weight of the scaled identity in the shrunk covariance, in [0, 1).
    """

    def __init__(self, n_channels, embed_dim, window, stride, shrinkage):
        super().__init__()
        self.window, self.stride, self.shrinkage = window, stride, shrinkage
        rows, cols = torch.triu_indices(n_channels, n_channels)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        scale = torch.where(rows == cols, 1.0, math.sqrt(2.0))
        self.register_buffer("scale", scale, persistent=False)
        self.projection = nn.Linear(len(rows), embed_dim, bias=False)
        self.norm       = nn.BatchNorm1d(embed_dim)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_channels, n_times)

        Returns
        -------
        Tensor, shape (batch, n_windows, embed_dim)
        """
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            w = x.float().unfold(2, self.window, self.stride)             # (batch, channels, n_windows, window)
            w = w.permute(0, 2, 1, 3)
            w = w - w.mean(dim=-1, keepdim=True)
            cov = w @ w.transpose(-1, -2) / self.window                   # (batch, n_windows, channels, channels)
            eye = torch.eye(cov.shape[-1], device=cov.device)
            trace = cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1)[..., None, None]
            cov = (1 - self.shrinkage) * cov + self.shrinkage * trace * eye
            # Batched eigh on small matrices is much faster with LAPACK on the CPU than with cuSOLVER.
            eigvals, eigvecs = torch.linalg.eigh(cov.cpu())
            log_cov = (eigvecs * torch.log(eigvals.clamp(min=1e-8)).unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
            log_cov = log_cov.to(x.device)
            features = log_cov[..., self.rows, self.cols] * self.scale    # (batch, n_windows, n_features)
        tokens = self.projection(features)
        batch, n_windows, dim = tokens.shape
        tokens = self.activation(self.norm(tokens.reshape(-1, dim))).reshape(batch, n_windows, dim)
        return tokens


class PhaseEmbedding(nn.Module):
    """Tokenizes EEG epochs via inter-channel phase synchrony of sliding windows.

    The trial is band-passed and Hilbert transformed in one step in the
    frequency domain, giving an analytic signal per band and channel. Each
    window's complex phase-locking value between every channel pair is the
    mean of exp(i * phase difference) over the window. Its real and
    imaginary parts (the cosine and sine of the mean phase difference,
    weighted by its stability) are projected linearly to the token dimension.
    The imaginary part vanishes for zero-lag coupling, which is how volume
    conduction appears, so it carries the lagged coupling alone. The
    transform contains no learned weights, so it is computed without
    gradients in float32.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_times : int
        Number of time samples per epoch.
    embed_dim : int
        Token embedding dimension.
    window : int
        Number of time samples per synchrony window.
    stride : int
        Step between window starts, in samples.
    pad : int
        Samples of zero phasor added at each end of the trial, so that a window
        longer than the main tokenizer's patch stays centered on it. Edge windows
        see fewer real samples, which lowers their synchrony magnitude.
    bands : list of list of float
        [low, high] edges in Hz of each frequency band.
    sfreq : float
        Sampling rate of the epochs, in Hz.

    Raises
    ------
    ValueError
        If a band is empty or reaches beyond the Nyquist frequency.
    """

    def __init__(self, n_channels, n_times, embed_dim, window, stride, pad, bands, sfreq):
        super().__init__()
        if any(not 0 <= low < high <= sfreq / 2 for low, high in bands):
            raise ValueError(f"phase_bands must satisfy 0 <= low < high <= {sfreq / 2} Hz, got {bands}")
        self.window, self.stride, self.pad = window, stride, pad
        rows, cols = torch.triu_indices(n_channels, n_channels, offset=1)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        freqs = torch.fft.fftfreq(n_times, 1.0 / sfreq)
        # Positive frequencies doubled and negative ones zeroed give the analytic signal.
        masks = torch.stack([2.0 * ((freqs >= low) & (freqs < high)).float() for low, high in bands])
        self.register_buffer("masks", masks, persistent=False)
        self.projection = nn.Linear(2 * len(bands) * len(rows), embed_dim, bias=False)
        self.norm       = nn.BatchNorm1d(embed_dim)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_channels, n_times)

        Returns
        -------
        Tensor, shape (batch, n_windows, embed_dim)
        """
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.fft(x.float(), dim=-1)                                      # (batch, channels, times)
            analytic = torch.fft.ifft(spectrum[:, None] * self.masks[None, :, None], dim=-1) # (batch, bands, channels, times)
            phasor   = analytic / analytic.abs().clamp(min=1e-8)
            phasor   = F.pad(phasor, (self.pad, self.pad))
            w = phasor.unfold(-1, self.window, self.stride).permute(0, 1, 3, 2, 4)          # (batch, bands, n_windows, channels, window)
            plv = w @ w.conj().transpose(-1, -2) / self.window                               # (batch, bands, n_windows, channels, channels)
            plv = plv[..., self.rows, self.cols]                                             # (batch, bands, n_windows, n_pairs)
            features = torch.cat([plv.real, plv.imag], dim=-1).permute(0, 2, 1, 3).flatten(2)
        tokens = self.projection(features)
        batch, n_windows, dim = tokens.shape
        tokens = self.activation(self.norm(tokens.reshape(-1, dim))).reshape(batch, n_windows, dim)
        return tokens


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
        band_kernel      = cfg.band_kernel
        n_filtered       = n_times - band_kernel + 1 if band_kernel else n_times
        window           = cfg.cov_window or cfg.patch_size
        if n_filtered < window:
            raise ValueError(f"{n_filtered} filtered samples are fewer than the token window of {window}")
        self.n_patches   = (n_filtered - window) // patch_stride + 1
        if cfg.cov_window:
            if band_kernel or cfg.multiband_kernels or cfg.channel_groups or cfg.channel_gate_reduction or cfg.factorized_spatial:
                raise ValueError("cov_window cannot be combined with the other tokenizer options")
            self.patch_embed = CovarianceEmbedding(n_channels, cfg.embed_dim, cfg.cov_window, patch_stride, cfg.cov_shrinkage)
        else:
            self.patch_embed = PatchEmbedding(n_channels, cfg.embed_dim, cfg.patch_size, patch_stride, cfg.channel_groups, cfg.channel_gate_reduction, cfg.factorized_spatial, band_kernel, cfg.multiband_kernels)
        self.n_groups    = len(cfg.channel_groups) if cfg.channel_groups else 1
        self.region_embed = nn.Parameter(torch.randn(self.n_groups, cfg.embed_dim) * 0.02) if cfg.channel_groups else None

        self.phase_embed = None
        if cfg.phase_window:
            if not cfg.phase_bands:
                raise ValueError("phase_window requires phase_bands")
            if cfg.channel_groups:
                raise ValueError("phase_window cannot be combined with channel_groups")
            if cfg.phase_window < window or (cfg.phase_window - window) % 2:
                raise ValueError(f"phase_window must exceed the token window of {window} samples by an even number of samples")
            pad = (cfg.phase_window - window) // 2
            n_phase = (n_times + 2 * pad - cfg.phase_window) // patch_stride + 1
            if n_phase != self.n_patches:
                raise ValueError(f"phase_window gives {n_phase} windows but the main tokenizer gives {self.n_patches}")
            self.phase_embed = PhaseEmbedding(n_channels, n_times, cfg.embed_dim, cfg.phase_window, patch_stride, pad, cfg.phase_bands, cfg.phase_sfreq)

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
        if self.phase_embed is not None:
            tokens = tokens + self.phase_embed(x)
        pe     = sinusoidal_positional_encoding(tokens.shape[1] // self.n_groups, tokens.shape[2], tokens.device)
        pe     = pe.repeat_interleave(self.n_groups, dim=0)              # one time position per window, shared by its groups
        if self.region_embed is not None:
            pe = pe + self.region_embed.repeat(tokens.shape[1] // self.n_groups, 1)
        tokens = tokens + pe.unsqueeze(0)
        tokens = self.encoder(tokens)
        pooled = tokens.mean(dim=1)                                      # average over patch tokens
        return self.classifier(self.dropout(pooled))
