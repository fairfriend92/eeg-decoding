"""Batch-level EEG augmentation applied while training.

Every function works on a (batch, n_channels, n_times) tensor on the training
device. Randomness comes from a CPU generator so a run is reproducible whatever
the device.
"""

import numpy as np
import torch

from config import AugmentConfig


def _time_shift(x, max_shift, generator):
    """Shifts each window by its own random number of samples, zero filling the gap."""
    n_times = x.shape[-1]
    shift = torch.randint(-max_shift, max_shift + 1, (x.shape[0], 1), generator=generator).to(x.device)
    idx = torch.arange(n_times, device=x.device).unsqueeze(0) - shift
    valid = ((idx >= 0) & (idx < n_times)).unsqueeze(1)
    return torch.gather(x, 2, idx.clamp(0, n_times - 1).unsqueeze(1).expand_as(x)) * valid


def _segment_recombine(x, y, n_segments, generator):
    """Swaps each segment with the same segment of a random same-class window."""
    out = x.clone()
    bounds = torch.linspace(0, x.shape[-1], n_segments + 1).round().long().tolist()
    for label in y.unique():
        members = (y == label).nonzero(as_tuple=True)[0]
        for start, stop in zip(bounds[:-1], bounds[1:]):
            donors = members[torch.randperm(len(members), generator=generator).to(x.device)]
            out[members, :, start:stop] = x[donors, :, start:stop]
    return out


def augment_batch(x, y, cfg: AugmentConfig, sample_rate, generator):
    """Applies the techniques enabled in cfg to one training batch.

    Parameters
    ----------
    x : Tensor, shape (batch, n_channels, n_times)
        Training windows.
    y : Tensor, shape (batch,)
        Window labels, used only by segmentation and recombination.
    cfg : AugmentConfig
        Enabled techniques and their strengths. All off returns x unchanged.
    sample_rate : float
        Sampling rate in Hz, converting time_shift_ms to samples.
    generator : torch.Generator
        CPU generator that drives every random draw.

    Returns
    -------
    x : Tensor, shape (batch, n_channels, n_times)
        Augmented windows. With keep_original, the clean windows followed by
        their augmented copies, shuffled together, so the batch is doubled.
    y : Tensor, shape (batch,)
        Labels matching x.
    """
    clean = x
    if cfg.sr_segments:
        x = _segment_recombine(x, y, cfg.sr_segments, generator)
    if cfg.time_shift_ms:
        x = _time_shift(x, round(cfg.time_shift_ms * sample_rate / 1000), generator)
    if cfg.amp_scale:
        scale = 1 + cfg.amp_scale * (2 * torch.rand(x.shape[0], 1, 1, generator=generator) - 1)
        x = x * scale.to(x.device)
    if cfg.noise_std:
        noise = torch.randn(x.shape, generator=generator).to(x.device)
        x = x + cfg.noise_std * x.std(dim=(1, 2), keepdim=True) * noise
    if cfg.keep_original:
        x, y = torch.cat([clean, x]), torch.cat([y, y])
        order = torch.randperm(len(y), generator=generator).to(x.device)
        x, y = x[order], y[order]
    return x, y


def mixup_batch(x, y, alpha, n_classes, generator):
    """Blends every window with a random partner of the same batch.

    Parameters
    ----------
    x : Tensor, shape (batch, n_channels, n_times)
        Training windows.
    y : Tensor, shape (batch,)
        Integer labels.
    alpha : float
        Concentration of the Beta(alpha, alpha) blend weight.
    n_classes : int
        Number of classes, sizing the soft labels.
    generator : torch.Generator
        CPU generator that drives the partner choice and the blend weights.

    Returns
    -------
    x_mixed : Tensor, shape (batch, n_channels, n_times)
        Blended windows.
    y_mixed : Tensor, shape (batch, n_classes)
        Blended one-hot labels.
    """
    partner = torch.randperm(len(y), generator=generator).to(x.device)
    rng = np.random.default_rng(int(torch.randint(2**31, (1,), generator=generator)))
    lam = torch.from_numpy(rng.beta(alpha, alpha, size=len(y))).float().to(x.device)
    x_mixed = lam.view(-1, 1, 1) * x + (1 - lam.view(-1, 1, 1)) * x[partner]
    onehot = torch.nn.functional.one_hot(y, n_classes).float()
    return x_mixed, lam.view(-1, 1) * onehot + (1 - lam.view(-1, 1)) * onehot[partner]
