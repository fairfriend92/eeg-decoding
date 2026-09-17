"""CSP-based initialization for EEGNet's depthwise spatial filters.

EEGNet's spatial_conv is a depthwise Conv2d: each of the f1 temporal-filter
output channels gets `depth_multiplier` learned spatial filters over the
22 EEG channels (see models/eegnet.py). Left at its default random init,
these filters start with no information about which channel combinations
actually discriminate the motor-imagery classes.

Common Spatial Patterns (CSP) computes exactly that: per (temporal-band,
class-set) spatial filters that maximize variance ratio between classes.
This module runs CSP separately on each of the f1 temporal-filter outputs
(using the model's own, randomly-initialized-but-fixed temporal_conv as the
band-pass bank) and writes the resulting filters into spatial_conv's weight
as an initialization -- training then proceeds as usual via backprop, this
only replaces the starting point.
"""

import numpy as np
import torch
from mne.decoding import CSP


def csp_initialize(model, X_train, y_train, device):
    """Overwrites an EEGNet's spatial_conv weights with per-band CSP filters.

    Parameters
    ----------
    model : EEGNet
        Freshly constructed model (temporal_conv still at its random init --
        used here as a fixed filter bank, not yet trained).
    X_train, y_train : ndarray
        Training data used to fit CSP, shape (n_trials, n_channels, n_times)
        and (n_trials,). Should be whatever the model is about to train on
        (crops, if cropped training is enabled), so the spatial statistics
        match the actual training distribution.
    device : torch.device
        Device the model lives on.

    Returns
    -------
    EEGNet
        The same model instance, mutated in place (spatial_conv.weight
        replaced); returned for convenience.
    """
    f1 = model.temporal_conv[0].out_channels
    out_channels, _, n_channels, _ = model.spatial_conv[0].weight.shape
    d = out_channels // f1

    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_train, dtype=torch.float32, device=device).unsqueeze(1)
        band_signals = model.temporal_conv(x).cpu().numpy()    # (n_trials, f1, n_channels, n_times)

    new_weight = np.zeros((out_channels, 1, n_channels, 1), dtype=np.float32)
    for band in range(f1):
        csp = CSP(n_components=d, reg="ledoit_wolf", log=False, norm_trace=False)
        csp.fit(band_signals[:, band, :, :].astype(np.float64), y_train)
        filters = csp.filters_[:d]                              # (d, n_channels)
        filters = filters / (np.linalg.norm(filters, axis=1, keepdims=True) + 1e-8)
        new_weight[band * d:(band + 1) * d, 0, :, 0] = filters

    model.spatial_conv[0].weight.data.copy_(torch.tensor(new_weight, device=device))
    return model
