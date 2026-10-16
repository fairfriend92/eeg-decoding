"""Codabench submission for Track 02 (BCI decoding): this project's EEGNet.

A submission ZIP may only contain ``submission.py`` plus non-Python weight
files (no local package imports), so this file inlines a copy of
``src/models/eegnet.py``'s ``EEGNet`` with ``src/config.py``'s
``EEGNetConfig`` defaults hardcoded as constructor arguments. Keep the two in
sync by hand; nothing enforces it automatically.

An optional ``align`` entry in ``weights.pt`` switches on a causal per-stream
z-score and Euclidean alignment of each window (``CausalAlign``, inlined from
``src/causal_alignment.py`` by ``build_submission.py``). Without it the model
sees the raw window.

``weights.pt`` (built by ``scripts/neuralbench_eval.py --save-checkpoint``,
packaged by ``build_submission.py``) must hold ``{"net": ..., "probe": ...}``
state dicts. The ``probe`` is not optional decoration: NeuralBench's
``finetune_flatten`` downstream wrapper (the one
``scripts/neuralbench_eval.py`` trains through) adds a trainable
``nn.Linear(n_classes, n_classes)`` on top of the backbone's own output and
trains both together, so predictions without it do not match the trained
model. See ``neuralbench.modules.DownstreamWrapperModel``/``DownstreamWrapper``
(``probe_config="linear"`` in ``neuralbench/defaults/downstream_wrappers.yaml``).
"""

import numpy as np
import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver

# BEGIN causal_alignment (build_submission.py inlines src/causal_alignment.py's CausalAlign here)
# END causal_alignment


def _make_aligner(spec):
    """CausalAlign from the optional ``align`` entry of weights.pt, None when absent.

    The entry is a dict with optional ``mean`` and ``var`` tensors (per-channel
    training-pool statistics) and the float prior weights ``z_prior_windows``
    and ``cov_prior_windows``.
    """
    if spec is None:
        return None
    return CausalAlign(
        prior_mean=spec["mean"].cpu().numpy() if "mean" in spec else None,
        prior_var=spec["var"].cpu().numpy() if "var" in spec else None,
        z_prior_windows=spec.get("z_prior_windows", 0.0),
        cov_prior_windows=spec.get("cov_prior_windows", 1.0),
    )

# EEGNetConfig defaults (src/config.py), inlined: see module docstring.
_F1 = 8
_DEPTH_MULTIPLIER = 2
_F2 = 16
_KERNEL_LENGTH = 125
_DROPOUT = 0.25


class EEGNet(nn.Module):
    """Compact CNN baseline factorizing convolution into temporal, spatial,
    and separable stages (Lawhern et al., 2018). Verbatim copy of
    ``src/models/eegnet.py``'s architecture; see there for the full
    docstring and rationale.
    """

    def __init__(self, n_channels, n_times, n_classes):
        super().__init__()

        f1 = _F1
        d = _DEPTH_MULTIPLIER
        f2 = _F2
        k = _KERNEL_LENGTH

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, f1, (1, k), padding=(0, k // 2), bias=False),
            nn.BatchNorm2d(f1),
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(_DROPOUT),
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(_DROPOUT),
        )

        flat_dim = self._infer_flat_dim(n_channels, n_times)
        self.classifier = nn.Linear(flat_dim, n_classes)

    def _infer_flat_dim(self, n_channels, n_times):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self._features(dummy)
        return out.shape[1]

    def _features(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        return x.flatten(start_dim=1)

    def forward(self, x, channel_positions=None, ch_names=None):
        x = x.unsqueeze(1)  # (batch, channels, times) -> (batch, 1, channels, times)
        x = self._features(x)
        return self.classifier(x)


class _ProbedEEGNet(nn.Module):
    """EEGNet plus the linear probe NeuralBench's downstream wrapper trains
    on top of it (see module docstring)."""

    def __init__(self, n_chans, n_times, n_classes, aligner=None):
        super().__init__()
        self.aligner = aligner
        self.net = EEGNet(n_chans, n_times, n_classes)
        self.probe = nn.Linear(n_classes, n_classes)

    def reset_state(self):
        """Forgets the stream's running statistics. Called before each stream."""
        if self.aligner is not None:
            self.aligner.reset()

    def forward(self, x):
        if self.aligner is not None:
            # Windows are aligned one at a time, in order, so a batch is the same stream.
            x = torch.stack([torch.from_numpy(self.aligner(w.detach().cpu().numpy())) for w in x]).to(x.device, x.dtype)
        return self.probe(self.net(x))

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        return self(X).argmax(dim=1)  # (B,)


class Solver(CompetSolver):

    name = "OursEEGNet"

    def load_model(self, meta):
        model = _ProbedEEGNet(
            n_chans=meta["n_chans"], n_times=meta["n_times"], n_classes=meta["n_classes"],
        )
        weights = meta["submission_dir"] / "weights.pt"
        state = torch.load(weights, map_location=meta["device"], weights_only=True)
        model.net.load_state_dict(state["net"])
        model.probe.load_state_dict(state["probe"])
        model.aligner = _make_aligner(state.get("align"))
        return model.to(meta["device"]).eval()
