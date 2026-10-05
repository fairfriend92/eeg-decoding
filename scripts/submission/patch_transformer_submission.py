"""Codabench submission for Track 02 (BCI decoding): this project's PatchTransformer.

A submission ZIP may only contain ``submission.py`` plus non-Python weight
files (no local package imports), so this file inlines a copy of
``src/models/patch_transformer.py``'s ``PatchTransformer`` (plus its ``PatchEmbedding``
helper and ``sinusoidal_positional_encoding`` function) with
``src/config.py``'s ``PatchTransformerConfig`` defaults hardcoded as constructor
arguments. Keep the three in sync by hand; nothing enforces it
automatically.

``weights.pt`` (built by ``scripts/neuralbench_eval.py --save-checkpoint``,
packaged by ``build_submission.py --model patch_transformer``) must hold
``{"net": ..., "probe": ...}`` state dicts. The ``probe`` is not optional
decoration: NeuralBench's ``finetune_flatten`` downstream wrapper (the one
``scripts/neuralbench_eval.py`` trains through) adds a trainable
``nn.Linear(n_classes, n_classes)`` on top of the backbone's own output and
trains both together, so predictions without it do not match the trained
model. See ``neuralbench.modules.DownstreamWrapperModel``/``DownstreamWrapper``
(``probe_config="linear"`` in ``neuralbench/defaults/downstream_wrappers.yaml``).
"""

import math

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver

# PatchTransformerConfig defaults (src/config.py), inlined: see module docstring.
_PATCH_SIZE = 25
_EMBED_DIM = 64
_N_HEADS = 8
_N_LAYERS = 4
_FF_DIM = 256
_DROPOUT = 0.5


class PatchEmbedding(nn.Module):
    """Tokenizes EEG epochs via temporal patches, followed by a shared
    spatial filter across channels. Verbatim copy of
    ``src/models/patch_transformer.py``'s class; see there for the full docstring.
    """

    def __init__(self, n_channels, embed_dim, patch_size):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, embed_dim, (1, patch_size), stride=(1, patch_size), bias=False)
        self.spatial_conv  = nn.Conv2d(embed_dim, embed_dim, (n_channels, 1), bias=False)
        self.norm          = nn.BatchNorm2d(embed_dim)
        self.activation    = nn.ELU()

    def forward(self, x):
        x = x.unsqueeze(1)                  # (batch, 1, channels, times)
        x = self.temporal_conv(x)           # (batch, embed_dim, channels, n_patches)
        x = self.spatial_conv(x)            # (batch, embed_dim, 1, n_patches)
        x = self.activation(self.norm(x))
        x = x.squeeze(2)                    # (batch, embed_dim, n_patches)
        return x.transpose(1, 2)            # (batch, n_patches, embed_dim)


def sinusoidal_positional_encoding(seq_len, embed_dim, device):
    """Computes fixed sinusoidal position encodings. Verbatim copy of
    ``src/models/patch_transformer.py``'s function; see there for the full
    docstring."""
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, device=device) * (-math.log(10000.0) / embed_dim)
    )
    pe          = torch.zeros(seq_len, embed_dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PatchTransformer(nn.Module):
    """Transformer encoder for EEG motor imagery classification. Verbatim
    copy of ``src/models/patch_transformer.py``'s architecture; see there for the
    full docstring and rationale.
    """

    def __init__(self, n_channels, n_times, n_classes):
        super().__init__()

        self.patch_embed = PatchEmbedding(n_channels, _EMBED_DIM, _PATCH_SIZE)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=_EMBED_DIM,
            nhead=_N_HEADS,
            dim_feedforward=_FF_DIM,
            dropout=_DROPOUT,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=_N_LAYERS)

        self.dropout    = nn.Dropout(_DROPOUT)
        self.classifier = nn.Linear(_EMBED_DIM, n_classes)

    def forward(self, x, channel_positions=None, ch_names=None):
        tokens = self.patch_embed(x)                                     # (batch, n_patches, embed_dim)
        pe     = sinusoidal_positional_encoding(tokens.shape[1], tokens.shape[2], tokens.device)
        tokens = tokens + pe.unsqueeze(0)
        tokens = self.encoder(tokens)
        pooled = tokens.mean(dim=1)                                      # average over patch tokens
        return self.classifier(self.dropout(pooled))


class _ProbedPatchTransformer(nn.Module):
    """PatchTransformer plus the linear probe NeuralBench's downstream wrapper
    trains on top of it (see module docstring)."""

    def __init__(self, n_chans, n_times, n_classes):
        super().__init__()
        self.net = PatchTransformer(n_chans, n_times, n_classes)
        self.probe = nn.Linear(n_classes, n_classes)

    def forward(self, x):
        return self.probe(self.net(x))

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        return self(X).argmax(dim=1)  # (B,)


class Solver(CompetSolver):

    name = "OursPatchTransformer"

    def load_model(self, meta):
        model = _ProbedPatchTransformer(
            n_chans=meta["n_chans"], n_times=meta["n_times"], n_classes=meta["n_classes"],
        )
        weights = meta["submission_dir"] / "weights.pt"
        state = torch.load(weights, map_location=meta["device"], weights_only=True)
        model.net.load_state_dict(state["net"])
        model.probe.load_state_dict(state["probe"])
        return model.to(meta["device"]).eval()
