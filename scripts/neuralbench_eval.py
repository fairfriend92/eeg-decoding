"""Runs one of our models through NeuralBench's own harness.

NeuralBench (https://github.com/facebookresearch/neuroai/tree/main/neuralbench-repo)
takes an already-built torch.nn.Module via neuralbench.check_model /
neuralbench.evaluate_model, no repo-side registration needed. See
docs/SESSION_HANDOFF.md for why this exists: our own dreyer2023 accuracy
numbers use different preprocessing and aren't comparable to what the
NeurIPS 2026 EEG/EMG Foundation Challenge will actually score.

Builds a freshly initialized model, not one of our trained checkpoints: a
checkpoint's classifier layer is shape-locked to our own preprocessing's
(n_channels, n_times), which NeuralBench's own preprocessing does not
reproduce (see the Lazy*Adapter classes below).

Must be run with src/ importable; this file inserts its repo's sibling
src/ onto sys.path itself, so it works both from a local checkout and
after scripts/colab_neuralbench.sh uploads it alongside src/ on Colab.
"""

import argparse
import sys
from pathlib import Path

# globals().get fallback: colab_neuralbench.sh's bootstrap runs this file's
# source through exec(), where __file__ is not defined; cwd is REMOTE_ROOT/
# scripts there (see the bootstrap's chdir), so 'src' still resolves as a
# sibling of it.
_here = Path(globals().get("__file__", Path.cwd() / "neuralbench_eval.py")).resolve()
sys.path.insert(0, str(_here.parent.parent / "src"))

import torch.nn as nn

from config import EEGNetConfig, ConformerConfig
from models.eegnet import EEGNet
from models.conformer import EEGConformer


class LazyEEGNetAdapter(nn.Module):
    """Builds a real EEGNet sized to the first batch it sees, then delegates.

    EEGNet's constructor fixes n_channels/n_times into its classifier layer
    (see EEGNet._infer_flat_dim), but NeuralBench probes several montage
    widths in check_model and only reveals a task's real shape at the first
    forward call. Rebuilding whenever the observed shape changes mirrors
    NeuralBench's own model-building contract: neuralbench.model_factory.
    init_lazy_layers already runs one dummy forward pass before real
    training starts specifically to materialize shape-dependent layers.
    """

    def __init__(self, n_classes, cfg: EEGNetConfig):
        super().__init__()
        self.n_classes    = n_classes
        self.cfg          = cfg
        self.net          = None
        self._built_shape = None

    def forward(self, x, channel_positions=None, ch_names=None):
        shape = tuple(x.shape[1:])
        if shape != self._built_shape:
            self.net          = EEGNet(shape[0], shape[1], self.n_classes, self.cfg).to(x.device)
            self._built_shape = shape
        return self.net(x, channel_positions=channel_positions, ch_names=ch_names)


class LazyConformerAdapter(nn.Module):
    """Builds a real EEGConformer sized to the first batch it sees.

    Same rebuild-on-shape-change contract as LazyEEGNetAdapter, since
    EEGConformer's PatchEmbedding also fixes n_channels at construction.
    """

    def __init__(self, n_classes, cfg: ConformerConfig):
        super().__init__()
        self.n_classes    = n_classes
        self.cfg          = cfg
        self.net          = None
        self._built_shape = None

    def forward(self, x, channel_positions=None, ch_names=None):
        shape = tuple(x.shape[1:])
        if shape != self._built_shape:
            self.net          = EEGConformer(shape[0], shape[1], self.n_classes, self.cfg).to(x.device)
            self._built_shape = shape
        return self.net(x, channel_positions=channel_positions, ch_names=ch_names)


MODEL_ADAPTERS = {
    "eegnet": (LazyEEGNetAdapter, EEGNetConfig),
    "conformer": (LazyConformerAdapter, ConformerConfig),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_ADAPTERS), default="eegnet")
    parser.add_argument("--task", default="motor_imagery")
    parser.add_argument("--dataset", default="dreyer2023")
    parser.add_argument(
        "--n-classes", type=int, default=2,
        help="NeuralBench's own label count for --task/--dataset, not necessarily "
             "src/datasets.py's n_classes preset for the same dataset name.",
    )
    parser.add_argument(
        "--downstream-wrapper", default="finetune_flatten",
        help="finetune_* unfreezes the whole model, matching how train.py trains "
             "our models end to end (the default linear_probe_mean instead freezes "
             "it, correct only for a pretrained foundation model). '_flatten' "
             "aggregation is a no-op reshape for a (batch, n_classes) output; '_mean' "
             "would incorrectly average across classes, so avoid it here.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Run check_model's fast synthetic-shape probe and exit, skip evaluate_model.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Reduced epochs/batches, forced local execution. A wiring check, not a "
             "real score; see evaluate_model's own docstring.",
    )
    args = parser.parse_args()

    from neuralbench import check_model, evaluate_model

    adapter_cls, cfg_cls = MODEL_ADAPTERS[args.model]
    model = adapter_cls(args.n_classes, cfg_cls())

    print("== check_model ==")
    print(check_model(model, "eeg", args.task, dataset=args.dataset).to_string())

    if args.check_only:
        return

    print("== evaluate_model ==")
    scores = evaluate_model(
        model, "eeg", args.task,
        name=f"ours-{args.model}",
        dataset=args.dataset,
        downstream_wrapper=args.downstream_wrapper,
        debug=args.debug,
    )
    print(scores.to_string())


if __name__ == "__main__":
    main()
