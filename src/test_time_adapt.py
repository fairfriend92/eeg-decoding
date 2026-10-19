"""Label-free test-time adaptation of a fitted model, replayed window by window.

Each stream (one participant session) is scored one window at a time, in
recording order, the way the Codabench scorer calls predict. A window is
adapted on once it arrives and the adapted state is discarded when the stream
ends. With an inner k-fold ensemble, every member is replayed under the
variant and the members' probabilities are averaged. Three variants share the same fitted model and the same causally aligned
windows: "none" (no adaptation), "abn" (BatchNorm statistics re-estimated from
the windows seen so far) and "tent" (one entropy-minimization step per window
on the BatchNorm affine parameters). Writes
output/{dataset}/cross_subject/predictions_{member}{suffix}_tta_{variant}{tag}{fold}.npz
per variant, in the layout analyze_causal_replay.py reads.

Must be run with this file's own directory (src/) as the script's
directory. See train.py for why.
"""

import argparse
import copy
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import Config, add_dataset_cli_args, resolve_config
from cross_subject_members import alignment_suffix
from data_loader import get_cross_subject_data, participant_folds
from train import fit_full, fit_on_split, output_dir_for

VARIANTS = ("none", "abn", "tent")


class _RunningBatchNorm:
    """Forward pre-hook that sets a BatchNorm layer's statistics from the stream so far.

    The statistics are the pooled mean and second moment of every window seen,
    this one included, shrunk toward the layer's stored statistics by a prior
    worth a number of windows. The layer must be in eval mode, so it normalizes
    with the buffers this hook writes.

    Parameters
    ----------
    norm : nn.BatchNorm2d
        Layer to adapt.
    prior_windows : float
        Weight of the stored statistics, in windows.
    """

    def __init__(self, norm, prior_windows):
        self.prior_mean = norm.running_mean.clone()
        self.prior_sq = norm.running_var + norm.running_mean ** 2
        self.prior_windows = float(prior_windows)
        self.n_windows = 0
        self.sum_mean = torch.zeros_like(self.prior_mean)
        self.sum_sq = torch.zeros_like(self.prior_mean)
        norm.register_forward_pre_hook(self)

    def __call__(self, norm, inputs):
        x = inputs[0].detach()
        self.n_windows += 1
        self.sum_mean += x.mean(dim=(0, 2, 3))
        self.sum_sq += (x ** 2).mean(dim=(0, 2, 3))
        weight = self.prior_windows + self.n_windows
        mean = (self.prior_windows * self.prior_mean + self.sum_mean) / weight
        var = (self.prior_windows * self.prior_sq + self.sum_sq) / weight - mean ** 2
        norm.running_mean.copy_(mean)
        norm.running_var.copy_(var.clamp_min(0))


def _entropy(logits):
    log_probs = torch.log_softmax(logits, dim=1)
    return -(log_probs.exp() * log_probs).sum(dim=1).mean()


def replay_stream(model, X, variant, lr, prior_windows, device):
    """Scores one stream window by window under one adaptation variant.

    Parameters
    ----------
    model : nn.Module
        Fitted model with BatchNorm layers. Left unmodified, a copy is adapted.
    X : ndarray, shape (n_windows, n_channels, n_times)
        The stream's windows in recording order.
    variant : str
        One of VARIANTS.
    lr : float
        Adam learning rate of the "tent" variant.
    prior_windows : float
        Weight, in windows, of the stored BatchNorm statistics in "abn".

    Returns
    -------
    ndarray, shape (n_windows, n_classes)
        Softmax probabilities, each from the state after adapting on that window
        ("abn") or before the step on it ("tent").

    Raises
    ------
    ValueError
        If variant is not in VARIANTS.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choices: {VARIANTS}")
    adapted = copy.deepcopy(model).to(device).eval()
    norms = [m for m in adapted.modules() if isinstance(m, nn.BatchNorm2d)]
    if variant == "abn":
        for norm in norms:
            _RunningBatchNorm(norm, prior_windows)
    if variant == "tent":
        for p in adapted.parameters():
            p.requires_grad_(False)
        params = [p for norm in norms for p in (norm.weight, norm.bias)]
        for p in params:
            p.requires_grad_(True)
        optimizer = torch.optim.Adam(params, lr=lr)

    probs = []
    for window in X:
        x = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
        if variant == "tent":
            logits = adapted(x)
            optimizer.zero_grad()
            _entropy(logits).backward()
            optimizer.step()
            probs.append(torch.softmax(logits.detach(), dim=1).cpu().numpy()[0])
        else:
            with torch.no_grad():
                probs.append(torch.softmax(adapted(x), dim=1).cpu().numpy()[0])
    return np.stack(probs)


def replay_all(model, data, variant, lr, prior_windows, device):
    """Replays every (participant, session) stream of the test rows.

    Parameters
    ----------
    model : nn.Module
        Fitted model.
    data : dict
        As returned by data_loader.get_cross_subject_data. Rows of a stream
        are in recording order.
    variant, lr, prior_windows
        See replay_stream.

    Returns
    -------
    probs : ndarray, shape (n_test, n_classes)
        In the row order of data.
    seconds_per_window : float
        Wall time of the replay divided by the number of windows.
    """
    keys = np.char.add(data["subject_test"].astype(str), "/" + data["session_test"].astype(str))
    probs = np.zeros((len(keys), 3), dtype=np.float32)
    start = time.perf_counter()
    for key in dict.fromkeys(keys):
        idx = np.where(keys == key)[0]
        probs[idx] = replay_stream(model, data["X_test"][idx], variant, lr, prior_windows, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return probs, (time.perf_counter() - start) / len(keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", default="patch_transformer", choices=["eegnet", "patch_transformer", "conformer"])
    parser.add_argument("--n-folds", type=int, default=0, help="Cross-validate over this many disjoint test-participant folds. 0 runs the preset split.")
    parser.add_argument("--n-jobs", type=int, default=6, help="Workers epoching participants missing from the cache.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Persistent directory for the output and epoch caches, shared across sessions.")
    parser.add_argument("--exclude-subjects", type=int, nargs="*", default=[], help="Participants left out of the pool entirely, neither trained on nor tested.")
    parser.add_argument("--epochs", type=int, default=None, help="Fixed epoch count of the final fit. Ignored with --kfold.")
    parser.add_argument("--kfold", type=int, default=0, help="Fit a participant-grouped k-fold ensemble per fold (see cross_subject_members.dump_kfold_ensemble) instead of one fixed-epoch model.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate of the tent variant.")
    parser.add_argument("--prior-windows", type=float, default=10.0, help="Weight, in windows, of the stored BatchNorm statistics in the abn variant.")
    parser.add_argument("--tag", default="", help="Extra dump file-name text placed before the fold index.")
    add_dataset_cli_args(parser)
    args = parser.parse_args()

    if not args.kfold and args.epochs is None:
        raise ValueError("--epochs is required without --kfold.")
    cfg = resolve_config(args)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, subject_ids=[s for s in cfg.data.subject_ids if s not in args.exclude_subjects]))
    cache_dir = args.cache_dir / "epochs_cache" if args.cache_dir else None
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    folds = participant_folds(cfg.data, args.n_folds) if args.n_folds else [cfg.data.test_subject_ids]
    for k, test_ids in enumerate(folds):
        fold_cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, test_subject_ids=test_ids))
        data = get_cross_subject_data(fold_cfg.data, n_jobs=args.n_jobs, cache_dir=cache_dir)
        X_all = np.concatenate([data["X_fit"], data["X_val"]])
        y_all = np.concatenate([data["y_fit"], data["y_val"]])
        if args.kfold:
            subject_all = np.concatenate([data["subject_fit"], data["subject_val"]])
            shuffled = np.random.default_rng(fold_cfg.data.val_subject_seed).permutation(np.unique(subject_all))
            models = []
            for j, held_out in enumerate(np.array_split(shuffled, args.kfold)):
                mask = np.isin(subject_all, held_out)
                model, acc, best_epoch = fit_on_split(
                    X_all[~mask], y_all[~mask], X_all[mask], y_all[mask], args.member, fold_cfg, seed=fold_cfg.train.seed + j)
                models.append(model)
                print(f"[{args.member}] inner fold {j}: held out {sorted(int(s) for s in held_out)}, held-out acc={acc:.3f}, best epoch={best_epoch}", flush=True)
        else:
            models = [fit_full(X_all, y_all, args.member, fold_cfg, args.epochs)]
        out_dir = output_dir_for(fold_cfg) / "cross_subject"
        out_dir.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS:
            replays = [replay_all(model, data, variant, args.lr, args.prior_windows, device) for model in models]
            probs = np.mean([r[0] for r in replays], axis=0)
            seconds = sum(r[1] for r in replays)
            dump = {
                "subject": data["subject_test"], "y": data["y_test"], "session": data["session_test"],
                "probs": probs, "seconds_per_window": np.array(seconds),
            }
            np.savez(out_dir / f"predictions_{args.member}{alignment_suffix(fold_cfg)}_tta_{variant}{args.tag}{'' if not args.n_folds else f'_f{k}'}.npz", **dump)
            print(f"[{args.member}] fold {k} {variant}: test acc={(probs.argmax(axis=1) == data['y_test']).mean():.3f}, {1000 * seconds:.1f} ms/window", flush=True)
