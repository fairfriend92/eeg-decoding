"""Cross-subject ensemble members for datasets with a whole-participant split.

Fits each member on pooled participants and writes the per-trial probability
dump the ensemble harness reads, to output/{dataset}/cross_subject/
predictions_{member}.npz. Test probabilities come from a fit on every
training participant (fitting plus validation). Validation probabilities
come from a fit on the fitting participants only, so ensemble selection never
sees validation labels. The split is data_loader.get_cross_subject_data.

Must be run with this file's own directory (src/) as the script's
directory. See train.py for why.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from config import Config, add_dataset_cli_args, resolve_config
from data_loader import get_cross_subject_data
from fbcsp_member import fit_predict_fbcsp
from riemann_member import fit_predict_riemann
from train import compute_matched_epochs, fit, fit_full, output_dir_for, predict_probs


def _riemann_probs(data, cfg: Config):
    X_all = np.concatenate([data["X_fit"], data["X_val"]])
    y_all = np.concatenate([data["y_fit"], data["y_val"]])
    subject_all = np.concatenate([data["subject_fit"], data["subject_val"]])
    recenter = cfg.data.alignment == "riemannian"
    n_trials = cfg.data.alignment_n_trials
    return (
        fit_predict_riemann(
            data["X_fit"], data["y_fit"], data["X_val"],
            data["subject_fit"] if recenter else None, data["subject_val"] if recenter else None, n_trials),
        fit_predict_riemann(
            X_all, y_all, data["X_test"],
            subject_all if recenter else None, data["subject_test"] if recenter else None, n_trials),
    )


def _fbcsp_probs(data, cfg: Config):
    X_all = np.concatenate([data["X_fit"], data["X_val"]])
    y_all = np.concatenate([data["y_fit"], data["y_val"]])
    return (
        fit_predict_fbcsp(data["X_fit"], data["y_fit"], data["X_val"], cfg),
        fit_predict_fbcsp(X_all, y_all, data["X_test"], cfg),
    )


def _neural_probs(data, cfg: Config, model_name):
    X_all = np.concatenate([data["X_fit"], data["X_val"]])
    y_all = np.concatenate([data["y_fit"], data["y_val"]])
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    batch_size = cfg.train.batch_size

    # fit() holds out a fifth of the fitting trials for early stopping, so the
    # epoch budget is matched on the trials it actually trained on.
    model, _, best_epoch = fit(data["X_fit"], data["y_fit"], model_name, cfg)
    val_probs = predict_probs(model, data["X_val"], device, batch_size)
    n_fit_train = len(data["X_fit"]) - round(len(data["X_fit"]) * 0.2)
    n_epochs = compute_matched_epochs([(best_epoch, n_fit_train)], len(X_all), batch_size)

    final = fit_full(X_all, y_all, model_name, cfg, n_epochs)
    return val_probs, predict_probs(final, data["X_test"], device, batch_size)


MEMBERS = {
    "riemann": _riemann_probs,
    "fbcsp": _fbcsp_probs,
    "eegnet": lambda data, cfg: _neural_probs(data, cfg, "eegnet"),
    "patch_transformer": lambda data, cfg: _neural_probs(data, cfg, "patch_transformer"),
    "conformer": lambda data, cfg: _neural_probs(data, cfg, "conformer"),
}


def alignment_suffix(cfg: Config):
    """File-name suffix of a dump, empty for the unaligned setting."""
    suffix = {"none": "", "euclidean": "_ea", "riemannian": "_ra"}[cfg.data.alignment]
    if suffix and cfg.data.alignment_n_trials:
        suffix += f"_n{cfg.data.alignment_n_trials}"
    return suffix


def dump_cross_subject_member(member, data, cfg: Config):
    """Fits one member and writes its probability dump.

    Parameters
    ----------
    member : str
        Key of MEMBERS.
    data : dict
        As returned by data_loader.get_cross_subject_data.
    cfg : Config
        Full project configuration.

    Returns
    -------
    dict
        Arrays "subject", "y" and "probs" for the test rows, and
        "val_subject", "val_y" and "val_probs" for the validation rows.
    """
    if cfg.data.alignment == "riemannian" and member != "riemann":
        raise ValueError("Riemannian alignment applies to the riemann member only.")
    val_probs, probs = MEMBERS[member](data, cfg)
    dump = {
        "subject": data["subject_test"], "y": data["y_test"], "probs": probs,
        "val_subject": data["subject_val"], "val_y": data["y_val"], "val_probs": val_probs,
    }
    out_dir = output_dir_for(cfg) / "cross_subject"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"predictions_{member}{alignment_suffix(cfg)}.npz", **dump)
    print(f"[{member}] test acc={(probs.argmax(axis=1) == data['y_test']).mean():.3f}, "
          f"val acc={(val_probs.argmax(axis=1) == data['y_val']).mean():.3f}", flush=True)
    return dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("members", nargs="+", choices=list(MEMBERS))
    parser.add_argument("--n-jobs", type=int, default=6, help="Workers epoching participants missing from the cache.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Persistent directory for the output and epoch caches, shared across sessions.")
    add_dataset_cli_args(parser)
    args = parser.parse_args()

    cfg = resolve_config(args)
    cache_dir = args.cache_dir / "epochs_cache" if args.cache_dir else None
    data = get_cross_subject_data(cfg.data, n_jobs=args.n_jobs, cache_dir=cache_dir)
    for name in args.members:
        dump_cross_subject_member(name, data, cfg)
