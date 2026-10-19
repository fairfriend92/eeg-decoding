"""Fits the patch transformer on every public participant and writes its Codabench weights.

An early-stopped fit on all training participants (fitting plus validation) sets
the epoch count, then a new fit on the same windows trains for the step-matched
count. The output is the
weights.pt that scripts/submission/build_submission.py packages with
patch_transformer_submission.py.

By default the preset's test participants stay out of the fit and are scored,
so the result estimates generalization (the Codabench warm-up scores those
participants). --all-participants fits on every participant instead, for the
sealed phase.

Must be run with this file's own directory (src/) as the script's
directory. See train.py for why.
"""

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from config import Config, add_dataset_cli_args, resolve_config
from data_loader import get_cross_subject_data
from train import compute_matched_epochs, fit, fit_full, predict_probs


def export_patch_transformer(data, cfg: Config, path, n_epochs=None):
    """Fits the patch transformer on fitting plus validation participants and saves it.

    Parameters
    ----------
    data : dict
        As returned by data_loader.get_cross_subject_data. Its test arrays, when
        present, only score the fitted model.
    cfg : Config
        Full project configuration. cfg.data.alignment must be "euclidean", the
        fit alignment the streamed causal alignment is scored against.
    path : Path
        Destination of the checkpoint, written with torch.save.
    n_epochs : int, optional
        Epochs of the final fit. When None, the count comes from an early-stopped
        fit on the same windows, step-matched to the full set.

    Returns
    -------
    dict
        The saved checkpoint: "net" (state dict), "arch" (patch_size and
        patch_stride in samples) and "align" (z_prior_windows and
        cov_prior_windows of the submission's CausalAlign). When data has test
        participants, the model is also scored on them with the balanced
        accuracy pooled over windows and averaged over (participant, session)
        cells, printed.

    Raises
    ------
    ValueError
        If cfg.data.alignment is not "euclidean".
    """
    if cfg.data.alignment != "euclidean":
        raise ValueError(f"The submission aligns causally against a full-trial Euclidean fit, not '{cfg.data.alignment}'.")
    X_all = np.concatenate([data["X_fit"], data["X_val"]])
    y_all = np.concatenate([data["y_fit"], data["y_val"]])

    if n_epochs is None:
        # fit() holds out a fifth of the windows for early stopping, so the epoch
        # budget is matched on the windows it actually trained on.
        _, _, best_epoch = fit(X_all, y_all, "patch_transformer", cfg)
        n_train = len(X_all) - round(len(X_all) * 0.2)
        n_epochs = compute_matched_epochs([(best_epoch, n_train)], len(X_all), cfg.train.batch_size)
    model = fit_full(X_all, y_all, "patch_transformer", cfg, n_epochs)

    patch = cfg.patch_transformer
    checkpoint = {
        "net": {k: v.cpu() for k, v in model.state_dict().items()},
        "arch": {"patch_size": patch.patch_size, "patch_stride": patch.patch_stride or patch.patch_size},
        "align": {"z_prior_windows": 0.0, "cov_prior_windows": cfg.data.causal_cov_prior_windows},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"[patch_transformer] {n_epochs} epochs on {len(X_all)} trials, wrote {path}", flush=True)

    if "X_test" in data:
        device = next(model.parameters()).device
        pred = predict_probs(model, data["X_test"], device, cfg.train.batch_size).argmax(axis=1)
        cells = sorted(set(zip(data["subject_test"], data["session_test"])))
        cell_scores = [
            balanced_accuracy_score(data["y_test"][mask], pred[mask])
            for mask in ((data["subject_test"] == s) & (data["session_test"] == e) for s, e in cells)
        ]
        print(f"[patch_transformer] test participants {sorted(set(int(s) for s in data['subject_test']))}: "
              f"pooled balanced acc={balanced_accuracy_score(data['y_test'], pred):.3f}, "
              f"cell-averaged={np.mean(cell_scores):.3f} over {len(cell_scores)} cells", flush=True)
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Destination of the weights file.")
    parser.add_argument("--all-participants", action="store_true", help="Fit on every participant, leaving no test participants. For the sealed phase.")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs of the final fit. Skips the early-stopped fit that otherwise sets the count.")
    parser.add_argument("--n-jobs", type=int, default=6, help="Workers epoching participants missing from the cache.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Persistent directory for the epoch cache.")
    add_dataset_cli_args(parser)
    args = parser.parse_args()

    cfg = resolve_config(args)
    if args.all_participants:
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, test_subject_ids=[]))
    cache_dir = args.cache_dir / "epochs_cache" if args.cache_dir else None
    export_patch_transformer(get_cross_subject_data(cfg.data, n_jobs=args.n_jobs, cache_dir=cache_dir), cfg, args.out, args.epochs)
