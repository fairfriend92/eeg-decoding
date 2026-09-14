"""Hyperparameter sweep for EEGNet and EEG Conformer.

Selects by mean validation accuracy across subjects; never touches the
test session (session_E). Each subject's data is loaded once and reused
across every hyperparameter combination, since loading does not depend on
the hyperparameters being swept.

Validation strategy is chosen via n_folds:
    n_folds=1  -- single 80/20 split per subject, cheapest, noisiest
               per-config estimate.
    n_folds>=2 -- stratified k-fold cross-validation per subject, more
               expensive, less noisy per-config estimate. Default is 5.

Must be run with this file's own directory (src/) as the script's
directory -- see train.py for why.

Default search grids, applied on top of Config() defaults:

    eegnet:    lr in {1e-3, 3e-4}, dropout in {0.25, 0.5}
    conformer: lr in {1e-3, 3e-4}, dropout in {0.3, 0.5}

Each grid is 4 configs x 9 subjects x n_folds short, early-stopped
training runs (e.g. 180 runs at the n_folds=5 default). Edit
DEFAULT_GRIDS below to widen or narrow the grid.
"""

import argparse
import itertools
import json

from sklearn.model_selection import StratifiedKFold

from config import OUTPUT_DIR, SUBJECT_IDS, Config
from data_loader import get_subject_data
from train import fit, fit_on_split

DEFAULT_GRIDS = {
    "eegnet": {
        "lr":      [1e-3, 3e-4],
        "dropout": [0.25, 0.5],
    },
    "conformer": {
        "lr":      [1e-3, 3e-4],
        "dropout": [0.3, 0.5],
    },
}


def expand_grid(grid):
    """Expands a dict of parameter lists into a list of individual configs.

    Parameters
    ----------
    grid : dict
        Field name -> list of candidate values.

    Returns
    -------
    list of dict
        One dict per combination, e.g. {"lr": 1e-3, "dropout": 0.25}.
    """
    keys, value_lists = zip(*grid.items())
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def build_config(model_name, overrides):
    """Builds a Config with overrides applied to train or model sub-config.

    Parameters
    ----------
    model_name : str
        Either "eegnet" or "conformer".
    overrides : dict
        Field name -> value. Fields matching TrainConfig attributes are
        applied there; all others are applied to the model-specific config.

    Returns
    -------
    Config

    Raises
    ------
    ValueError
        If a key in `overrides` matches no field on either sub-config.
    """
    cfg       = Config()
    model_cfg = getattr(cfg, model_name)

    for key, value in overrides.items():
        if hasattr(cfg.train, key):
            setattr(cfg.train, key, value)
        elif hasattr(model_cfg, key):
            setattr(model_cfg, key, value)
        else:
            raise ValueError(f"Unknown hyperparameter '{key}' for {model_name}")

    return cfg


def evaluate_subject(X, y, model_name, cfg: Config, n_folds, subject_id=None):
    """Evaluates one subject under one config: a single split (n_folds=1)
    or stratified k-fold cross-validation (n_folds>=2).

    Never touches test data -- X, y should be a subject's session_T only.

    Parameters
    ----------
    X, y : ndarray
        A subject's full training-session trials and labels.
    model_name : str
        Either "eegnet" or "conformer".
    cfg : Config
        Full project configuration.
    n_folds : int
        1 for a single 80/20 split; >=2 for that many stratified folds.
    subject_id : int, optional
        Used to label progress messages and to derive a base seed, so
        results are reproducible but distinct across subjects.

    Returns
    -------
    list of float
        Validation accuracy for each split (length 1 if n_folds=1, else
        length n_folds).
    """
    if n_folds == 1:
        _, val_acc = fit(X, y, model_name, cfg, subject_id=subject_id)
        return [val_acc]

    base_seed = cfg.train.seed if subject_id is None else cfg.train.seed + subject_id
    splitter  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=base_seed)

    fold_accs = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        # Distinct seed per fold, so folds don't all train from identical
        # initialization; still reproducible given (subject, fold).
        _, val_acc = fit_on_split(
            X[train_idx], y[train_idx], X[val_idx], y[val_idx], model_name, cfg,
            subject_id=subject_id, seed=base_seed + fold_idx,
        )
        fold_accs.append(val_acc)

    return fold_accs


def run_sweep(model_name, grid=None, subjects=None, n_folds=5):
    """Evaluates each hyperparameter combination across subjects.

    Parameters
    ----------
    model_name : str
        Either "eegnet" or "conformer".
    grid : dict, optional
        Field name -> list of candidate values; defaults to
        `DEFAULT_GRIDS[model_name]`.
    subjects : list of int, optional
        Subjects to evaluate each config on; defaults to SUBJECT_IDS.
    n_folds : int
        1 for a single 80/20 split per subject; >=2 for that many
        stratified cross-validation folds per subject.

    Returns
    -------
    list of dict
        One entry per config, sorted by descending mean validation
        accuracy: the overrides, per-subject split-level accuracies, and
        the overall mean across all subjects and splits.
    """
    grid     = grid or DEFAULT_GRIDS[model_name]
    subjects = subjects or SUBJECT_IDS
    configs  = expand_grid(grid)

    print(f"[sweep:{model_name}] {len(configs)} configs x {len(subjects)} subjects x {n_folds} split(s)")

    # Loaded once per subject; reused across every config below, since
    # data loading does not depend on the hyperparameters being swept.
    data_cache = {sid: get_subject_data(sid, Config().data)[:2] for sid in subjects}

    sweep_results = []
    for overrides in configs:
        cfg = build_config(model_name, overrides)

        subject_results = []
        all_split_accs  = []
        for sid in subjects:
            X_train_full, y_train_full = data_cache[sid]
            split_accs = evaluate_subject(X_train_full, y_train_full, model_name, cfg, n_folds, subject_id=sid)
            mean_acc   = sum(split_accs) / len(split_accs)
            subject_results.append({
                "subject":          sid,
                "split_accuracies": split_accs,
                "mean_accuracy":    mean_acc,
            })
            all_split_accs.extend(split_accs)

        overall_mean = sum(all_split_accs) / len(all_split_accs)
        print(f"[sweep:{model_name}] {overrides} -> mean val_acc={overall_mean:.3f} (across all subjects/splits)")
        sweep_results.append({
            **overrides,
            "mean_val_accuracy": overall_mean,
            "subjects":          subject_results,
        })

    sweep_results.sort(key=lambda r: r["mean_val_accuracy"], reverse=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{model_name}_sweep.json"
    with open(path, "w") as f:
        json.dump(sweep_results, f, indent=2)

    best = sweep_results[0]
    print(f"[sweep:{model_name}] best: { {k: v for k, v in best.items() if k != 'subjects'} }")

    return sweep_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=list(DEFAULT_GRIDS.keys()))
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="1 for a single 80/20 split per subject; >=2 for that many stratified CV folds (default: 5).",
    )
    args = parser.parse_args()

    run_sweep(args.model, n_folds=args.n_folds)
