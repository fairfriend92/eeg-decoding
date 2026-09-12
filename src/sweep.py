"""Hyperparameter sweep for EEGNet and EEG Conformer.

Selects by mean validation accuracy across subjects; never touches the
test session (session_E) -- see train.fit for why. Each subject's data is
loaded once and reused across every hyperparameter combination, since
loading does not depend on the hyperparameters being swept.

Must be run with this file's own directory (src/) as the script's
directory -- see train.py for why.

Default search grids, applied on top of Config() defaults:

    eegnet:    lr in {1e-3, 3e-4}, dropout in {0.25, 0.5}
    conformer: lr in {1e-3, 3e-4}, dropout in {0.3, 0.5}

Each grid is 4 configs x 9 subjects = 36 short, early-stopped training
runs. Edit DEFAULT_GRIDS below to widen or narrow the search.
"""

import argparse
import itertools
import json

from config import OUTPUT_DIR, SUBJECT_IDS, Config
from data_loader import get_subject_data
from train import fit

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


def run_sweep(model_name, grid=None, subjects=None):
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

    Returns
    -------
    list of dict
        One entry per config, sorted by descending mean validation
        accuracy: the overrides, per-subject validation accuracies, and
        their mean.
    """
    grid     = grid or DEFAULT_GRIDS[model_name]
    subjects = subjects or SUBJECT_IDS
    configs  = expand_grid(grid)

    print(f"[sweep:{model_name}] {len(configs)} configs x {len(subjects)} subjects")

    # Loaded once per subject; reused across every config below, since
    # data loading does not depend on the hyperparameters being swept.
    data_cache = {sid: get_subject_data(sid, Config().data)[:2] for sid in subjects}

    sweep_results = []
    for overrides in configs:
        cfg      = build_config(model_name, overrides)
        val_accs = []
        for sid in subjects:
            X_train_full, y_train_full = data_cache[sid]
            _, val_acc = fit(X_train_full, y_train_full, model_name, cfg, subject_id=sid)
            val_accs.append(val_acc)

        mean_acc = sum(val_accs) / len(val_accs)
        print(f"[sweep:{model_name}] {overrides} -> mean val_acc={mean_acc:.3f}")
        sweep_results.append({
            **overrides,
            "mean_val_accuracy": mean_acc,
            "val_accuracies":    val_accs,
        })

    sweep_results.sort(key=lambda r: r["mean_val_accuracy"], reverse=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{model_name}_sweep.json"
    with open(path, "w") as f:
        json.dump(sweep_results, f, indent=2)

    best = sweep_results[0]
    print(f"[sweep:{model_name}] best: { {k: v for k, v in best.items() if k != 'val_accuracies'} }")

    return sweep_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=list(DEFAULT_GRIDS.keys()))
    args = parser.parse_args()

    run_sweep(args.model)
