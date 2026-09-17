"""Scores train.py's saved checkpoints on the held-out test set (session_E).

Relies on config.py's model hyperparameters matching what was used at
training time -- do not change config.py between running train.py and
this script.

n_folds=1 checkpoints: reports test accuracy/kappa directly.
n_folds>=2 checkpoints: reports fold-mean test accuracy/kappa (mean +/-
std across the fold models' individual scores) and ensemble test
accuracy/kappa.

Must be run with this file's own directory (src/) as the script's
directory -- see train.py for why.
"""

import argparse
import json
import statistics

import torch

from config import N_CHANNELS, N_CLASSES, OUTPUT_DIR, Config, CroppedConfig
from data_loader import get_subject_data
from models.conformer import EEGConformer
from models.eegnet import EEGNet
from train import evaluate_arrays, evaluate_cropped, evaluate_ensemble, evaluate_ensemble_cropped

MODEL_REGISTRY = {
    "eegnet": EEGNet,
    "conformer": EEGConformer,
}


def load_checkpoint(checkpoint_path, model_name, n_times, cfg: Config, device):
    """Reconstructs a model from current config.py and loads saved weights.

    Parameters
    ----------
    checkpoint_path : str
        Path to a .pt file written by train.py.
    model_name : str
        Either "eegnet" or "conformer".
    n_times : int
        Number of time samples per epoch, needed to reconstruct the
        model's architecture.
    cfg : Config
        Full project configuration -- must match what was used to train
        this checkpoint.
    device : torch.device
        Device to load the model onto.

    Returns
    -------
    nn.Module
        The loaded model, in eval mode.
    """
    model_cls = MODEL_REGISTRY[model_name]
    model_cfg = getattr(cfg, model_name)
    model     = model_cls(N_CHANNELS, n_times, N_CLASSES, model_cfg).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def evaluate_subject(entry, model_name, cfg: Config):
    """Scores one subject's checkpoint(s) from a train.py manifest entry.

    Parameters
    ----------
    entry : dict
        One entry from output/{model}_checkpoints.json.
    model_name : str
        Either "eegnet" or "conformer".
    cfg : Config
        Full project configuration -- must match what was used to train
        the checkpoint(s).

    Returns
    -------
    dict
        n_folds=1: subject id, val accuracy, test accuracy, test kappa,
        final-model test accuracy/kappa.
        n_folds>=2: subject id, fold-mean test accuracy/kappa (mean +/-
        std), ensemble test accuracy/kappa, final-model test
        accuracy/kappa, and per-fold detail.
    """
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    subject_id = entry["subject"]
    _, _, X_test, y_test = get_subject_data(subject_id, cfg.data)

    cropped_entry = entry.get("cropped")
    if cropped_entry is not None:
        cropped_cfg = CroppedConfig(
            crop_size=cropped_entry["crop_size"],
            crop_stride=cropped_entry["crop_stride"],
            eval_crop_stride=cfg.cropped.eval_crop_stride,
        )
        model_n_times = cropped_cfg.crop_size
    else:
        cropped_cfg = None
        model_n_times = X_test.shape[-1]

    def score(model):
        if cropped_cfg is not None:
            return evaluate_cropped(model, X_test, y_test, cropped_cfg, device, cfg.train.batch_size)
        return evaluate_arrays(model, X_test, y_test, device, cfg.train.batch_size)

    def score_ensemble(models):
        if cropped_cfg is not None:
            return evaluate_ensemble_cropped(models, X_test, y_test, cropped_cfg, device, cfg.train.batch_size)
        return evaluate_ensemble(models, X_test, y_test, device, cfg.train.batch_size)

    if entry["n_folds"] == 1:
        model = load_checkpoint(entry["checkpoint"], model_name, model_n_times, cfg, device)
        test_acc, test_kappa = score(model)
        print(f"[{model_name}] subject {subject_id}: test_acc={test_acc:.3f}, test_kappa={test_kappa:.3f}")

        final_model = load_checkpoint(entry["final_checkpoint"], model_name, model_n_times, cfg, device)
        final_acc, final_kappa = score(final_model)
        print(f"[{model_name}] subject {subject_id}: final-fit test_acc={final_acc:.3f}, test_kappa={final_kappa:.3f}")

        return {
            "subject":                 subject_id,
            "model":                   model_name,
            "n_folds":                 1,
            "val_accuracy":            entry["val_accuracy"],
            "test_accuracy":           test_acc,
            "test_kappa":              test_kappa,
            "final_model_test_accuracy": final_acc,
            "final_model_test_kappa":    final_kappa,
        }

    fold_results = []
    fold_models  = []
    for fold_entry in entry["fold_results"]:
        model = load_checkpoint(fold_entry["checkpoint"], model_name, model_n_times, cfg, device)
        test_acc, test_kappa = score(model)
        print(
            f"[{model_name}] subject {subject_id} fold {fold_entry['fold']}: "
            f"test_acc={test_acc:.3f}, test_kappa={test_kappa:.3f}"
        )
        fold_results.append({
            "fold":          fold_entry["fold"],
            "val_accuracy":  fold_entry["val_accuracy"],
            "test_accuracy": test_acc,
            "test_kappa":    test_kappa,
        })
        fold_models.append(model)

    test_accs   = [r["test_accuracy"] for r in fold_results]
    test_kappas = [r["test_kappa"] for r in fold_results]
    ensemble_test_acc, ensemble_test_kappa = score_ensemble(fold_models)

    final_model = load_checkpoint(entry["final_checkpoint"], model_name, model_n_times, cfg, device)
    final_acc, final_kappa = score(final_model)

    print(
        f"[{model_name}] subject {subject_id}: fold-mean test_acc="
        f"{statistics.mean(test_accs):.3f}+/-{statistics.stdev(test_accs):.3f}, "
        f"ensemble test_acc={ensemble_test_acc:.3f}, final-fit test_acc={final_acc:.3f}"
    )

    return {
        "subject":                   subject_id,
        "model":                     model_name,
        "n_folds":                   entry["n_folds"],
        "test_accuracy":             statistics.mean(test_accs),
        "test_accuracy_std":         statistics.stdev(test_accs),
        "test_kappa":                statistics.mean(test_kappas),
        "test_kappa_std":            statistics.stdev(test_kappas),
        "ensemble_test_accuracy":    ensemble_test_acc,
        "ensemble_test_kappa":       ensemble_test_kappa,
        "final_model_test_accuracy": final_acc,
        "final_model_test_kappa":    final_kappa,
        "fold_results":              fold_results,
    }


def evaluate_all(model_name, cfg: Config = None):
    """Loads a checkpoint manifest and scores every subject's checkpoint(s).

    Parameters
    ----------
    model_name : str
        Either "eegnet" or "conformer".
    cfg : Config, optional
        Full project configuration; defaults to `Config()`. Must match
        what was used to train the checkpoints being loaded.

    Returns
    -------
    list of dict
        Per-subject evaluation results, in subject order.
    """
    cfg = cfg or Config()

    manifest_path = OUTPUT_DIR / f"{model_name}_checkpoints.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    results = [evaluate_subject(entry, model_name, cfg) for entry in manifest]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / f"{model_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def summarize(results):
    """Aggregates per-subject results into summary statistics across subjects.

    Standard deviation uses ddof=1 (sample std). Also aggregates the
    full-data final-fit model's metrics (present for both n_folds=1 and
    n_folds>=2 runs); for n_folds>=2, also aggregates the ensemble metrics.

    Parameters
    ----------
    results : list of dict
        Per-subject results, as returned by `evaluate_all`.

    Returns
    -------
    dict
        accuracy/kappa mean and std (fold-mean for n_folds>=2, single
        score for n_folds=1), final-model accuracy/kappa mean and std,
        subject count, n_folds; plus ensemble accuracy/kappa mean and
        std when n_folds>=2.
    """
    n_folds     = results[0]["n_folds"]
    accuracies  = [r["test_accuracy"] for r in results]
    kappas      = [r["test_kappa"] for r in results]
    final_accs  = [r["final_model_test_accuracy"] for r in results]
    final_kaps  = [r["final_model_test_kappa"] for r in results]

    summary = {
        "n_subjects":          len(results),
        "n_folds":             n_folds,
        "accuracy_mean":       statistics.mean(accuracies),
        "accuracy_std":        statistics.stdev(accuracies),
        "kappa_mean":          statistics.mean(kappas),
        "kappa_std":           statistics.stdev(kappas),
        "final_accuracy_mean": statistics.mean(final_accs),
        "final_accuracy_std":  statistics.stdev(final_accs),
        "final_kappa_mean":    statistics.mean(final_kaps),
        "final_kappa_std":     statistics.stdev(final_kaps),
    }

    if n_folds >= 2:
        ensemble_accs   = [r["ensemble_test_accuracy"] for r in results]
        ensemble_kappas = [r["ensemble_test_kappa"] for r in results]
        summary["ensemble_accuracy_mean"] = statistics.mean(ensemble_accs)
        summary["ensemble_accuracy_std"]  = statistics.stdev(ensemble_accs)
        summary["ensemble_kappa_mean"]    = statistics.mean(ensemble_kappas)
        summary["ensemble_kappa_std"]     = statistics.stdev(ensemble_kappas)

    return summary


def compare_models(model_names, cfg: Config = None):
    """Evaluates, summarizes, and reports results for one or more models.

    Parameters
    ----------
    model_names : list of str
        Models to evaluate, e.g. ["eegnet", "conformer"].
    cfg : Config, optional
        Full project configuration; defaults to `Config()`.

    Returns
    -------
    dict
        Model name mapped to its summary dict.
    """
    summaries = {}
    for name in model_names:
        results = evaluate_all(name, cfg)
        summary = summarize(results)
        summaries[name] = summary

        if summary["n_folds"] == 1:
            print(
                f"{name}: accuracy = {summary['accuracy_mean']:.3f} +/- {summary['accuracy_std']:.3f}, "
                f"kappa = {summary['kappa_mean']:.3f} +/- {summary['kappa_std']:.3f} "
                f"(n={summary['n_subjects']})"
            )
        else:
            print(
                f"{name}: fold-mean accuracy = {summary['accuracy_mean']:.3f} +/- {summary['accuracy_std']:.3f}, "
                f"kappa = {summary['kappa_mean']:.3f} +/- {summary['kappa_std']:.3f} "
                f"(n_folds={summary['n_folds']}, n={summary['n_subjects']})"
            )
            print(
                f"{name}: ensemble accuracy = {summary['ensemble_accuracy_mean']:.3f} +/- "
                f"{summary['ensemble_accuracy_std']:.3f}, kappa = {summary['ensemble_kappa_mean']:.3f} +/- "
                f"{summary['ensemble_kappa_std']:.3f}"
            )
        print(
            f"{name}: final-fit (100% of session_T) accuracy = {summary['final_accuracy_mean']:.3f} +/- "
            f"{summary['final_accuracy_std']:.3f}, kappa = {summary['final_kappa_mean']:.3f} +/- "
            f"{summary['final_kappa_std']:.3f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()

    compare_models(args.models)
