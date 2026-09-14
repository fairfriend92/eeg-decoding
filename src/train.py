"""Per-subject training loop for EEGNet and EEG Conformer models.

n_folds=1 (default): single 80/20 split per subject, one model.
n_folds>=2: stratified k-fold cross-validation per subject, one model per
fold, plus a softmax-averaged ensemble score -- more expensive, gives
fold-level test accuracy/kappa (mean +/- std) instead of a single draw.

Must be run with this file's own directory (src/) as the script's
directory, e.g. `python train.py` from within src/, or `python src/train.py`
from the repository root. Do not invoke via `python -m src.train`, which
changes the Python path and breaks the flat `from config import ...` style
imports used throughout src/.
"""

import argparse
import json
import statistics

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config import N_CHANNELS, N_CLASSES, OUTPUT_DIR, SUBJECT_IDS, Config
from data_loader import get_subject_data
from models.conformer import EEGConformer
from models.eegnet import EEGNet

MODEL_REGISTRY = {
    "eegnet": EEGNet,
    "conformer": EEGConformer,
}


def make_loaders(X_train, y_train, X_val, y_val, batch_size, generator=None):
    """Wraps train/val arrays into DataLoaders.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : ndarray
        Split arrays for a single subject.
    batch_size : int
        Batch size for both loaders.
    generator : torch.Generator, optional
        Governs the training loader's shuffle order. Passed explicitly
        (rather than relying on global torch RNG state) so shuffle order
        is reproducible independent of what else has consumed random
        numbers beforehand.

    Returns
    -------
    DataLoader, DataLoader
        Training and validation loaders.
    """
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate_loader(model, loader, device):
    """Computes accuracy and Cohen's kappa over a DataLoader.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate; switched to eval mode internally.
    loader : DataLoader
        Data to evaluate on.
    device : torch.device
        Device to run inference on.

    Returns
    -------
    float, float
        Accuracy and Cohen's kappa.
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds.extend(model(xb).argmax(dim=1).cpu().numpy())
            targets.extend(yb.numpy())
    return accuracy_score(targets, preds), cohen_kappa_score(targets, preds)


def evaluate_arrays(model, X, y, device, batch_size):
    """Computes accuracy and Cohen's kappa directly from arrays.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    X, y : ndarray
        Evaluation data and labels.
    device : torch.device
        Device to run inference on.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    float, float
        Accuracy and Cohen's kappa.
    """
    ds     = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return evaluate_loader(model, loader, device)


def predict_probs(model, X, device, batch_size):
    """Computes softmax class probabilities for X via batched inference.

    Parameters
    ----------
    model : nn.Module
        Trained model; switched to eval mode internally.
    X : ndarray
        Data to predict on.
    device : torch.device
        Device to run inference on.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    ndarray, shape (len(X), n_classes)
        Class probabilities.
    """
    ds     = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            all_probs.append(torch.softmax(model(xb), dim=1).cpu())
    return torch.cat(all_probs, dim=0).numpy()


def evaluate_ensemble(models, X, y, device, batch_size):
    """Scores the average of several models' softmax predictions.

    Parameters
    ----------
    models : list of nn.Module
        Trained models (e.g. one per cross-validation fold) to ensemble.
    X, y : ndarray
        Evaluation data and labels.
    device : torch.device
        Device to run inference on.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    float, float
        Accuracy and Cohen's kappa of the ensembled predictions.
    """
    avg_probs = sum(predict_probs(model, X, device, batch_size) for model in models) / len(models)
    preds     = avg_probs.argmax(axis=1)
    return accuracy_score(y, preds), cohen_kappa_score(y, preds)



def fit_on_split(X_train, y_train, X_val, y_val, model_name, cfg: Config, subject_id=None, seed=None):
    """Trains a single model on an explicit train/val split, with early
    stopping.

    Core training loop, reused by both `fit` (single 80/20 split) and any
    cross-validation caller that supplies its own fold split.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : ndarray
        Already-split training and validation data.
    model_name : str
        Either "eegnet" or "conformer".
    cfg : Config
        Full project configuration.
    subject_id : int, optional
        Used only to label progress messages.
    seed : int, optional
        Seeds model init and the training loader's shuffle order. Defaults
        to `cfg.train.seed` (or `cfg.train.seed + subject_id` if given),
        matching `fit`'s behavior; a cross-validation caller should pass a
        distinct seed per fold.

    Returns
    -------
    model : nn.Module
        Trained model, loaded with its best-validation-accuracy weights.
    best_val_acc : float
        Best validation accuracy reached during training.
    """
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    label  = f"subject {subject_id}" if subject_id is not None else "subject"

    if seed is None:
        seed = cfg.train.seed if subject_id is None else cfg.train.seed + subject_id
    torch.manual_seed(seed)
    loader_generator = torch.Generator().manual_seed(seed)

    train_loader, val_loader = make_loaders(
        X_train, y_train, X_val, y_val, cfg.train.batch_size, generator=loader_generator
    )

    n_times   = X_train.shape[-1]
    model_cls = MODEL_REGISTRY[model_name]
    model_cfg = getattr(cfg, model_name)
    model     = model_cls(N_CHANNELS, n_times, N_CLASSES, model_cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.lr, weight_decay=cfg.train.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(
        f"[{model_name}] {label}: training on {len(X_train)} trials, "
        f"validating on {len(X_val)}, device={device}"
    )

    best_val_acc       = -1.0
    best_state         = None
    epochs_no_improve  = 0

    for epoch in range(cfg.train.n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_acc, _ = evaluate_loader(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            best_state        = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epochs_no_improve >= cfg.train.patience:
            print(
                f"[{model_name}] {label}: epoch {epoch:>3} - "
                f"val_acc={val_acc:.3f} (best={best_val_acc:.3f}, "
                f"no_improve={epochs_no_improve}/{cfg.train.patience})"
            )

        if epochs_no_improve >= cfg.train.patience:
            print(f"[{model_name}] {label}: early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_val_acc


def fit(X_train_full, y_train_full, model_name, cfg: Config, subject_id=None):
    """Trains a single model on already-loaded data, with early stopping.

    Splits the given training data into a single 80/20 train/validation
    split for early stopping only; takes no test data, so callers
    (including a hyperparameter sweep) cannot leak test-set information
    into model selection even indirectly.

    Parameters
    ----------
    X_train_full, y_train_full : ndarray
        A subject's full session_T trials and labels, as returned by
        `get_subject_data`.
    model_name : str
        Either "eegnet" or "conformer".
    cfg : Config
        Full project configuration.
    subject_id : int, optional
        Used only to label progress messages.

    Returns
    -------
    model : nn.Module
        Trained model, loaded with its best-validation-accuracy weights.
    best_val_acc : float
        Best validation accuracy reached during training.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, stratify=y_train_full, random_state=cfg.train.seed,
    )
    return fit_on_split(X_train, y_train, X_val, y_val, model_name, cfg, subject_id=subject_id)


def train_one_subject(subject_id, model_name, cfg: Config, n_folds=1):
    """Loads a subject's data, fits model(s), saves them, evaluates on test.

    n_folds=1 (default): single 80/20 split, one model, one checkpoint --
    unchanged from the original behavior.

    n_folds>=2: stratified k-fold cross-validation over session_T, one
    model per fold, each saved separately. Reports fold-level test
    accuracy/kappa as mean +/- std (matching the project's fold-level
    reporting standard) plus a softmax-averaged ensemble score across the
    fold models.

    Parameters
    ----------
    subject_id : int
        Subject identifier (1-9).
    model_name : str
        Either "eegnet" or "conformer".
    cfg : Config
        Full project configuration.
    n_folds : int
        1 for a single split; >=2 for that many cross-validation folds.

    Returns
    -------
    dict
        n_folds=1: subject id, val accuracy, test accuracy, test kappa,
        checkpoint path.
        n_folds>=2: subject id, mean/std test accuracy and kappa across
        folds, ensemble test accuracy and kappa, and per-fold detail.
    """
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    print(f"[{model_name}] subject {subject_id}: loading and epoching EEG data...")
    X_train_full, y_train_full, X_test, y_test = get_subject_data(subject_id, cfg.data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if n_folds == 1:
        model, best_val_acc = fit(X_train_full, y_train_full, model_name, cfg, subject_id=subject_id)

        ckpt_path = OUTPUT_DIR / f"{model_name}_subject{subject_id}.pt"
        torch.save(model.state_dict(), ckpt_path)

        test_acc, test_kappa = evaluate_arrays(model, X_test, y_test, device, cfg.train.batch_size)
        print(
            f"[{model_name}] subject {subject_id}: done - "
            f"val_acc={best_val_acc:.3f}, test_acc={test_acc:.3f}, test_kappa={test_kappa:.3f}"
        )

        return {
            "subject":       subject_id,
            "model":         model_name,
            "n_folds":       1,
            "val_accuracy":  best_val_acc,
            "test_accuracy": test_acc,
            "test_kappa":    test_kappa,
            "checkpoint":    str(ckpt_path),
        }

    base_seed = cfg.train.seed + subject_id
    splitter  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=base_seed)

    fold_results = []
    fold_models  = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_full, y_train_full)):
        model, val_acc = fit_on_split(
            X_train_full[train_idx], y_train_full[train_idx],
            X_train_full[val_idx], y_train_full[val_idx],
            model_name, cfg, subject_id=subject_id, seed=base_seed + fold_idx,
        )

        ckpt_path = OUTPUT_DIR / f"{model_name}_subject{subject_id}_fold{fold_idx}.pt"
        torch.save(model.state_dict(), ckpt_path)

        test_acc, test_kappa = evaluate_arrays(model, X_test, y_test, device, cfg.train.batch_size)
        print(
            f"[{model_name}] subject {subject_id} fold {fold_idx}: "
            f"val_acc={val_acc:.3f}, test_acc={test_acc:.3f}, test_kappa={test_kappa:.3f}"
        )

        fold_results.append({
            "fold":          fold_idx,
            "val_accuracy":  val_acc,
            "test_accuracy": test_acc,
            "test_kappa":    test_kappa,
            "checkpoint":    str(ckpt_path),
        })
        fold_models.append(model)

    test_accs   = [r["test_accuracy"] for r in fold_results]
    test_kappas = [r["test_kappa"] for r in fold_results]
    ensemble_test_acc, ensemble_test_kappa = evaluate_ensemble(
        fold_models, X_test, y_test, device, cfg.train.batch_size
    )

    print(
        f"[{model_name}] subject {subject_id}: done - "
        f"test_acc={statistics.mean(test_accs):.3f}+/-{statistics.stdev(test_accs):.3f}, "
        f"test_kappa={statistics.mean(test_kappas):.3f}+/-{statistics.stdev(test_kappas):.3f}, "
        f"ensemble_test_acc={ensemble_test_acc:.3f}, ensemble_test_kappa={ensemble_test_kappa:.3f}"
    )

    return {
        "subject":                subject_id,
        "model":                  model_name,
        "n_folds":                n_folds,
        "test_accuracy":          statistics.mean(test_accs),
        "test_accuracy_std":      statistics.stdev(test_accs),
        "test_kappa":             statistics.mean(test_kappas),
        "test_kappa_std":         statistics.stdev(test_kappas),
        "ensemble_test_accuracy": ensemble_test_acc,
        "ensemble_test_kappa":    ensemble_test_kappa,
        "fold_results":           fold_results,
    }


def run_all_subjects(model_name, cfg: Config = None, n_folds=1):
    """Trains one model (or one per fold) per subject; writes results to disk.

    Parameters
    ----------
    model_name : str
        Either "eegnet" or "conformer".
    cfg : Config, optional
        Full project configuration; defaults to `Config()`.
    n_folds : int
        1 for a single 80/20 split per subject (default, matches prior
        behavior); >=2 for that many cross-validation folds per subject.

    Returns
    -------
    list of dict
        Per-subject results, in subject order.
    """
    cfg = cfg or Config()
    print(
        f"Training {model_name} for {len(SUBJECT_IDS)} subjects "
        f"(n_folds={n_folds}, within-subject, official session_T/session_E split)"
    )
    results = [train_one_subject(sid, model_name, cfg, n_folds=n_folds) for sid in SUBJECT_IDS]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / f"{model_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument(
        "--n-folds", type=int, default=1,
        help="1 for a single 80/20 split per subject (default); >=2 for that many "
             "stratified CV folds per subject, with an added softmax-averaged ensemble score.",
    )
    args = parser.parse_args()

    run_all_subjects(args.model, n_folds=args.n_folds)
