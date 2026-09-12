"""Per-subject training loop for EEGNet and EEG Conformer models.

Must be run with this file's own directory (src/) as the script's
directory, e.g. `python train.py` from within src/, or `python src/train.py`
from the repository root. Do not invoke via `python -m src.train`, which
changes the Python path and breaks the flat `from config import ...` style
imports used throughout src/.
"""

import argparse
import json

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config import N_CHANNELS, N_CLASSES, OUTPUT_DIR, SUBJECT_IDS, Config
from data_loader import get_subject_data
from models.conformer import EEGConformer
from models.eegnet import EEGNet

MODEL_REGISTRY = {
    "eegnet": EEGNet,
    "conformer": EEGConformer,
}


def make_loaders(X_train, y_train, X_val, y_val, batch_size):
    """Wraps train/val arrays into DataLoaders.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : ndarray
        Split arrays for a single subject.
    batch_size : int
        Batch size for both loaders.

    Returns
    -------
    DataLoader, DataLoader
        Training and validation loaders.
    """
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
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


def train_one_subject(subject_id, model_name, cfg: Config):
    """Trains a single model for a single subject, with early stopping.

    The training session (session_T) is further split into train/validation
    for early stopping only. The test session (session_E) is held out
    entirely until final evaluation, never used for model selection.

    Parameters
    ----------
    subject_id : int
        Subject identifier (1-9).
    model_name : str
        Either "eegnet" or "conformer".
    cfg : Config
        Full project configuration.

    Returns
    -------
    dict
        Subject id, best validation accuracy, test accuracy, test kappa,
        and the checkpoint path.
    """
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    print(f"[{model_name}] subject {subject_id}: loading and epoching EEG data...")
    X_train_full, y_train_full, X_test, y_test = get_subject_data(subject_id, cfg.data)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, stratify=y_train_full, random_state=cfg.train.seed,
    )
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, cfg.train.batch_size)

    n_times   = X_train.shape[-1]
    model_cls = MODEL_REGISTRY[model_name]
    model_cfg = getattr(cfg, model_name)
    model     = model_cls(N_CHANNELS, n_times, N_CLASSES, model_cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(
        f"[{model_name}] subject {subject_id}: training on {len(X_train)} trials, "
        f"validating on {len(X_val)} (session_T split), device={device}"
    )

    best_val_acc       = 0.0
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
                f"[{model_name}] subject {subject_id}: epoch {epoch:>3} - "
                f"val_acc={val_acc:.3f} (best={best_val_acc:.3f}, "
                f"no_improve={epochs_no_improve}/{cfg.train.patience})"
            )

        if epochs_no_improve >= cfg.train.patience:
            print(f"[{model_name}] subject {subject_id}: early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUTPUT_DIR / f"{model_name}_subject{subject_id}.pt"
    torch.save(best_state, ckpt_path)

    test_acc, test_kappa = evaluate_arrays(model, X_test, y_test, device, cfg.train.batch_size)
    print(
        f"[{model_name}] subject {subject_id}: done - "
        f"val_acc={best_val_acc:.3f}, test_acc={test_acc:.3f}, test_kappa={test_kappa:.3f}"
    )

    return {
        "subject":       subject_id,
        "model":         model_name,
        "val_accuracy":  best_val_acc,
        "test_accuracy": test_acc,
        "test_kappa":    test_kappa,
        "checkpoint":    str(ckpt_path),
    }


def run_all_subjects(model_name, cfg: Config = None):
    """Trains one model per subject and writes aggregate results to disk.

    Parameters
    ----------
    model_name : str
        Either "eegnet" or "conformer".
    cfg : Config, optional
        Full project configuration; defaults to `Config()`.

    Returns
    -------
    list of dict
        Per-subject results, in subject order.
    """
    cfg = cfg or Config()
    print(f"Training {model_name} for {len(SUBJECT_IDS)} subjects (within-subject, official session_T/session_E split)")
    results = [train_one_subject(sid, model_name, cfg) for sid in SUBJECT_IDS]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / f"{model_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()

    run_all_subjects(args.model)
