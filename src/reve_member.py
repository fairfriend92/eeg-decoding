"""Frozen-REVE plus logistic regression member for ensembling.

Per subject, fits a standardized L2 logistic regression on the REVE
features from reve_embeddings.py and writes softmax probabilities in the
same format as dump_predictions.py, to output/{dataset}/predictions_reve.npz.
Test probabilities come from a fit on the training split, validation
probabilities from a fit on the training split minus its trailing validation
slice. Runs on CPU from the saved embeddings.

Must be run with this file's own directory (src/) as the script's
directory. See train.py for why.
"""

import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import Config, add_dataset_cli_args, resolve_config
from train import output_dir_for


def _fit_predict(emb_fit, y_fit, emb_eval, pooling):
    emb_fit, emb_eval = emb_fit.astype(np.float32), emb_eval.astype(np.float32)
    if pooling == "mean":
        emb_fit, emb_eval = emb_fit.mean(axis=1), emb_eval.mean(axis=1)
    else:
        emb_fit, emb_eval = emb_fit.reshape(len(emb_fit), -1), emb_eval.reshape(len(emb_eval), -1)
    # With fewer trials than features, keeping every principal component is a rotation of the
    # trials' own subspace, so the L2 solution is unchanged and the fit is much faster.
    # The regularization strength is chosen by inner CV on the fitting trials only.
    clf = make_pipeline(StandardScaler(), PCA(svd_solver="full"), LogisticRegressionCV(Cs=np.logspace(-5, 0, 6), cv=5, max_iter=1000))
    clf.fit(emb_fit, y_fit)
    return clf.predict_proba(emb_eval)


def dump_reve(cfg: Config, subjects=None, pooling="channel"):
    """Fits the per-subject classifiers and dumps their probabilities.

    Parameters
    ----------
    cfg : Config
        Full project configuration. output/{dataset}/reve_embeddings.npz
        must exist.
    subjects : list of int, optional
        Subset of subject ids; defaults to every subject in the embeddings.
        A subset is not written to disk.
    pooling : str
        "mean" averages the features over channels. "channel" concatenates
        the per-channel features, which keeps spatial information.

    Returns
    -------
    dict
        Arrays "subject", "y" and "probs", one row per test trial, and
        "val_subject", "val_y" and "val_probs", one row per validation trial.
    """
    emb = np.load(output_dir_for(cfg) / "reve_embeddings.npz")
    val, test = {"subject": [], "y": [], "probs": []}, {"subject": [], "y": [], "probs": []}
    for subject_id in subjects or np.unique(emb["train_subject"]):
        rows = {name: emb[f"{name}_subject"] == subject_id for name in ("fit", "val", "train", "test")}
        pv = _fit_predict(emb["fit_emb"][rows["fit"]], emb["fit_y"][rows["fit"]], emb["val_emb"][rows["val"]], pooling)
        p = _fit_predict(emb["train_emb"][rows["train"]], emb["train_y"][rows["train"]], emb["test_emb"][rows["test"]], pooling)
        yv, y = emb["val_y"][rows["val"]], emb["test_y"][rows["test"]]
        print(f"[reve] subject {subject_id}: acc={(p.argmax(axis=1) == y).mean():.3f}", flush=True)
        for part, labels, probs in ((val, yv, pv), (test, y, p)):
            part["subject"].append(np.full(len(labels), subject_id))
            part["y"].append(labels)
            part["probs"].append(probs)

    dump = {k: np.concatenate(v) for k, v in test.items()}
    dump.update({f"val_{k}": np.concatenate(v) for k, v in val.items()})
    if subjects is None:
        np.savez(output_dir_for(cfg) / "predictions_reve.npz", **dump)
    return dump


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Subset to score without writing files.")
    parser.add_argument("--pooling", choices=["mean", "channel"], default="channel")
    add_dataset_cli_args(parser)
    args = parser.parse_args()

    dump_reve(resolve_config(args), subjects=args.subjects, pooling=args.pooling)
