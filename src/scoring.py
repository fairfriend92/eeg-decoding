"""Balanced accuracy as the competition leaderboard computes it."""

import numpy as np


def balanced_accuracy(y, pred):
    """Mean recall over the classes present in the labels or the predictions.

    A class that is predicted but absent from the labels counts with recall 0,
    where scikit-learn's balanced_accuracy_score leaves it out.

    Parameters
    ----------
    y, pred : ndarray, shape (n_rows,)
        Integer labels and predicted classes.

    Returns
    -------
    float
        NaN when there are no rows.
    """
    classes = np.union1d(y, pred)
    if not len(classes):
        return float("nan")
    return float(np.mean([(pred[y == c] == c).mean() if (y == c).any() else 0.0 for c in classes]))


def cell_balanced_accuracies(y, pred, subject, session, context=None):
    """Balanced accuracy of every (participant, session) or (participant, session, context) cell.

    The mean of the returned scores weighs each cell equally, so a participant
    with many windows does not outweigh one with few.

    Parameters
    ----------
    y, pred : ndarray, shape (n_rows,)
        Integer labels and predicted classes.
    subject, session : ndarray, shape (n_rows,)
        Participant and session label of each row.
    context : ndarray, shape (n_rows,), optional
        Context label of each row. When None, a cell is one session of one
        participant.

    Returns
    -------
    ndarray, shape (n_cells,)
        Score of each cell, cells in the sorted order of their string keys
        (participant 10 comes before participant 2).
    """
    keys = [subject.astype(str), session.astype(str)] + ([] if context is None else [context.astype(str)])
    _, cell_of_row = np.unique(np.stack(keys, axis=1), axis=0, return_inverse=True)
    cell_of_row = cell_of_row.reshape(-1)
    return np.array([balanced_accuracy(y[cell_of_row == c], pred[cell_of_row == c]) for c in range(cell_of_row.max() + 1)])
