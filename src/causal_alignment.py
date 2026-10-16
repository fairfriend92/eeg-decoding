"""Causal per-participant z-score and Euclidean alignment for streamed windows.

A streaming scorer hands the model one window at a time, so the full-trial
statistics of data_loader.get_cross_subject_data are not available. This
module computes the same two transforms from the windows seen so far in one
stream, each window included once it arrives. It depends on numpy only, so
scripts/submission/ carries a verbatim copy.
"""

import numpy as np


class CausalAlign:
    """Running per-channel z-score followed by running Euclidean alignment.

    The z-score uses the pooled mean and variance of every sample seen so far
    in the stream. The alignment whitens the z-scored window with the inverse
    square root of the mean covariance of the z-scored windows seen so far.
    Both are shrunk toward a prior worth a number of windows, so the first
    windows are defined. Nothing depends on the channel count or window length.

    Parameters
    ----------
    prior_mean, prior_var : ndarray, shape (n_channels,), optional
        Per-channel mean and variance of the training pool, the z-score prior.
        None uses a prior of zero mean and unit variance, which only makes
        sense with z_prior_windows=0.
    z_prior_windows : float
        Weight of the z-score prior, in windows. 0 estimates the z-score from
        the stream alone, which is defined from the first window.
    cov_prior_windows : float
        Weight, in windows, of an identity covariance prior for the alignment.
        Z-scored windows have unit variance per channel, so identity is the
        natural prior. It keeps the first whitener full rank.
    eps : float
        Floor on the covariance eigenvalues, as in data_loader.euclidean_align.
    """

    def __init__(self, prior_mean=None, prior_var=None, z_prior_windows=0.0, cov_prior_windows=1.0, eps=1e-12):
        self.prior_mean = None if prior_mean is None else np.asarray(prior_mean, dtype=np.float64)
        self.prior_var = None if prior_var is None else np.asarray(prior_var, dtype=np.float64)
        self.z_prior_windows = float(z_prior_windows)
        self.cov_prior_windows = float(cov_prior_windows)
        self.eps = eps
        self.reset()

    def reset(self):
        """Forgets every window seen, ready for a new stream."""
        self.n_windows = 0
        self.sum_mean = None
        self.sum_sq = None
        self.sum_cov = None

    def __call__(self, window):
        """Adds one window to the running statistics and returns it aligned.

        Parameters
        ----------
        window : ndarray, shape (n_channels, n_times)
            The next window of the stream.

        Returns
        -------
        ndarray, shape (n_channels, n_times)
            The z-scored, whitened window, as float32.
        """
        x = np.asarray(window, dtype=np.float64)
        n_channels, n_times = x.shape
        if self.n_windows == 0:
            self.sum_mean = np.zeros(n_channels)
            self.sum_sq = np.zeros(n_channels)
            self.sum_cov = np.zeros((n_channels, n_channels))
        self.n_windows += 1
        self.sum_mean += x.mean(axis=1)
        self.sum_sq += (x ** 2).mean(axis=1)

        k = self.z_prior_windows if self.prior_mean is not None else 0.0
        prior_mean = self.prior_mean if k else 0.0
        prior_sq = (self.prior_var + self.prior_mean ** 2) if k else 0.0
        mean = (k * prior_mean + self.sum_mean) / (k + self.n_windows)
        var = (k * prior_sq + self.sum_sq) / (k + self.n_windows) - mean ** 2
        z = (x - mean[:, None]) / (np.sqrt(np.maximum(var, 0.0)) + 1e-8)[:, None]

        self.sum_cov += z @ z.T / n_times
        cov = (self.cov_prior_windows * np.eye(n_channels) + self.sum_cov) / (self.cov_prior_windows + self.n_windows)
        eigvals, eigvecs = np.linalg.eigh(cov)
        whitener = (eigvecs / np.sqrt(np.maximum(eigvals, self.eps))) @ eigvecs.T
        return (whitener @ z).astype(np.float32)


def causal_align_stream(X, **kwargs):
    """Aligns one stream of windows causally, in recording order.

    Parameters
    ----------
    X : ndarray, shape (n_windows, n_channels, n_times)
        One participant's windows (or one session's), in recording order.
    **kwargs
        Passed to CausalAlign.

    Returns
    -------
    ndarray, shape (n_windows, n_channels, n_times)
        Each window aligned using only itself and the windows before it.
    """
    aligner = CausalAlign(**kwargs)
    return np.stack([aligner(window) for window in X])
