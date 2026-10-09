"""Download configuration for every MOABB dataset fetch.

Importing this module patches MOABB's downloader once: verified HTTPS, the
OSF storage-host redirect, a longer read timeout, retries on rate limits and
timeouts, a quiet log, and no NEMAR round-trips for a subject already on
disk. It covers libraries built on MOABB as well, so a
process that fetches MOABB data without the rest of the pipeline can import
just this module.
"""

import logging
import re
import time
from pathlib import Path

import mne_bids
import moabb.datasets.base as moabb_base
import moabb.datasets.download as moabb_download
import pooch
import requests

MAX_DOWNLOAD_ATTEMPTS = 10

# A file URL on OSF's storage hosts, e.g. files.de-1.osf.io/v1/resources/<node>/providers/osfstorage/<id>.
OSF_STORAGE_URL = re.compile(r"^https://files(\.[a-z0-9-]+)?\.osf\.io/v1/resources/[^/]+/providers/osfstorage/([A-Za-z0-9]+)/?$")


def direct_osf_url(url):
    """Rewrites an OSF storage-host file URL to the osf.io/download redirect.

    The storage hosts (files.osf.io, files.de-1.osf.io) return HTTP 429 after a
    few requests. The redirect serves the same file without that limit.

    Parameters
    ----------
    url : str
        Any download URL.

    Returns
    -------
    str
        The redirect URL for an OSF storage-host file URL, otherwise url unchanged.
    """
    match = OSF_STORAGE_URL.match(url)
    return f"https://osf.io/download/{match.group(2)}" if match else url


def configure_downloads():
    """Configures MOABB's dataset downloads: verified HTTPS, OSF redirect, retries, quiet log.

    Applied once at import. Idempotent. Covers every MOABB dataset, including
    those fetched by libraries built on MOABB, once this module is imported.
    """
    # Pooch logs the SHA256 of every file fetched without a known hash.
    pooch.get_logger().setLevel(logging.WARNING)

    original_choose_downloader = getattr(moabb_download, "_original_choose_downloader", moabb_download.choose_downloader)
    original_retrieve = getattr(moabb_download, "_original_retrieve", moabb_download.retrieve)

    def verified_choose_downloader(*args, **kwargs):
        downloader = original_choose_downloader(*args, **kwargs)
        if hasattr(downloader, "kwargs"):
            # MOABB defaults to verify=False, which makes every request warn.
            downloader.kwargs["verify"] = True
            # The default 30 s read timeout is too short for OSF's redirect.
            downloader.kwargs["timeout"] = 120
        return downloader

    def retrying_retrieve(url, *args, **kwargs):
        url = direct_osf_url(url)
        for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            try:
                return original_retrieve(url, *args, **kwargs)
            except requests.exceptions.RequestException as e:
                if attempt == MAX_DOWNLOAD_ATTEMPTS:
                    raise
                # A rate limit needs a long cooldown. A timeout or reset usually clears at once.
                cooldown_s = 300 if "429" in str(e) else 30
                print(
                    f"[download] {type(e).__name__}, retry {attempt}/{MAX_DOWNLOAD_ATTEMPTS} "
                    f"in {cooldown_s}s", flush=True,
                )
                time.sleep(cooldown_s)

    moabb_download._original_choose_downloader = original_choose_downloader
    moabb_download._original_retrieve = original_retrieve
    moabb_download.choose_downloader = verified_choose_downloader
    moabb_download.retrieve = retrying_retrieve


# BIDS filter names as moabb passes them to nemar.download, mapped to mne_bids' plural search keys.
NEMAR_FILTER_KEYS = {
    "acquisition": "acquisitions",
    "run": "runs",
    "session": "sessions",
    "suffix": "suffixes",
    "task": "tasks",
}


def nemar_subject_is_local(target_dir, subject, bids_filters):
    """Whether a subject's EEG recordings are already in a NEMAR BIDS root.

    Parameters
    ----------
    target_dir : pathlib.Path
        Local BIDS root of the NEMAR deposit.
    subject : str
        BIDS subject label, as passed to ``nemar.download``.
    bids_filters : dict
        Extra BIDS filters of the dataset (``task``, ``run``, ...).

    Returns
    -------
    bool
        True when at least one matching recording exists and every match is a
        non-empty file. A partly transferred subject is not detected, so a
        corpus is completed by a ``--download`` pass, not by a run.
    """
    if not target_dir.is_dir():
        return False
    search_params = {
        "root": target_dir,
        "subjects": subject,
        "datatypes": "eeg",
        "extensions": moabb_base._RAW_EXTENSIONS,
    }
    for key, value in bids_filters.items():
        if key in NEMAR_FILTER_KEYS:
            search_params[NEMAR_FILTER_KEYS[key]] = value
    bids_paths = mne_bids.find_matching_paths(**search_params)
    return bool(bids_paths) and all(
        path.fpath.is_file() and path.fpath.stat().st_size > 0 for path in bids_paths
    )


def skip_local_nemar_downloads():
    """Makes MOABB skip ``nemar.download`` for a subject already on disk.

    ``nemar.download`` walks index, version, metadata and manifest over HTTPS
    before it consults ``trust_existing``, so a call with nothing to fetch still
    costs several round-trips per recording. The wrapper answers from the local
    BIDS tree instead and only reaches NEMAR for a missing subject or a
    ``force_update`` request. Applied once at import. Idempotent.
    """
    original_nemar_dl = getattr(moabb_base, "_original_nemar_dl", moabb_base.nemar_dl)

    def local_first_nemar_dl(nemar_id, dataset_code, path=None, force_update=False, subject=None, **kwargs):
        # Mirrors the target directory nemar_dl derives for the same arguments.
        target_dir = (
            Path(moabb_download.get_dataset_path(dataset_code, path))
            / f"MNE-{dataset_code.lower()}-data"
            / nemar_id
        )
        bids_filters = {k: v for k, v in kwargs.items() if k != "verbose"}
        if not force_update and subject is not None and nemar_subject_is_local(target_dir, subject, bids_filters):
            return str(target_dir)
        return original_nemar_dl(
            nemar_id, dataset_code, path=path, force_update=force_update, subject=subject, **kwargs
        )

    moabb_base._original_nemar_dl = original_nemar_dl
    moabb_base.nemar_dl = local_first_nemar_dl


configure_downloads()
skip_local_nemar_downloads()
