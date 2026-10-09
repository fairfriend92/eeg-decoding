"""Download-only fetch of a registered dataset's per-subject files.

Populates the MOABB/MNE data directory (``MNE_DATA``) without epoching or
training, so the download can run on a CPU-only session ahead of a GPU run.
Subjects whose files are already present are skipped by MOABB. Retries on
rate limits and timeouts are applied by moabb_downloads.configure_downloads.
"""

import argparse
import threading
import time

import data_loader  # noqa: F401 (applies MNE_DATA and the download configuration)
from datasets import DATASET_PRESETS, MOABB_DATASET_CLASSES


def _start_heartbeat(interval_s=30):
    """Prints a heartbeat line from a daemon thread until the process exits.

    Parameters
    ----------
    interval_s : float
        Seconds between heartbeat lines.
    """
    def beat():
        while True:
            time.sleep(interval_s)
            print("[download] heartbeat", flush=True)

    threading.Thread(target=beat, daemon=True).start()


def download_dataset(dataset_name, subject_ids=None):
    """Downloads every requested subject of a registered dataset.

    Parameters
    ----------
    dataset_name : str
        Key of datasets.DATASET_PRESETS.
    subject_ids : list of int, optional
        Subjects to fetch. Defaults to the preset's subject_ids.

    Raises
    ------
    ValueError
        If dataset_name is not registered.
    """
    if dataset_name not in DATASET_PRESETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choices: {sorted(DATASET_PRESETS)}")

    subject_ids = subject_ids or DATASET_PRESETS[dataset_name].subject_ids
    dataset = MOABB_DATASET_CLASSES[dataset_name]()
    _start_heartbeat()

    for i, subject_id in enumerate(subject_ids, start=1):
        dataset.data_path(subject_id)
        print(f"[download] subject {subject_id} ready ({i}/{len(subject_ids)})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASET_PRESETS))
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=None,
        help="Subject ids to fetch. Defaults to the dataset preset's subject list.",
    )
    args = parser.parse_args()
    download_dataset(args.dataset, args.subjects)
