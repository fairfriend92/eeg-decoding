"""Packages a trained checkpoint into a Codabench-ready ZIP for Track 02.

Every file in the ZIP must sit at its root and the entry Codabench loads
must be named exactly ``submission.py`` (its own contract; see
README_NEURIPS2026.md), so this renames the selected model's source file
(``{model}_submission.py``) to ``submission.py`` on the way into the
archive, alongside the checkpoint renamed to ``weights.pt``.

Usage
-----
    python scripts/submission/build_submission.py CHECKPOINT [--model {eegnet,patch_transformer,conformer}] [-o OUTPUT]

CHECKPOINT is the file written by
``scripts/neuralbench_eval.py --save-checkpoint``.
"""

import argparse
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path,
        help="net+probe state dict from scripts/neuralbench_eval.py --save-checkpoint",
    )
    parser.add_argument(
        "--model", choices=["eegnet", "patch_transformer"], default="eegnet",
        help="Which {model}_submission.py to package (default: eegnet).",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("my_submission.zip"),
        help="Output ZIP path (default: my_submission.zip in the current directory).",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    submission_py = Path(__file__).parent / f"{args.model}_submission.py"
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_py, "submission.py")
        zf.write(args.checkpoint, "weights.pt")

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
