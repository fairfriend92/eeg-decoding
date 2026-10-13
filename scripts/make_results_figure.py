"""Regenerates the figures in output/figures/ embedded in README.md.

One figure per README results table (BCI2a, Dreyer2023 per subject, Dreyer2023
cross-subject), plus one each for the alignment and tokenization sections. The
Dreyer2023 per-subject figure reads output/dreyer2023/summary.json. All other
values are the README table entries.

Usage
-----
    python scripts/make_results_figure.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import OUTPUT_DIR

FIGURES_DIR = OUTPUT_DIR / "figures"
BLUE, ORANGE, GRAY = "#4c72b0", "#dd8452", "#8c8c8c"


def draw_groups(ax, labels, series, chance, xlabel):
    """Draws horizontal grouped bars with error bars and a dashed chance line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    labels : list of str
        One label per group, drawn top to bottom.
    series : list of (str, str, list)
        Legend name, color, and one `(mean, std)` entry per group. A `None`
        entry leaves that bar out.
    chance : float
        Chance level, drawn as a dashed vertical line.
    xlabel : str
        Label of the score axis.
    """
    height = 0.8 / len(series)
    for k, (name, color, values) in enumerate(series):
        offset = (k - (len(series) - 1) / 2) * height
        present = [i for i, value in enumerate(values) if value is not None]
        for i in present:
            mean, std = values[i]
            y = i + offset
            ax.barh(y, mean, height * 0.92, xerr=std, capsize=3, color=color, label=name if i == present[0] else None)
            ax.text(0.01, y, f"{mean:.3f}", va="center", color="white", fontsize=9)
    ax.axvline(chance, color="gray", linestyle="--", label=f"chance ({chance:g})")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)


def save(fig, name):
    """Writes a figure into output/figures/ and closes it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        File name, including extension.
    """
    fig.tight_layout()
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def make_bci2a():
    """Writes the BCI2a EEGNet figure (README Results, BCI2a)."""
    labels = ["Ensemble, cropped", "Final fit, cropped", "Ensemble, cropped + CSP-init", "Final fit, cropped + CSP-init"]
    values = [(0.636, 0.168), (0.632, 0.177), (0.685, 0.115), (0.643, 0.119)]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    draw_groups(ax, labels, [("EEGNet", BLUE, values)], 0.25, "Test accuracy, mean ± std across 9 subjects")
    save(fig, "results_bci2a.png")


def make_dreyer_per_subject():
    """Writes the Dreyer2023 per-subject figure from summary.json."""
    with open(OUTPUT_DIR / "dreyer2023" / "summary.json") as f:
        summary = json.load(f)
    labels = ["Best checkpoint", "Final fit"]
    series = []
    for model, name, color in [("eegnet", "EEGNet", BLUE), ("conformer", "EEG Conformer", ORANGE)]:
        s = summary[model]
        series.append((name, color, [(s["accuracy_mean"], s["accuracy_std"]), (s["final_accuracy_mean"], s["final_accuracy_std"])]))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    draw_groups(ax, labels, series, 0.5, "Test accuracy, mean ± std across 87 subjects")
    save(fig, "results_dreyer2023.png")


def make_dreyer_cross_subject():
    """Writes the Dreyer2023 cross-subject members and ensembles figure."""
    members = [
        ("Conformer, dense tokens", None, (0.869, 0.074)),
        ("Conformer", None, (0.864, 0.074)),
        ("EEGNet", (0.813, 0.107), (0.861, 0.075)),
        ("Patch transformer,\noverlapping tokens", (0.818, 0.086), (0.850, 0.069)),
        ("REVE, fine-tuned", (0.794, 0.089), None),
        ("Riemannian, recentering", None, (0.685, 0.150)),
        ("Riemannian tangent space", (0.641, 0.137), (0.677, 0.144)),
        ("FBCSP", (0.625, 0.123), (0.671, 0.149)),
    ]
    ensembles = [
        ("Unaligned\nEEGNet, patch transformer, REVE", (0.829, 0.095), (0.834, 0.092)),
        ("Aligned\nEEGNet, patch transformer", (0.869, 0.074), (0.874, 0.069)),
        ("Aligned, plus Conformer", (0.879, 0.067), (0.882, 0.068)),
        ("Aligned, plus dense-token Conformer", (0.883, 0.065), (0.883, 0.065)),
        ("Aligned, plus both Conformers", (0.886, 0.066), (0.886, 0.066)),
        ("Aligned and unaligned mixed", (0.858, 0.084), (0.861, 0.080)),
    ]
    fig, (ax_m, ax_e) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1]})
    draw_groups(
        ax_m, [m[0] for m in members],
        [("Unaligned", GRAY, [m[1] for m in members]), ("Euclidean-aligned", ORANGE, [m[2] for m in members])],
        0.5, "Balanced accuracy, mean ± std across 21 participants",
    )
    draw_groups(
        ax_e, [e[0] for e in ensembles],
        [("Selected subset", BLUE, [e[1] for e in ensembles]), ("Uniform average of pool", GRAY, [e[2] for e in ensembles])],
        0.5, "Balanced accuracy, mean ± std across 21 participants",
    )
    ax_m.set_title("Members")
    ax_e.set_title("Ensembles")
    save(fig, "results_dreyer2023_cross_subject.png")


def make_alignment():
    """Writes the alignment figure (README Methodology, Alignment)."""
    members = [
        ("Conformer", None, (0.864, 0.074)),
        ("EEGNet", (0.813, 0.107), (0.861, 0.075)),
        ("Patch transformer", (0.818, 0.086), (0.850, 0.069)),
        ("Riemannian, recentering", (0.641, 0.137), (0.685, 0.150)),
        ("Riemannian, Euclidean", (0.641, 0.137), (0.677, 0.144)),
        ("FBCSP", (0.625, 0.123), (0.671, 0.149)),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    draw_groups(
        ax, [m[0] for m in members],
        [("Unaligned", GRAY, [m[1] for m in members]), ("Aligned", ORANGE, [m[2] for m in members])],
        0.5, "Balanced accuracy, mean ± std across 21 participants",
    )
    save(fig, "alignment.png")


def make_tokenization():
    """Writes the tokenization figure (README Methodology, Tokenization)."""
    labels = ["Patch transformer,\ndefault tokens", "Patch transformer,\noverlapping tokens", "Conformer", "Conformer,\ndense tokens"]
    bci2a = [(0.364, 0.086), (0.418, 0.145), (0.520, 0.167), (0.518, 0.147)]
    unaligned = [(0.736, 0.094), (0.818, 0.086), None, None]
    aligned = [None, (0.850, 0.069), (0.864, 0.074), (0.869, 0.074)]
    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(12, 4.4))
    draw_groups(ax_b, labels, [("Single split", BLUE, bci2a)], 0.25, "BCI2a accuracy, mean ± std across 9 subjects")
    draw_groups(
        ax_c, labels, [("Unaligned", GRAY, unaligned), ("Euclidean-aligned", ORANGE, aligned)],
        0.5, "Cross-subject balanced accuracy, mean ± std across 21 participants",
    )
    ax_c.set_yticklabels([])
    save(fig, "tokenization.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    make_bci2a()
    make_dreyer_per_subject()
    make_dreyer_cross_subject()
    make_alignment()
    make_tokenization()


if __name__ == "__main__":
    main()
