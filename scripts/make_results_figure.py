"""Regenerates output/figures/results_summary.png, the figure embedded in README.md.

Left panel: BCI2a EEGNet variants. Right panel: Dreyer2023 EEGNet against the
EEG Conformer, read from output/dreyer2023/summary.json.

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

# Mean, std across the 9 subjects: the BCI2a Results table in README.md.
BCI2A_BARS = [
    ("EEGNet\n(cropped, ensemble)", 0.636, 0.168),
    ("EEGNet\n(cropped + CSP-init,\nensemble)", 0.685, 0.115),
]

BAR_COLORS = ["#4c72b0", "#4c72b0", "#c44e52"]


def draw_panel(ax, title, bars, chance, colors):
    """Draws one bar panel with error bars and a dashed chance line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    title : str
        Panel title.
    bars : list of (str, float, float)
        Label, mean, and std for each bar.
    chance : float
        Chance-level accuracy.
    colors : list of str
        One bar color per entry in `bars`.
    """
    labels, means, stds = zip(*bars)
    ax.bar(labels, means, yerr=stds, capsize=6, color=colors[: len(bars)])
    for x, mean in enumerate(means):
        ax.text(x, 0.04, f"{mean:.2f}", ha="center", color="white", fontsize=14)
    ax.axhline(chance, color="gray", linestyle="--", label=f"chance ({chance:.0%})")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper right")


def main():
    with open(OUTPUT_DIR / "dreyer2023" / "summary.json") as f:
        dreyer = json.load(f)
    dreyer_bars = [
        ("EEGNet", dreyer["eegnet"]["final_accuracy_mean"], dreyer["eegnet"]["final_accuracy_std"]),
        ("EEG Conformer", dreyer["conformer"]["final_accuracy_mean"], dreyer["conformer"]["final_accuracy_std"]),
    ]

    fig, (ax_bci, ax_dreyer) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [2, 2]})
    draw_panel(ax_bci, "BCI2a, 4 classes, 9 subjects", BCI2A_BARS, 0.25, BAR_COLORS)
    draw_panel(ax_dreyer, "Dreyer2023, 2 classes, 87 subjects", dreyer_bars, 0.5, [BAR_COLORS[0], BAR_COLORS[2]])
    ax_bci.set_ylabel("Test accuracy, mean ± std across subjects")

    fig.tight_layout()
    out = OUTPUT_DIR / "figures" / "results_summary.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
