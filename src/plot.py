"""Plot every scored question against the original sentence index."""

import argparse
import csv
import json
import math
import os
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(run: dict, output: Path):
    questions = run["questions"]
    sentences = run["document"]["sentences"]
    results = {row["ix"]: row["probabilities"] for row in run["results"]}
    if not results:
        raise ValueError("No scores to plot; run src/jev.py first")
    indices = [s["ix"] for s in sentences]
    # Missing sentences remain NaN: partial runs never look like zero probabilities.
    values = np.array([[results.get(ix, {}).get(q, np.nan) for ix in indices] for q in questions])
    if np.any(np.isinf(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Probabilities must be finite values between zero and one")
    output.mkdir(parents=True, exist_ok=True)
    subtitle = f"{run['model']} · {len(results)}/{len(sentences)} steps · full prefix context"
    fig, axes = plt.subplots(math.ceil(len(questions) / 2), 2, squeeze=False,
                             figsize=(15, 2.8 * math.ceil(len(questions) / 2)),
                             sharex=True, sharey=True, layout="constrained")
    for ax, (key, question), row in zip(axes.flat, questions.items(), values):
        ax.plot(indices, row, color="#2368a2", linewidth=1.2, marker=".", markersize=3)
        ax.set_title(textwrap.fill(question, 65), fontsize=9, loc="left")
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("P(yes)")
        ax.set_xlabel("Sentence ix")
        ax.tick_params(labelbottom=True)
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(questions):]:
        ax.set_visible(False)
    fig.suptitle(f"Jevinator — per-sentence probabilities\n{subtitle}", fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"questions.{suffix}", dpi=160)
    plt.close(fig)

    fig, (importance_ax, heat_ax) = plt.subplots(
        2, 1, figsize=(17, max(7, len(questions) * 0.36 + 2)),
        gridspec_kw={"height_ratios": [1, 5]}, layout="constrained",
    )
    importance_ax.plot(indices, [s["importance"] for s in sentences], color="#6c4c82")
    importance_ax.set_ylabel("Original\nimportance")
    importance_ax.set_xlim(indices[0] - 0.5, indices[-1] + 0.5)
    importance_ax.grid(alpha=0.2)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dedede")
    heat = heat_ax.imshow(values, aspect="auto", interpolation="nearest", vmin=0, vmax=1,
                         cmap=cmap, extent=[indices[0] - 0.5, indices[-1] + 0.5,
                                           len(questions) - 0.5, -0.5])
    heat_ax.set_yticks(range(len(questions)), [q.replace("_", " ") for q in questions])
    heat_ax.set_xlabel("Sentence ix (gray = unscored)")
    fig.colorbar(heat, ax=heat_ax, label="P(yes)", shrink=0.8)
    fig.suptitle(f"Jevinator — all questions\n{subtitle}")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"heatmap.{suffix}", dpi=160)
    plt.close(fig)

    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ix", "classification", "importance", "text", *questions])
        for sentence in sentences:
            writer.writerow([sentence[k] for k in ("ix", "classification", "importance", "text")]
                            + [results.get(sentence["ix"], {}).get(q, "") for q in questions])
    print(f"Saved question plots, heatmap (PNG/PDF), and scores.csv to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=ROOT / "results/jev.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/plots")
    args = parser.parse_args()
    plot(json.loads(args.input.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
