"""Generate nine clean, publication-style JPEG plots from episode metrics.

Usage:
    python plot_episode_analysis.py path/to/episode_metrics.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


BALANCE = 50_000.0
BLUE = "#043294"
TEAL = "#A30909"
GOLD = "#000000"
RED = "#000000"
INK = "#000000"
MUTED = "#000000"
GRID = "#888D92"


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_").lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create nine episode-level analysis figures.")
    parser.add_argument("csv", type=Path, help="Episode metrics CSV")
    parser.add_argument("-o", "--output-dir", type=Path, help="Output folder")
    parser.add_argument("--initial-balance", type=float, default=BALANCE,
                        help="Starting portfolio balance for return percentage (default: 50000)")
    args = parser.parse_args()
    if args.initial_balance <= 0:
        parser.error("--initial-balance must be greater than zero")

    data = pd.read_csv(args.csv)
    data.columns = [str(c).strip().lower() for c in data.columns]
    if data.empty:
        raise ValueError("CSV contains no rows")
    x = pd.to_numeric(data.get("episode", pd.Series(np.arange(1, len(data) + 1))), errors="coerce")
    if x.isna().all():
        x = pd.Series(np.arange(1, len(data) + 1))
    out = args.output_dir or (Path(__file__).resolve().parent / "visualizations" / f"{args.csv.stem}_plots")
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#AEB7C0", "axes.linewidth": 0.75,
        "grid.color": GRID, "grid.linewidth": 0.65,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })

    def save(fig: plt.Figure, name: str) -> None:
        fig.tight_layout(pad=1.15)
        fig.savefig(out / f"{slug(args.csv.stem)}_{name}.jpeg", dpi=350,
                    format="jpeg", pil_kwargs={"quality": 94, "subsampling": 0})
        plt.close(fig)

    def line_chart(column: str, ylabel: str, name: str, color: str, scale: float = 1.0,
                   zero_line: bool = False, formatter: str | None = None) -> None:
        if column not in data:
            return
        y = pd.to_numeric(data[column], errors="coerce") * scale
        valid = x.notna() & y.notna()
        if not valid.any():
            return
        fig, ax = plt.subplots(figsize=(6.8, 4.1))
        ax.plot(x[valid], y[valid], color=color, linewidth=1.55, solid_capstyle="round")
        ax.scatter(x[valid], y[valid], s=13, color=color, edgecolor="white", linewidth=0.45, zorder=3)
        if zero_line:
            ax.axhline(0, color=MUTED, linewidth=0.75, linestyle=(0, (3, 3)), zorder=0)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        if formatter:
            ax.yaxis.set_major_formatter(formatter)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, name)

    line_chart("final_portfolio_value", "Final portfolio value", "final_portfolio_value", BLUE)
    line_chart("sharpe_ratio", "Sharpe ratio", "sharpe_ratio", TEAL, zero_line=True)
    line_chart("maximum_drawdown", "Maximum drawdown (%)", "maximum_drawdown", RED,
               scale=100, zero_line=True)
    # Compare action counts using a restrained, colorblind-aware three-color set.
    action_columns = [c for c in ("buy_count", "sell_count", "hold_count") if c in data]
    if action_columns:
        fig, ax = plt.subplots(figsize=(6.8, 4.1))
        action_style = {"buy_count": (BLUE, "Buy"), "sell_count": (GOLD, "Sell"), "hold_count": (TEAL, "Hold")}
        for col in action_columns:
            y = pd.to_numeric(data[col], errors="coerce")
            valid = x.notna() & y.notna()
            color, label = action_style[col]
            ax.plot(x[valid], y[valid], color=color, linewidth=1.4, label=label)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Action count")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.0),
                  handlelength=1.8, columnspacing=1.6)
        save(fig, "action_counts")

    # Spearman correlation across the requested performance, risk, activity and loss metrics.
    corr_cols = [c for c in ("total_return", "final_portfolio_value", "sharpe_ratio", "maximum_drawdown",
                             "mean_daily_return", "trade_count", "buy_count", "sell_count", "hold_count",
                             "mean_training_loss") if c in data]
    corr_data = data[corr_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if len(corr_data.columns) >= 2:
        corr = corr_data.corr(method="spearman")
        labels = [c.replace("_", " ").replace("maximum drawdown", "max drawdown")
                  .replace("mean daily return", "daily return").replace("mean training loss", "train loss")
                  .replace("final portfolio value", "final value").title() for c in corr.columns]
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest", aspect="equal")
        ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(labels)), labels)
        ax.tick_params(length=0, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = corr.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(value) > 0.58 else INK)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04, ticks=[-1, -0.5, 0, 0.5, 1])
        colorbar.set_label("Spearman ρ", rotation=90, labelpad=10)
        colorbar.outline.set_visible(False)
        save(fig, "spearman_correlation")

    generated = sorted(out.glob(f"{slug(args.csv.stem)}_*.jpeg"))
    print(f"Created {len(generated)} JPEG figures in {out}")


if __name__ == "__main__":
    main()
