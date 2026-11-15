"""Plot DQN episode rewards using the layout from train.py."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CSV = Path("logs\DQN_episode_metrics_^GSPC_2012-2014_seed202611.csv")


def main():
    parser = argparse.ArgumentParser(description="Plot DQN rewards by episode.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Episode metrics CSV")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    parser.add_argument("--epochs", type=int, default=51, help="Number of initial episodes to plot")
    args = parser.parse_args()

    data = pd.read_csv(args.csv).sort_values("episode").head(args.epochs)
    if "total_return" not in data.columns:
        raise ValueError(f"CSV must contain a 'total_return' column: {args.csv}")
    if data.empty:
        raise ValueError(f"No episodes found in {args.csv}")
    output = args.output or Path("logs") / f"DQN_GSPC_2012-2014_returns_ep{len(data)}.jpeg"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Follow the typography used by the episode analysis figures, at a larger scale.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "axes.titleweight": "semibold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    # Keep the same figure settings and styling as utils.plot_portfolio_returns_across_episodes.
    plt.figure(figsize=(10, 5), dpi=100)
    plt.title("Portfolio Returns")
    plt.plot(data["total_return"], color="black")
    plt.xlabel("Episode")
    plt.ylabel("Return Value")
    plt.grid()
    plt.savefig(output)
    print(f"Saved plot to: {output.resolve()}")


if __name__ == "__main__":
    main()
