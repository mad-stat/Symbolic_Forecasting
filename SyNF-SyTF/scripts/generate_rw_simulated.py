#!/usr/bin/env python3
"""Generate the random-walk dataset used by SyTF (200 observations)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_rw(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Random walk starting at x=0 with Gaussian increments."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1, size=n - 1)
    x = np.zeros(n)
    x[1:] = np.cumsum(steps)
    return pd.DataFrame({"Time": np.arange(1, n + 1), "x": x})


def main():
    parser = argparse.ArgumentParser(description="Generate RW_Simulated_datset.csv")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "RW_Simulated_datset.csv",
    )
    parser.add_argument("-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_rw(n=args.n, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
