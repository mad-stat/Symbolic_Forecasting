"""Shared time-series utilities for SyNF and SyTF."""

from __future__ import annotations

import pandas as pd
import sympy as sy


def series_to_supervised(
    data,
    n_in: int = 1,
    n_out: int = 1,
    dropnan: bool = True,
) -> pd.DataFrame:
    """Frame a univariate series as a supervised learning dataset."""
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f"var{j + 1}(t-{i})" for j in range(n_vars)]

    for i in range(n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f"var{j + 1}(t)" for j in range(n_vars)]
        else:
            names += [f"var{j + 1}(t+{i})" for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def str_to_function(func_str: str, lags: int):
    """Parse a sympy equation string into a numpy-callable function."""
    try:
        variables = sy.symbols(" ".join(f"x{i}" for i in range(lags)))
        expr = sy.sympify(func_str)
        return sy.lambdify(variables, expr, modules=["numpy"])
    except (sy.SympifyError, TypeError, ValueError) as exc:
        print(f"Error: Invalid mathematical expression: {func_str}")
        print(f"Exception: {exc}")
        return None
