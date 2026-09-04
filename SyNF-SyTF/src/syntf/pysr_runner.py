"""PySR symbolic regression wrapper for SyTF."""

from __future__ import annotations

import time
from pathlib import Path

from pysr import PySRRegressor

from src.config import ensure_results_dir


def fit_syntf(
    X_train,
    y_train,
    *,
    n_lags: int,
    niterations: int = 25,
    run_name: str = "syntf_rw_simulated_lag1",
) -> tuple[PySRRegressor, str, float]:
    """
    Fit SyTF (PySR) and save the discovered equation.

    Returns (fitted_model, equation_string, execution_time_seconds).
    """
    out_dir = ensure_results_dir(run_name)

    start = time.time()
    model = PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "-", "*"],
        unary_operators=["sin", "cos"],
        model_selection="best",
    )
    model.fit(X_train.values, y_train.values)
    elapsed = time.time() - start

    best = model.get_best()
    equation = str(best["sympy_format"])

    eq_path = out_dir / f"{run_name}_equation.txt"
    eq_path.write_text(equation, encoding="utf-8")
    (out_dir / "execution_time.txt").write_text(
        f"Main function execution time: {elapsed:.2f} seconds\n",
        encoding="utf-8",
    )
    return model, equation, elapsed
