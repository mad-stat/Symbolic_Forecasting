"""Build cleaned SyNF and SyTF notebooks for GitHub."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"


def _cell(source: str, cell_type: str = "code") -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line if line.endswith("\n") else line + "\n" for line in source.splitlines()],
        "outputs": [],
        "execution_count": None,
    }


def _md(text: str) -> dict:
    return _cell(text, "markdown")


def build_synf() -> None:
    cells = [
        _md("# SyNF — Symbolic Neural Forecasting (San Juan weekly cases)\n\nTrains the Equation Learner (EQL) on weekly dengue cases with lag 4. Last 52 weeks are held out for testing."),
        _cell(
            """import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "src").exists():
    REPO_ROOT = Path.cwd().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import metrics

from src.config import DATA_DIR, ensure_results_dir
from src.data_utils import series_to_supervised, str_to_function
from src.synf.model import train_synf"""
        ),
        _cell(
            """# Load data and configure train/test split
lags = 4
run_name = "synf_sanjuan_lag4"

df = pd.read_csv(DATA_DIR / "Sanjuan_data_weekly.csv")
print(df.head())

col_index = 3  # Cases column
dat = np.array(df.iloc[:, col_index]).reshape(-1, 1)
split_point = dat.shape[0] - 52

tr_dat, ts_dat = dat[:split_point], dat[split_point:]
fore_hor = len(ts_dat)

train_series = TimeSeries.from_values(tr_dat.astype(np.float32))
test_series = TimeSeries.from_values(ts_dat.astype(np.float32))"""
        ),
        _cell(
            """# Supervised training matrix
tr_data = series_to_supervised(tr_dat, n_in=lags, n_out=1)
train_data = tr_data.reset_index(drop=True)

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

print("Training samples:", len(X_train))
print(X_train.head())"""
        ),
        _cell(
            """# Train SyNF (EQL)
func_val, elapsed = train_synf(
    X_train,
    y_train,
    n_lags=lags,
    units=["id", "mul", "cos", "sin", "div"],
    reg=1e-4,
    mask_thresh=0.05,
    iterations=1000,
    run_name=run_name,
)

print("Discovered equation:")
print(func_val)
print(f"Training time: {elapsed:.2f} s")"""
        ),
        _cell(
            """# Evaluate discovered equation
math_func = str_to_function(func_val, lags)

train_pred = [math_func(*np.array(X_train.iloc[i])) for i in range(len(X_train))]

plt.figure()
plt.plot(train_pred, label="train_pred")
plt.plot(y_train.values.flatten(), label="y_train")
plt.legend()
plt.title("SyNF training fit")
plt.show()"""
        ),
        _cell(
            """# One-step-ahead test predictions
full_data = series_to_supervised(dat, n_in=lags, n_out=1)
test_data = full_data.tail(len(ts_dat)).reset_index(drop=True)

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

model_name = "SyNF-Div-Reg"
test_pred_one = [math_func(*np.array(X_test.iloc[i])) for i in range(len(X_test))]

smape = metrics.smape(
    TimeSeries.from_values(np.array(y_test)),
    TimeSeries.from_values(np.array(test_pred_one)),
)
mae = metrics.mae(
    TimeSeries.from_values(np.array(y_test)),
    TimeSeries.from_values(np.array(test_pred_one)),
)
rmse = metrics.rmse(
    TimeSeries.from_values(np.array(y_test)),
    TimeSeries.from_values(np.array(test_pred_one)),
)

print(f"SMAPE: {smape:.4f}")
print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")

plt.figure()
plt.plot(test_pred_one, label="test_pred")
plt.plot(y_test.values.flatten(), label="y_test")
plt.legend()
plt.title("SyNF test fit (last 52 weeks)")
plt.show()"""
        ),
        _cell(
            """# Save test predictions
out_dir = ensure_results_dir(run_name)
pred_path = out_dir / f"{run_name}_predictions.csv"
pd.DataFrame({"y_test": y_test.values, "y_pred": test_pred_one}).to_csv(pred_path, index=False)
print(f"Saved predictions to {pred_path}")"""
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    (NOTEBOOKS / "SyNF.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")


def build_syntf() -> None:
    cells = [
        _md("# SyTF — Symbolic Time-series Forecasting (simulated random walk)\n\nFits PySR on a simulated random-walk series (200 observations, lag 1, 180 train / 20 test)."),
        _cell(
            """import os
import sys
from pathlib import Path

# PySR/Julia: disable stdio interception when supported
os.environ.setdefault("PYTHONCALL_STDIO", "0")

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "src").exists():
    REPO_ROOT = Path.cwd().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from darts import TimeSeries
from darts.metrics import metrics

from src.config import DATA_DIR, ensure_results_dir
from src.data_utils import series_to_supervised
from src.syntf.pysr_runner import fit_syntf"""
        ),
        _cell(
            """# Load simulated random-walk data
lags = 1
run_name = f"SyTF_RW_Simulated_lag{lags}"
train_size = 180
test_size = 20

data_path = DATA_DIR / "RW_Simulated_datset.csv"
if not data_path.exists():
    raise FileNotFoundError(
        f"Missing {data_path.name}. Run: python scripts/generate_rw_simulated.py"
    )

df = pd.read_csv(data_path)
print(df.head())
print("Number of rows:", len(df))

dat = df["x"].values.reshape(-1, 1)
tr_dat = dat[:train_size]
ts_dat = dat[train_size:]
fore_hor = len(ts_dat)"""
        ),
        _cell(
            """# Build supervised training matrix
tr_data_supervised = series_to_supervised(tr_dat, n_in=lags, n_out=1)
train_data = tr_data_supervised.reset_index(drop=True)

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# Clean column names for PySR
X_train.columns = X_train.columns.str.replace(r"[^0-9a-zA-Z_]+", "", regex=True)

print("X_train columns:", X_train.columns.tolist())
print(X_train.head())"""
        ),
        _cell(
            """# Fit SyTF (PySR)
model, final_eq, train_time = fit_syntf(
    X_train,
    y_train,
    n_lags=lags,
    niterations=25,
    run_name=run_name,
)

print("Sympy equation:")
print(final_eq)
print(f"Training time: {train_time:.2f} s")"""
        ),
        _cell(
            """# Training predictions
train_pred = model.predict(X_train.values)

plt.figure()
plt.plot(train_pred, label="train_pred")
plt.plot(y_train.values.flatten(), label="y_train")
plt.legend()
plt.title("SyTF training fit")
plt.show()"""
        ),
        _cell(
            """# Test set (last 20 points)
full_data = series_to_supervised(dat, n_in=lags, n_out=1)
test_data = full_data.tail(len(ts_dat)).reset_index(drop=True)

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]
X_test.columns = X_test.columns.str.replace(r"[^0-9a-zA-Z_]+", "", regex=True)

start_time = time.time()
test_pred_one = model.predict(X_test.values)
pred_time = time.time() - start_time
total_time = train_time + pred_time

model_name = "SyTF_lag1"
print(f"Total time (train + test prediction): {total_time:.2f} s")

plt.figure()
plt.plot(test_pred_one, label="test_pred")
plt.plot(y_test.values.flatten(), label="y_test")
plt.legend()
plt.title("SyTF test fit")
plt.show()"""
        ),
        _cell(
            """# Metrics
smape = metrics.smape(
    TimeSeries.from_values(np.array(y_test)),
    TimeSeries.from_values(test_pred_one),
)
mae = metrics.mae(
    TimeSeries.from_values(np.array(y_test)),
    TimeSeries.from_values(test_pred_one),
)
rmse = metrics.rmse(
    TimeSeries.from_values(np.array(y_test)),
    TimeSeries.from_values(test_pred_one),
)
marre = metrics.marre(
    TimeSeries.from_values(np.array(y_test)),
    TimeSeries.from_values(test_pred_one),
)

print(f"SMAPE: {smape:.4f}")
print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"MARRE: {marre:.4f}")"""
        ),
        _cell(
            """# Save test predictions
out_dir = ensure_results_dir(run_name)
pred_path = out_dir / f"{run_name}_predictions.csv"
pd.DataFrame({"y_test": y_test.values, "y_pred": test_pred_one}).to_csv(pred_path, index=False)
print(f"Saved predictions to {pred_path}")"""
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    (NOTEBOOKS / "SyTF.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    build_synf()
    build_syntf()
    print(f"Wrote notebooks to {NOTEBOOKS}")


if __name__ == "__main__":
    main()
