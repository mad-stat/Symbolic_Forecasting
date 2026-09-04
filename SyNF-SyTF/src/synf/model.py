"""PyTorch Lightning wrapper for the Equation Learner (SyNF)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytorch_lightning as pl
import sympy as sy
import torch
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, TensorDataset

from src.config import EQL_DIR, ensure_results_dir


def _import_eql():
    """Add vendored EQL-Pytorch modules to the import path."""
    eql_path = str(EQL_DIR)
    if eql_path not in sys.path:
        sys.path.insert(0, eql_path)
    from eqlearner import EQL  # noqa: WPS433

    return EQL


class LitEQL(pl.LightningModule):
    """Train EQL with L1 regularization and hard-concrete masking."""

    def __init__(
        self,
        n_lags: int,
        units,
        reg: float,
        mask_thresh: float,
        total_epochs: int,
    ):
        super().__init__()
        EQL = _import_eql()
        self.reg = reg
        self.model = EQL(n_lags, 1, units)
        self.total_epochs = total_epochs
        self.T1 = self.total_epochs // 4
        self.T2 = (self.total_epochs * 19) // 20
        self.mask_thresh = mask_thresh
        self.loss = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)

        if self.current_epoch == self.T2:
            self.model.set_masks(self.mask_thresh)

        mse_loss = F.mse_loss(y_pred, y)
        l1_loss = sum(param.abs().sum() for param in self.model.parameters())
        loss = mse_loss

        if self.T1 < self.current_epoch < self.T2:
            loss += self.reg * l1_loss

        if self.current_epoch > self.T2:
            self.model.apply_masks()

        self.log("mse_loss", mse_loss)
        self.log("l1_loss", l1_loss)
        self.loss = loss
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def train_synf(
    X_train,
    y_train,
    *,
    n_lags: int,
    units=None,
    reg: float = 1e-4,
    mask_thresh: float = 0.05,
    iterations: int = 1000,
    run_name: str = "synf_sanjuan_lag4",
) -> tuple[str, float]:
    """
    Fit SyNF (EQL) and save the discovered equation.

    Returns (equation_string, execution_time_seconds).
    """
    if units is None:
        units = ["id", "mul", "cos", "sin", "div"]

    out_dir = ensure_results_dir(run_name)
    inputs = torch.tensor(X_train.values, dtype=torch.float32)
    targets = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    eql = LitEQL(
        n_lags=n_lags,
        units=units,
        reg=reg,
        mask_thresh=mask_thresh,
        total_epochs=iterations,
    )
    tb_logger = pl_loggers.TensorBoardLogger(str(out_dir / "logs"))
    trainer = pl.Trainer(max_epochs=iterations, logger=tb_logger)

    start = time.time()
    trainer.fit(eql, train_loader)
    elapsed = time.time() - start

    func = str(sy.expand(eql.model.get_symbolic_expr()[0]))
    eq_path = out_dir / f"{run_name}_equation.txt"
    eq_path.write_text(func, encoding="utf-8")
    (out_dir / "execution_time.txt").write_text(
        f"Main function execution time: {elapsed:.2f} seconds\n",
        encoding="utf-8",
    )
    print(f"Equation saved to: {eq_path}")
    return func, elapsed
