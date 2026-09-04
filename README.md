# Symbolic Machine Learning for Interpretable Forecasting of Chaotic Time Series

Official repository for the paper:

**Turning Time Series into Algebraic Equations: Symbolic Machine Learning for Interpretable Modeling of Chaotic Time Series**

This repository contains the code and datasets for implementing the symbolic forecasting frameworks Symbolic Neural Forecaster (SyNF) and Symbolic Tree Forecaster (SyTF), designed for interpretable forecasting of chaotic time series.

![Symbolic_Forecasting_Models](https://github.com/mad-stat/Symbolic_Forecasting/blob/main/Model_Image/Symbolic_Forecasting_Model_Architecturre.png)

Symbolic Neural Forecaster (SyNF) – learns explicit algebraic forecasting equations using a differentiable neural architecture.

Symbolic Tree Forecaster (SyTF) – discovers symbolic forecasting equations using evolutionary symbolic regression.

Both models produce interpretable mathematical equations while maintaining strong forecasting accuracy.

## Repository layout

```
SyNF-SyTF/
├── data/                    # Input datasets
├── external/eql/            # Minimal dependency (Equation Learner)
├── notebooks/               # SyNF.ipynb and SyTF.ipynb
├── results/                 # Generated equations and predictions (gitignored)
├── scripts/                 # Utility scripts
└── src/                     # Shared Python modules
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**SyTF** also requires [Julia](https://julialang.org/) with SymbolicRegression.jl. After installing Julia, run once inside Python:

```python
import pysr
pysr.install()
```

## Data

| File | Used by | Notes |
|------|---------|-------|
| `data/Sanjuan_data_weekly.csv` | SyNF | Weekly dengue cases (San Juan) |
| `data/RW_Simulated_datset.csv` | SyTF | Simulated random walk (200 rows) |

If `RW_Simulated_datset.csv` is missing, regenerate it with:

```bash
python scripts/generate_rw_simulated.py
```

The bundled file is a synthetic random walk (seed 42, starting at 0). For **exact** reproduction of paper results, replace it with your original `RW_Simulated_datset.csv` if you still have it.

## Running the notebooks

Open and run from the repository root so paths resolve correctly:

- `notebooks/SyNF.ipynb` — EQL training on San Juan weekly cases (lag 4, last 52 weeks test)
- `notebooks/SyTF.ipynb` — PySR symbolic regression on simulated random walk (lag 1, 180 train / 20 test)

Outputs (equations, predictions, logs) are written under `results/`.

## Methods (summary)

### SyTF
- **Library:** [Link](https://github.com/MilesCranmer/PySR)
- **Operators:** `+`, `-`, `*`, `sin`, `cos` (no division or exp)
- **Selection:** `model_selection="best"`, 25 iterations

### SyNF
- **Library:** [Link](https://github.com/lkmklsmn/EQL-Pytorch) (vendored under `external/eql/`)
- **Units:** `id`, `mul`, `cos`, `sin`, `div`
- **Training:** 1000 epochs, L1 reg `1e-4`, mask threshold `0.05`


## Citation

If you use this code, please cite the associated paper 

@article{panja2026turning,  
  title={Turning Time Series into Algebraic Equations: Symbolic Machine Learning for Interpretable Modeling of Chaotic Time Series},  
  author={Panja, Madhurima and Younes, Grace and Hens, Chittaranjan and Chakraborty, Tanujit},  
  journal={arXiv preprint arXiv:2603.07261},  
  year={2026}  
}

