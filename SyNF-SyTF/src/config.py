"""Repository path helpers — works from notebooks or scripts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
EQL_DIR = REPO_ROOT / "external" / "eql"


def ensure_results_dir(name: str) -> Path:
    """Create and return a subdirectory under results/."""
    path = RESULTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path
