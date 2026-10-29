"""Project paths independent of the shell or notebook working directory."""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Locate the source checkout, or use DATA_PROJECT_ROOT for wheel installs."""
    override = os.environ.get("DATA_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("DATA_PROJECT_ROOT must point to an existing directory")
        return root
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "src").is_dir():
        return root
    raise RuntimeError(
        "Cannot locate the project checkout. Set DATA_PROJECT_ROOT to its path "
        "when using a non-editable installation."
    )


PROJECT_ROOT = get_project_root()
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def ensure_project_dirs() -> None:
    """Create directories used for generated project artifacts."""
    for directory in (INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
