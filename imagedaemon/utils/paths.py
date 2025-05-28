import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust depth to reach repo root
GIT_REPO_ROOT = (
    PROJECT_ROOT.parent
)  # assuming this is one level up from the project root
ENV_FILE = GIT_REPO_ROOT / ".env"

# Load once when the module is imported
load_dotenv(dotenv_path=ENV_FILE, override=False)  # keeps any values already set


def get_path(name: str, default: str | Path | None = None) -> Path:
    """
    Return a pathlib.Path for the variable *name* defined in .env
    (or already present in the shell), falling back to *default*.

    Raises KeyError if the variable is missing and no default supplied.
    """
    value = os.getenv(name)
    if value is None:
        if default is None:
            raise KeyError(
                f"Environment variable '{name}' not found in {ENV_FILE} "
                "and no default provided."
            )
        value = default
    return Path(value).expanduser().resolve()


CONFIG_DIR = get_path("CONFIG_DIR", Path(PROJECT_ROOT, "config"))
print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"CONFIG_DIR = {CONFIG_DIR}")

DATA_DIR = get_path(
    "DATA_DIR", os.path.join(os.getenv("HOME"), "data", "image-daemon-data")
)
CAL_DATA_DIR = get_path("CAL_DATA_DIR", os.path.join(DATA_DIR, "calibration"))
RAW_DATA_DIR = get_path("RAW_DATA_DIR", os.path.join(DATA_DIR, "raw"))

MASTERBIAS_DIR = get_path("BIAS_DIR", os.path.join(CAL_DATA_DIR, "masterbias"))
MASTERFLAT_DIR = get_path("FLAT_DIR", os.path.join(CAL_DATA_DIR, "masterflats"))
MASTERDARK_DIR = get_path("DARK_DIR", os.path.join(CAL_DATA_DIR, "masterdarks"))

# Paths to the SExtractor configuration files
SEXTRACTOR_SEX_FILE = get_path("SEXTRACTOR_SEX_FILE", CONFIG_DIR.joinpath("astrom.sex"))
SEXTRACTOR_PARAM_FILE = get_path(
    "SEXTRACTOR_PARAM_FILE", CONFIG_DIR.joinpath("astrom.param")
)
SEXTRACTOR_CONV_FILE = get_path(
    "SEXTRACTOR_PARAM_FILE", CONFIG_DIR.joinpath("default.conv")
)
SEXTRACTOR_NNW_FILE = get_path(
    "SEXTRACTOR_PARAM_FILE", CONFIG_DIR.joinpath("default.nnw")
)

OUTPUT_DIR = get_path(
    "OUTPUT_DIR", os.path.join(os.getenv("HOME"), "data", "image-daemon-data", "output")
)

FOCUS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "focus")
