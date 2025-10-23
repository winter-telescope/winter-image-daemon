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


import os
from pathlib import Path
from typing import Union


def normalize_filepath(filepath: Union[str, Path], expanduser: bool = True) -> Path:
    """
    Normalize a filepath by converting all slashes to forward slashes and optionally
    expanding the user home directory.

    Handles mixed Windows/Linux paths like: ~\data\images\20251022\spring/file.fits

    Parameters
    ----------
    filepath : str or Path
        The filepath to normalize. Can contain mixed forward/backslashes.
    expanduser : bool, optional
        If True, expand ~ to the user's home directory. Default is True.

    Returns
    -------
    Path
        A normalized Path object with consistent separators.

    Examples
    --------
    >>> normalize_filepath(r"~\data\images\20251022\spring/file.fits")
    PosixPath('/home/winter/data/images/20251022/spring/file.fits')

    >>> normalize_filepath(r"~\data\images\file.fits", expanduser=False)
    PosixPath('~/data/images/file.fits')
    """
    # Convert to string if Path object
    if isinstance(filepath, Path):
        filepath = str(filepath)

    # Replace all backslashes with forward slashes
    normalized = filepath.replace("\\", "/")

    # Convert to Path object
    path = Path(normalized)

    # Expand user home directory if requested
    if expanduser:
        path = path.expanduser()

    return path


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
    "DATA_DIR", os.path.join(os.path.expanduser("~"), "data", "image-daemon-data")
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
    "OUTPUT_DIR",
    os.path.join(os.path.expanduser("~"), "data", "image-daemon-data", "output"),
)

FOCUS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "focus")
