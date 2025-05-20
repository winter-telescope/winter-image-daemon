import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust depth to reach repo root
ENV_FILE = PROJECT_ROOT / ".env"

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


CONFIG_DIR = get_path("CONFIG_DIR", Path(PROJECT_ROOT, "imagedaemon", "config"))
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

astrom_scamp = CONFIG_DIR.joinpath("scamp.conf")
astrom_sex = CONFIG_DIR.joinpath("astrom.sex")
astrom_param = CONFIG_DIR.joinpath("astrom.param")
astrom_filter = CONFIG_DIR.joinpath("default.conv")
astrom_swarp = CONFIG_DIR.joinpath("config.swarp")
astrom_nnw = CONFIG_DIR.joinpath("default.nnw")
photom_sex = CONFIG_DIR.joinpath("photomCat.sex")

OUTPUT_DIR = get_path(
    "OUTPUT_DIR", os.path.join(os.getenv("HOME"), "data", "image-daemon-data", "output")
)

FOCUS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "focus")

print(f"astro_scamp = {astrom_scamp}")
print(f"masterbias_dir = {MASTERBIAS_DIR}")
