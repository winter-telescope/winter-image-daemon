# imagedaemon/meta/base.py

from pathlib import Path
from typing import Dict, Optional, Tuple

from pydantic import BaseModel

from imagedaemon.utils.paths import CAL_DATA_DIR, CONFIG_DIR, FOCUS_OUTPUT_DIR


class BaseMeta(BaseModel):
    """
    Shared camera metadata:
      - name
      - pixel_scale
      - which calibration steps to run (for science & focus)
      - and a bunch of standard data‐dirs/files derived from `name`
    """

    name: str
    pixel_scale: float

    # which steps to run on a science image
    cal_steps: Dict[str, bool]
    # which steps to run in a focus loop
    focus_cal_steps: Dict[str, bool]

    """ Source Extractor configuration files:
    # where are the source extractor config files?
    Each camera may override *sex_cfg_dir* if it ships its own set of
    astrom.sex / .param / default.conv / default.nnw files.
    """

    # --- SExtractor configuration ---------------------------------------
    sex_cfg_dir: Path = CONFIG_DIR  # <── global default

    @property
    def sextractor_sex_file(self) -> Path:
        return self.sex_cfg_dir / "astrom.sex"

    @property
    def sextractor_param_file(self) -> Path:
        return self.sex_cfg_dir / "astrom.param"

    @property
    def sextractor_filter_file(self) -> Path:
        return self.sex_cfg_dir / "default.conv"

    @property
    def sextractor_nnw_file(self) -> Path:
        return self.sex_cfg_dir / "default.nnw"

    class Config:
        frozen = True  # immutable once constructed

    # ── master calibration dirs ──────────────────────────────────────

    @property
    def dark_dir(self) -> Path:
        """where to find your per‐exptime master darks"""
        return Path(CAL_DATA_DIR, self.name, "masterdarks")

    @property
    def lab_flat_file(self) -> Path:
        """the *one* laboratory flat to apply"""
        return Path(
            CAL_DATA_DIR, self.name, "masterflats", f"{self.name}_masterflat.fits"
        )

    @property
    def weight_dir(self) -> Path:
        """where to find your per‐sensor or single weight maps"""
        return Path(CAL_DATA_DIR, self.name, "weight")

    @property
    def focus_output_dir(self) -> Path:
        """where to drop focus plots & intermediates"""
        return Path(FOCUS_OUTPUT_DIR, self.name)

    def weight_file_path(self, addr: Optional[str] = None) -> Path:
        """
        Path to the weight image:

          • single‐sensor cams → <name>_weight.fits
          • multi‐sensor cams  → <name>_<addr>_weight.fits
        """
        stem = f"{self.name}_{addr}_weight" if addr else f"{self.name}_weight"
        return self.weight_dir / f"{stem}.fits"
