# pipeline overrides for the summer ccd camera

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from imagedaemon.pipelines.base import BasePipelines
from imagedaemon.processing import calibration
from imagedaemon.utils.image import Image
from imagedaemon.utils.paths import CAL_DATA_DIR

# ----------------------------------------------------------------------
# camera‑specific logger
# ----------------------------------------------------------------------
log = logging.getLogger("imagedaemon.camera.summer-ccd")


class SummerCCDPipelines(BasePipelines):
    """
    Pipelines for analyzing summer ccd images.
    """

    # ----- raw image loader -------------------
    def _load_raw_image(self, path: str | Path, *, addr: str | None) -> Image:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[summer-ccd] raw image not found: {path}")
        img = Image(path)
        log.debug("Loaded raw image %s from %s", addr, path.name)
        return img

    def _get_exptime(self, img: Image) -> float:
        """
        Get the exposure time from the image header.
        """
        exptime_key = "EXPOSURE TIME"
        exptime = img.header.get(exptime_key)
        if exptime is None:
            raise ValueError(f"{exptime_key} not found in header of {img.path}")
        return exptime
