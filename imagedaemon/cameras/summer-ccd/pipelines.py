# pipeline overrides for the summer ccd camera

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from astropy.coordinates import Angle

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

    def _load_dark(self, exptime: float, addr=None) -> Image:  # noqa: D401
        """Return the master‑dark Image that matches *img* (override)."""
        # eventually this needs to try to find the dark and if it can't,
        # then it should make one from the bias and the closest known dark
        # for now just return
        raise NotImplementedError("Summer CCD darks are not implemented yet. ")

    def _get_exptime(self, img: Image) -> float:
        """
        Get the exposure time from the image header.
        """
        exptime_key = "EXPTIME"
        exptime = img.header.get(exptime_key)
        if exptime is None:
            raise ValueError(f"{exptime_key} not found in header of {img.filename}")
        return exptime

    def _get_radeg_from_header(self, header) -> float:
        """
        Get the RA from the header.
        """
        # for summer-ccd, the header entry is:
        # RA      = '322:13:37.11101905' / Requested right ascension (deg:m:s)
        angle = Angle(header["RA"], unit="deg")
        radeg = angle.deg
        return radeg

    def _get_decdeg_from_header(self, header) -> float:
        """
        Get the Dec from the header.
        """
        # DEC     = '47:19:53.02733028'  / Requested declination (deg:m:s)
        angle = Angle(header["DEC"], unit="deg")
        decdeg = angle.deg

        return decdeg
