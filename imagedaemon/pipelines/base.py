from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Sequence

from imagedaemon.processing import astrometry, calibration
from imagedaemon.processing.calibration import CalibrationError
from imagedaemon.utils.image import Image  # thin wrapper around FITS/WCS

log = logging.getLogger("imagedaemon.pipeline")


class BasePipelines:
    """
    Camera‑agnostic orchestration of calibration + astrometry.

    Sub‑classes can override helper methods (_load_dark, _crop_sensor, …)
    but rarely need to re‑implement the public API.
    """

    # ---------- constructor --------------------------------------------------
    def __init__(self, meta):
        self.meta = meta  # e.g. WinterMeta PixelScale=1.12 …

    # ======================================================================
    #   PUBLIC METHODS  (called by the daemon or CLI)
    # ======================================================================
    def calibrate_image(
        self,
        *,
        addr: str | None = None,
        science_image: str | Path,
        background_image_list: Sequence[str] | None = None,
        override_steps: dict[str, bool] | None = None,
        **astrometry_opts,
    ):
        """
        Calibrate *science_image* .

        Parameters
        ----------
        addr
            Sub‑sensor identifier ('pa', 'pb', …) — ignored by cameras
            that have only one sensor.
        science_image
            Path to the raw MEF or single‑ext FITS.
        background_image_list
            Optional list of frames used to build a dither/sky flat.
        override_steps
            Dict like {'lab_flat': False} to toggle individual steps.

        """
        log.info(
            "Calibration started | camera=%s image=%s", self.meta.name, science_image
        )

        steps = self._decide_steps(override_steps)

        # -- 1. read science frame -----------------------------------------
        img = self._load_raw_image(science_image, addr=addr)

        exptime = self._get_exptime(img)

        log.debug("Loaded image %s  exptime=%.2fs", science_image, exptime)

        data = img.data

        if steps["mask"]:
            data = calibration.apply_mask(data, img.mask)

        """if steps["mask_hotpixels"]:
            data = calibration.mask_hot_pixels(data, img.mask)"""

        # -- 2. run requested calibration steps ----------------------------
        if steps["dark"]:
            log.debug("Applying master dark")
            data = calibration.subtract_dark(
                data, self._load_dark(exptime, addr=addr).data
            )

        if steps["lab_flat"]:
            log.debug("Applying lab flat")
            data = calibration.flat_correct(img, self._load_lab_flat(addr=addr).data)

        if steps["dither_flat"]:
            log.debug("Applying dither flat")
            data = calibration.flat_correct(
                data,
                self._make_dither_flat(
                    background_image_list or [], exptime=exptime, addr=addr
                ).data,
            )

        if steps["sky_flat"]:
            log.debug("Buiding and applying sky flat")
            data = calibration.flat_correct(
                data,
                self._make_sky_flat(background_image_list or [], addr=addr).data,
            )

        if steps["mask_hot_pixels"]:
            log.debug(f"Masking hot pixels > {self._hot_pixel_threshold()}")
            data = calibration.mask_hot_pixels(
                data, threshold=self._hot_pixel_threshold()
            )

        if steps["remove_horizontal_stripes"]:
            log.debug("Removing horizontal stripes")
            data = calibration.remove_horizontal_stripes(data, img.mask)

        if steps["replace_nans_with_median"]:
            log.debug("Replacing NaNs with median")
            data = calibration.replace_nans_with_median(data)

        # for now just return the calibrated Image
        return Image(data, img.header)

    def get_astrometric_solution(
        self,
        *,
        addr: str | None = None,
        science_image: str | Path,
        background_image_list: Sequence[str] | None = None,
        override_steps: dict[str, bool] | None = None,
        output_dir: Path | str | None = None,  # None | path | "tmp"
        **astrometry_opts,
    ):
        """
        Calibrate *science_image* (optional steps) and run solve‑field.

        Parameters
        ----------
        addr, science_image, background_image_list, override_steps
            Same meaning as in `calibrate_image`.
        output_dir
            • None  → run solve‑field where the calibrated FITS is written
            • Path  → copy the calibrated FITS there and keep artefacts
            • "tmp" → run in a TemporaryDirectory that is deleted afterwards
        astrometry_opts
            Extra hints passed to `run_astrometry` (scale bounds, RA/Dec, …).

        Returns
        -------
        astropy.wcs.WCS
        """
        # ------------------------------------------------------------------
        # 1. calibrate the raw frame ---------------------------------------
        calibrated = self.calibrate_image(
            addr=addr,
            science_image=science_image,
            background_image_list=background_image_list,
            override_steps=override_steps,
        )

        raw_path = Path(science_image)

        # ------------------------------------------------------------------
        # 2. choose where to write IMG.cal.fits ----------------------------
        if output_dir is None:
            work_dir = raw_path.parent
            cleanup_ctx = nullcontext()

        elif str(output_dir).lower() == "tmp":
            tmpobj = TemporaryDirectory()
            work_dir = Path(tmpobj.name)
            cleanup_ctx = tmpobj  # auto‑delete later

        else:
            work_dir = Path(output_dir).expanduser()
            work_dir.mkdir(parents=True, exist_ok=True)
            cleanup_ctx = nullcontext()

        cal_path = work_dir / (raw_path.stem + ".cal.fits")
        calibrated.save_image(cal_path)
        log.debug("Calibrated FITS written to %s", cal_path)

        # ------------------------------------------------------------------
        # 3. pixel‑scale bounds -------------------------------------------
        scale_low, scale_high = self._scale_bounds(astrometry_opts)
        astrometry_opts.setdefault("scale_low", scale_low)
        astrometry_opts.setdefault("scale_high", scale_high)

        # ------------------------------------------------------------------
        # 4. RA/Dec guess --------------------------------------------------
        ra = astrometry_opts.pop("ra", None) or calibrated.header.get("RADEG")
        dec = astrometry_opts.pop("dec", None) or calibrated.header.get("DECDEG")
        if ra is None or dec is None:
            raise ValueError("Supply ra=<deg>, dec=<deg> or keep them in the header.")

        # ------------------------------------------------------------------
        # 5. run solve‑field in the same work dir --------------------------
        with cleanup_ctx:  # deletes tmp dir if needed
            wcs = astrometry.run_astrometry(
                cal_path,
                ra=ra,
                dec=dec,
                output_dir=None,  # already in the right place
                **astrometry_opts,
            )
            return wcs

    # ======================================================================
    #   HELPER METHODS  (meant to be overridden by camera subclasses)
    # ======================================================================
    # ----- calibration assets ---------------------------------------------
    def _get_exptime(self, img: Image) -> float:
        """
        Return the exposure time of the image.

        Default: read from the header.
        """
        return img.header["EXPTIME"]

    def _load_dark(self, img: Image):  # noqa: D401
        """Return the master‑dark Image that matches *img* (override)."""
        raise CalibrationError("Dark frames not supported for this camera")

    def _load_lab_flat(self) -> Image:
        raise CalibrationError("Lab flats not supported for this camera")

    def _make_dither_flat(self, frames: Iterable[str]) -> Image:
        raise CalibrationError("Dither flat not supported for this camera")

    def _make_sky_flat(self, frames: Iterable[str]) -> Image:
        raise CalibrationError("Sky flat not supported for this camera")

    # ----- raw image loading ---------------------------------------------
    def _load_raw_image(self, path: str | Path, *, addr: str | None) -> Image:
        """Default: just read the FITS; no sensor selection."""
        return Image(path)

    # ----- internal ------------------------------------------------------
    def _scale_bounds(self, overrides: dict) -> tuple[float, float]:
        """
        Derive (scale_low, scale_high) in arcsec/px.

        Priority:
        1. Caller supplied 'scale_low' / 'scale_high' in astrometry_opts.
        2. Camera meta has absolute values (scale_low/high).
        3. Camera meta has pixel_scale ± scale_margin.
        """
        # caller overrides win
        if "scale_low" in overrides and "scale_high" in overrides:
            return overrides["scale_low"], overrides["scale_high"]

        meta = self.meta
        if hasattr(meta, "scale_low") and hasattr(meta, "scale_high"):
            return meta.scale_low, meta.scale_high

        margin = getattr(meta, "scale_margin", 0.05)  # default ±5 %
        low = meta.pixel_scale * (1 - margin)
        high = meta.pixel_scale * (1 + margin)
        return low, high

    def _hot_pixel_threshold(self) -> float:
        """
        Return the threshold for hot pixel masking.
        """
        return getattr(self.meta, "hot_pixel_threshold", None)

    def _decide_steps(self, override_steps: dict[str, bool] | None):
        steps = self.meta.cal_steps.copy()
        if override_steps:
            steps.update(override_steps)
        return steps
