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
        **_,
    ) -> "Image":
        """
        Apply the camera’s default calibration chain to *science_image*.

        Parameters
        ----------
        addr
            Sub‑sensor address for multi‑detector cameras (WINTER).
            Ignored by single‑detector cameras.
        science_image
            Raw MEF or single‑extension FITS file.
        background_image_list
            Frames used to build a dither/sky flat (if that step is enabled).
        override_steps
            Dict that toggles individual calibration steps
            (e.g. ``{'dither_flat': False}``).
        """
        log.info(
            "Calibration started | camera=%s image=%s",
            self.meta.name,
            science_image,
        )

        # 1. decide which steps to perform
        steps = self._decide_steps(override_steps)

        # 2. load raw frame (may already select sensor if addr is given)
        img = self._load_raw_image(science_image, addr=addr)
        exptime = self._get_exptime(img)
        log.debug("Loaded image   EXPTIME = %.2fs", exptime)

        # 3. call the unified calibrator
        data_cal = self._calibrate_data(
            img.data.copy(),
            mask=img.mask,
            header=img.header,
            addr=addr or "",
            exptime=exptime,
            steps=steps,
            bkg_images=background_image_list,
        )

        # 4. return new Image instance
        return Image(data_cal, img.header, mask=img.mask)

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
            info = astrometry.run_astrometry(
                cal_path,
                ra=ra,
                dec=dec,
                output_dir=None,  # already in the right place
                **astrometry_opts,
            )
            return info

    # ======================================================================
    #   HELPER METHODS  (meant to be overridden by camera subclasses)
    # ======================================================================
    # ----- calibration assets ---------------------------------------------

    def _calibrate_data(
        self,
        data,
        *,
        mask,
        header,
        addr,
        exptime,
        steps,
        bkg_images=None,  # list[str]|None
        precomputed=None,  # optional dict of ready‑made flats
    ):
        """
        Parameters
        ----------
        data, mask, header   numpy array + mask + FITS header
        addr                 sensor address or '' for 1‑sensor cams
        exptime              exposure time (sec)
        steps                dict from _decide_steps()
        bkg_images           focus‑loop list to build dither/sky flats
        precomputed          {'dither_flat': ndarray, 'sky_flat': ndarray, …}
        """
        precomputed = precomputed or {}

        if steps["mask"]:
            data = calibration.apply_mask(data, mask)

        if steps["dark"]:
            data = calibration.subtract_dark(data, self._load_dark(exptime, addr).data)

        if steps["lab_flat"]:
            data = calibration.flat_correct(data, self._load_lab_flat(addr).data)

        if steps["dither_flat"]:
            flat = (
                precomputed.get("dither_flat")
                or self._make_dither_flat(
                    bkg_images or [], exptime=exptime, addr=addr
                ).data
            )
            data = calibration.flat_correct(data, flat)

        if steps["sky_flat"]:
            flat = (
                precomputed.get("sky_flat")
                or self._make_sky_flat(bkg_images or [], addr=addr).data
            )
            data = calibration.flat_correct(data, flat)

        if steps["mask_hot_pixels"]:
            data = calibration.mask_hot_pixels(
                data, threshold=self._hot_pixel_threshold()
            )

        if steps["remove_horizontal_stripes"]:
            data = calibration.remove_horizontal_stripes(data, mask)

        if steps["replace_nans_with_median"]:
            data = calibration.replace_nans_with_median(data)

        return data

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
