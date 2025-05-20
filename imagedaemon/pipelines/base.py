from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Sequence

import numpy as np

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
            addr=addr,
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
    #   PUBLIC – batch calibration for focus loops
    # ======================================================================
    def calibrate_for_focus(
        self,
        image_paths: list[str | Path],
        out_dir: str | Path | None = None,
        **opts,
    ) -> list["Image"]:
        """
        Calibrate a *list* of raw `Image` objects for a focus loop.

        The default logic is ideal for single‑detector cameras:

        1.  Build a **single** dither flat from *all* frames
            (if those steps are enabled).
        2.  Apply the same master dark, lab flat, dither flat … to every frame.
        3.  Return the calibrated `Image` objects in the same order.

        Multi‑detector cameras (e.g. WINTER) override this method to
        group frames by sensor address and call
        `_calibrate_data()` separately per sensor.
        """
        # make a list of Image objects
        images = [
            Image(path) if isinstance(path, str) else path for path in image_paths
        ]
        if not images:
            raise ValueError("No images to calibrate")
        if len(images) < 4:
            raise ValueError("Need at least 4 images to fit a best focus parabola")
        steps = self._decide_steps(None)
        exptime = self._get_exptime(images[0])

        # ---------- build once‑per‑dataset reference dither flat --------------
        precomputed: dict[str, np.ndarray] = {}
        if steps["dither_flat"]:
            precomputed["dither_flat"] = self._make_dither_flat(
                [im.path for im in images],
                exptime=exptime,
                addr="",
            ).data

        # ---------- calibrate every frame -------------------------------
        calibrated: list["Image"] = []
        for im in images:
            data_cal = self._calibrate_data(
                im.data.copy(),
                mask=im.mask,
                header=im.header,
                addr="",  # single‑sensor camera
                exptime=exptime,
                steps=steps,
                precomputed=precomputed,
            )
            calibrated.append(Image(data_cal, im.header, mask=im.mask))

        return calibrated

    def measure_fwhm(
        self,
        image_list: Sequence[str | Path],  # list of raw image file names or paths
        focus_positions: Sequence[float] | None = None,  # list of focus positions
        addrs: (
            Sequence[str] | None
        ) = None,  # list of sensor addresses to use in the focus calculation
        override_steps: dict[str, bool] | None = None,
        focus_position_header_key: str = "FOCPOS",
        **kwargs,
    ) -> dict:
        """
        Measure the FWHM of the images in *image_list*.
        Optionally, the focus positions can be provided to override
        the FWHM with the focus position, but it will otherwise
        try to find the focus_position_header_key in the header and just use that.

        for winter cameras, the addrs parameter can be used to specify
        which sensors to use in the focus calculation. If None, just assumes one sensor.

        for winter, returns something like:
            {
            "pa": (focus_vals, fwhm_median, fwhm_std),
            "pb": (...),
                ...
            }

        for other cameras, returns something like:
            {"": (focus_vals, med, std)}

        """
        return {}

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
        filter_name: str | None = None,
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
        log.debug("Applying calibration steps: %s", steps)
        if addr is not None:
            log.debug("Sensor address: %s", addr)
        if steps["mask"]:
            log.debug("Applying mask")
            data = calibration.apply_mask(data, mask)

        if steps["dark"]:
            log.debug("Applying dark subtraction")
            data = calibration.subtract_dark(data, self._load_dark(exptime, addr).data)

        if steps["lab_flat"]:
            log.debug("pulling lab flat")
            data = calibration.flat_correct(data, self._load_lab_flat(addr).data)

        if steps["dither_flat"]:
            log.debug("building dither flat")
            flat = (
                precomputed.get("dither_flat")
                or self._make_flat(bkg_images or [], exptime=exptime, addr=addr).data
            )
            data = calibration.flat_correct(data, flat)

        if steps["sky_flat"]:
            # log that we are here

            # Ensure sky_flat is available or load it (filter_name required)
            flat = precomputed.get("sky_flat")

            if flat is None:
                if filter_name is None:
                    raise ValueError(
                        "`filter_name` must be provided when no precomputed sky flat is available."
                    )
                log.debug("pulling sky flat for filter %s", filter_name)
                flat = self._get_sky_flat(filter_name=filter_name, addr=addr).data
            else:
                log.debug("Using precomputed sky flat")
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

    def _load_dark(self, exptime: float):  # noqa: D401
        """Return the master‑dark Image that matches *img* (override)."""
        raise CalibrationError("Dark frames not supported for this camera")

    def _load_lab_flat(self) -> Image:
        raise CalibrationError("Lab flats not supported for this camera")

    def _get_sky_flat(self, filter_name: str, addr: str) -> Image:
        """
        Return the sky flat Image that matches *img* (override).
        """
        raise CalibrationError("Sky flats not supported for this camera")

    def _make_flat(
        self,
        image_list: list[Image] | list[str | Path],
        exptime: float,
        addr: str | None,
    ) -> Image:
        """
        takes in a list of Image objects and creates a Image object,
        or a list of paths to images. If it's a list of paths it will
        load the images into a list of Image objects. Then it takes
        the median of the images. Useful for dither flats,
        and making flats out of sky and lab images.

        It will median combine and then do a dark subtraction,
        and then normalize the flat to 1.0.
        """

        # load the images
        match image_list:
            case [Image(), *_]:
                images = image_list

            case [str() | Path(), *_]:
                images = [Image(path) for path in image_list]

            case _:
                raise ValueError(
                    "Invalid image list: Must contain Image objects or paths."
                )

        # now we have a list of Image objects

        # median combine the images
        data_list = [im.data for im in images]
        median_data = calibration.median_combine(data_list)

        # subtract the dark
        dark_img = self._load_dark(exptime, addr)
        median_dark_sub_data = calibration.subtract_dark(median_data, dark_img.data)

        # normalize the flat to 1.0
        flat_data = calibration.normalize(median_dark_sub_data)

        # create a new Image object with the flat data
        flat_img = Image(flat_data, header=images[0].header, mask=images[0].mask)

        return flat_img

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

    def _decide_steps(self, override: dict[str, bool] | None, *, kind="cal"):
        default = self.meta.focus_cal_steps if kind == "focus" else self.meta.cal_steps
        steps = default.copy()
        if override:
            steps.update(override)
        return steps
