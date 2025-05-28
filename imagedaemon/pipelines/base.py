from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Sequence

import numpy as np
from matplotlib import pyplot as plt

from imagedaemon.processing import astrometry, calibration
from imagedaemon.processing.calibration import CalibrationError
from imagedaemon.processing.focus import fit_parabola, parabola
from imagedaemon.processing.sextractor import get_img_fwhm
from imagedaemon.utils.image import Image  # thin wrapper around FITS/WCS
from imagedaemon.utils.notify import SlackNotifier
from imagedaemon.utils.serialization import sanitize_for_serialization

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
        ra = astrometry_opts.pop("ra", None) or self._get_radeg_from_header(
            calibrated.header
        )
        dec = astrometry_opts.pop("dec", None) or self._get_decdeg_from_header(
            calibrated.header
        )
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
    from pathlib import Path

    def calibrate_for_focus(
        self,
        image_paths: list[str | Path],
        addrs: list[str] | None = None,
        override_steps: dict[str, bool] | None = None,
        out_dir: str | Path | None = None,
        **opts,
    ) -> dict[str, list[Path]]:
        image_dict = self._load_focus_images(image_paths)
        if not image_dict:
            raise ValueError("No images to calibrate")

        # default to ALL sensors if the caller didn't down-select
        if addrs is None:
            addrs = list(image_dict.keys())

        steps = self._decide_steps(override_steps, kind="focus")

        # make our output directory, if requested
        out_path = Path(out_dir).expanduser() if out_dir else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        calibrated = {}

        # for each requested sensor
        for addr in addrs:
            sensor_imgs = image_dict[addr]  # [Image,Image,…]
            raw_for_this_sensor = image_paths  # same length & order

            # precompute any flats/darks …
            pre = {}
            if steps["dither_flat"]:
                pre["dither_flat"] = self._make_flat(
                    [p for p in raw_for_this_sensor],
                    exptime=self._get_exptime(sensor_imgs[0]),
                    addr=addr,
                ).data

                # save the dither flat to a tmp directory to verify it is working
                import os

                tmpdir = os.path.join(os.getenv("HOME"), "data", "tmp")
                dither_flat_image = Image(pre["dither_flat"])
                dither_flat_image.save_image(
                    os.path.join(tmpdir, f"{addr}_test_dither_flat.fits")
                )
            out_files: list[Path] = []

            # now calibrate each one
            for raw_path, im in zip(raw_for_this_sensor, sensor_imgs):
                data_cal = self._calibrate_data(
                    im.data.copy(),
                    mask=im.mask,
                    header=im.header,
                    addr=addr,
                    exptime=self._get_exptime(im),
                    steps=steps,
                    precomputed=pre,
                )
                cal_img = Image(data_cal, im.header, mask=im.mask)

                if out_path:
                    stem = Path(raw_path).stem  # e.g. "IMG00123"
                    fname = f"{stem}_{addr}.cal.fits"  # "IMG00123_pa.cal.fits"
                    cal_path = out_path / fname
                    cal_img.save_image(cal_path)
                    out_files.append(cal_path)
                else:
                    # if you wanted to keep them in-memory, you could skip saving
                    out_files.append(None)  # or store the Image itself

            calibrated[addr] = out_files

        return calibrated

    '''
    def measure_fwhm_of_calibrated_images(
        self,
        calibrated_images_dict: dict[str, list[str]],
        addrs: list[str] | None = None,
        override_steps: dict[str, bool] | None = None,
        out_dir: str | Path | None = None,
        focus_positions: list[float] | None = None,
        **kwargs,
    ) -> dict[str, list[str]]:
        """
        Take in the dictionary of calibrated images and measure the FWHM of each image,
        organized by sensor address, and return a dictionary of the per-sensor FWHM values,
        and the median across all specified sensors in the addrs list.
        """
        # instantiate a dictionary to hold the FWHM results
        fwhm_results_dict = {"median": {}, "sensors": {}}
        for sensor in calibrated_images_dict:
            image_paths = calibrated_images_dict[sensor]
            if not image_paths:
                raise ValueError(f"No images to calibrate for sensor {sensor}")

            # if no focus positions are provided, grab them from the images
            # this should only execute on the first sensor in the list,
            # which implicitly assumes that all sensors have the same focus positions.
            if focus_positions is None:
                focus_positions = [
                    self._get_focus_position(Image(path)) for path in image_paths
                ]
                # add the focus positions to the results dictionary
                fwhm_results_dict.update({"focus_positions": focus_positions})

            # Measure FWHM for each image in the list
            fwhm_results = self.measure_fwhm(
                image_paths=image_paths,
                override_steps=override_steps,
                out_dir=out_dir,
                **kwargs,
            )

            # Store the results in a dictionary
            fwhm_results_dict.update({sensor: fwhm_results})

        # median results across all SELECTED sensors at each focus position
        fwhm_median_combined = []
        fwhm_mean_combined = []
        fwhm_std_combined = []
        for addr in fwhm_results_dict:
            # add an entry to the results dict:
            if (addr in addrs) or (addrs is None):
                fwhm_median_combined.append(
                    fwhm_results_dict["sensors"][addr]["fwhm_median"]
                )
                fwhm_mean_combined.append(
                    fwhm_results_dict["sensors"][addr]["fwhm_mean"]
                )
                fwhm_std_combined.append(fwhm_results_dict["sensors"][addr]["fwhm_std"])

        # add the median results to the dictionary
        fwhm_median_combined = np.array(fwhm_median_combined)
        fwhm_mean_combined = np.array(fwhm_mean_combined)
        fwhm_std_combined = np.array(fwhm_std_combined)

        fwhm_median_medians = np.nanmedian(fwhm_median_combined, axis=0)
        fwhm_mean_medians = np.nanmedian(fwhm_mean_combined, axis=0)
        fwhm_std_medians = np.nanmedian(fwhm_std_combined, axis=0)

        # add the median results to the dictionary
        fwhm_results_dict.update(
            {
                "median": {
                    "fwhm_median": fwhm_median_medians,
                    "fwhm_mean": fwhm_mean_medians,
                    "fwhm_std": fwhm_std_medians,
                }
            }
        )

        # clean up the results dictionary so that it only contains native python types and
        # that the results are lists not np arrays. this is important for pyro serialization
        fwhm_results_dict = sanitize_for_serialization(fwhm_results_dict)
        return fwhm_results_dict
        '''

    def measure_fwhm_of_calibrated_images(
        self,
        calibrated_images_dict: dict[str, list[str]],
        addrs: list[str] | None = None,
        override_steps: dict[str, bool] | None = None,
        out_dir: str | Path | None = None,
        focus_positions: list[float] | None = None,
        **kwargs,
    ) -> dict:
        """
        Take in a dict mapping sensor → [calibrated FITS paths], measure FWHM,
        and return
        {
            "focus_positions": [ … ],
            "sensors": {
            "sa": {"fwhm_median":[…],"fwhm_mean":[…],"fwhm_std":[…]},
            "sb": {…}, …
            },
            "median": {
            "fwhm_median":[…],  # median across SELECTED sensors
            "fwhm_mean":[…],
            "fwhm_std":[…],
            }
        }

        If `addrs` is provided, only those sensors go into the `"median"` block
        (but *all* sensors are still reported under `"sensors"`).  In the
        single-detector case it will strip out the `"sensors"` block entirely.
        """

        results: dict = {}
        # 1) determine focus positions (once)
        if focus_positions is None:
            # pick the first non-empty sensor to read FOCPOS
            for sensor, paths in calibrated_images_dict.items():
                if paths:
                    focus_positions = [
                        self._get_focus_position(Image(p)) for p in paths
                    ]
                    break
            else:
                raise ValueError("No calibrated images provided")
        results["focus_positions"] = list(focus_positions)

        # 2) per‐sensor FWHM
        sensors_block: dict[str, dict] = {}
        for sensor, paths in calibrated_images_dict.items():
            if not paths:
                raise ValueError(f"No calibrated images for sensor {sensor!r}")
            fwhm_res = self.measure_fwhm(
                image_paths=paths,
                override_steps=override_steps,
                out_dir=out_dir,
                addr=sensor,
                **kwargs,
            )
            # fwhm_res is {"fwhm_median":[…],"fwhm_mean":[…],"fwhm_std":[…]}
            sensors_block[sensor] = fwhm_res

        # if single‐detector (only one key, typically "") then we omit the block
        if len(sensors_block) == 1 and list(sensors_block.keys())[0] in ("", None):
            # collapse to just the median
            single = sensors_block[list(sensors_block.keys())[0]]
            results["median"] = {
                "fwhm_median": single["fwhm_median"],
                "fwhm_mean": single["fwhm_mean"],
                "fwhm_std": single["fwhm_std"],
            }
            results["sensors"] = {}
            return sanitize_for_serialization(results)

        # otherwise keep all sensors
        results["sensors"] = sensors_block

        # 3) build combined median across the selected subset
        if addrs is None:
            selected = list(sensors_block.keys())
        else:
            # filter out any that aren’t present
            selected = [s for s in addrs if s in sensors_block]
            if not selected:
                raise ValueError(f"None of the requested sensors {addrs} present")

        # stack into arrays, shape = (n_sel, n_focus)
        med_stack = np.vstack([sensors_block[s]["fwhm_median"] for s in selected])
        mean_stack = np.vstack([sensors_block[s]["fwhm_mean"] for s in selected])
        std_stack = np.vstack([sensors_block[s]["fwhm_std"] for s in selected])

        results["median"] = {
            "fwhm_median": np.nanmedian(med_stack, axis=0).tolist(),
            "fwhm_mean": np.nanmedian(mean_stack, axis=0).tolist(),
            "fwhm_std": np.nanmedian(std_stack, axis=0).tolist(),
        }

        return sanitize_for_serialization(results)

    def measure_fwhm(
        self,
        image_paths: (
            Sequence[str | Path] | str | Path
        ),  # list of raw image file names or paths
        focus_positions: Sequence[float] | None = None,  # list of focus positions
        override_steps: dict[str, bool] | None = None,
        focus_position_header_key: str = "FOCPOS",
        addr: str | None = None,
        **kwargs,
    ) -> dict:
        """
        Measure the FWHM of the images in *image_list*.

        expects a list of images filenames/path objects which point to images
        which are already calibrated. It will then run sextractor on the images
        and return a dictionary:

        results = {"fwhm_median" : fwhm_median_list, "fwhm_mean": fwhm_mean_list, "fwhm_std" : fwhm_std_list}
        """
        # if it's a single image, make it a list
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]

        fwhm_mean = []
        fwhm_median = []
        fwhm_std = []

        # step 0: loop through the images
        for image_path in image_paths:

            # step 1 make sure there is a weight image for the sensor, and make it if not
            if True:  # not self._weight_image_exists(addr):
                # make the weight image
                img = Image(image_path)
                mask = self._get_default_mask(img, addr)
                img.mask = mask

                weight_image_path = img.save_weight_image(
                    filename=self.meta.weight_file_path(addr)
                )
                log.debug("Weight image generated for sensor %s", addr)
            else:
                log.debug(
                    f"Weight image already exists for sensor {addr}: {self.meta.weight_file_path(addr)}"
                )

            # step 2: run sextractor on the saved calibrated images, or just pull
            # the results from existing tables
            mean, median, std = get_img_fwhm(
                image_path,
                pixscale=self.meta.pixel_scale,
                weightimg=self.meta.weight_file_path(addr),
                xlolim=10,
                xuplim=2000,
                ylolim=10,
                yuplim=2000,
                exclude=False,
                sex_config=self.meta.sextractor_sex_file,
                sex_param=self.meta.sextractor_param_file,
                sex_filter=self.meta.sextractor_filter_file,
                sex_nnw=self.meta.sextractor_nnw_file,
            )
            # step 3: add the results to the lists
            fwhm_mean.append(mean)
            fwhm_median.append(median)
            fwhm_std.append(std)

        # step 4: return a dictionary with the fwhm and fwhm_std lists
        results = {
            "fwhm_median": fwhm_median,
            "fwhm_mean": fwhm_mean,
            "fwhm_std": fwhm_std,
        }

        return results

    # focus loop orchestrator
    # ------------ orchestrator --------------------------------------------

    def run_focus_loop(
        self,
        image_list,
        *,
        addrs=None,
        output_dir=None,
        post_plot_to_slack=False,
        **opts,
    ):
        """
        Generic driver – *pipeline* is a {Winter,Qcmos,…}Pipelines instance.

        Parameters
        ----------
        image_list   list[str]       raw images belonging to one focus sweep
        addrs        list[str]|None  subset of sensors (WINTER) or None
        output_dir   str|Path|None   where to drop plots & intermediates
        """
        outdir = Path(output_dir or self.meta.focus_output_dir).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # 1. load & quick‑calibrate
        # ------------------------------------------------------------------
        calibrated_images_dict = self.calibrate_for_focus(
            image_list, out_dir=outdir, **opts
        )

        # ------------------------------------------------------------------
        # 2. measure FWHM   ->  dict {addr: (focus, med, std)}
        # ------------------------------------------------------------------
        fwhm_dict = self.measure_fwhm_of_calibrated_images(
            calibrated_images_dict, addrs=addrs, **opts
        )
        """
        Example:
        fwhm_dict = {
                    "median": {
                        "fwhm_median": fwhm_median_medians,
                        "fwhm_mean": fwhm_mean_medians,
                        "fwhm_std": fwhm_std_medians,
                    },
                    "focus_positions": focus_positions,
                    "sensors": {
                        "pa": {
                            "fwhm_median": fwhm_median_pa,
                            "fwhm_mean": fwhm_mean_pa,
                            "fwhm_std": fwhm_std_pa,
                        },
                        ...
                    },
                }
        """
        # ------------------------------------------------------------------
        # 3. fit each sensor separately  +  plot
        # ------------------------------------------------------------------
        focus_positions = np.asarray(fwhm_dict["focus_positions"])
        sensors_dict = fwhm_dict["sensors"]  # {'pa': {...}, 'pb': {...}, …}

        n_panels = len(sensors_dict) + 1  # +1 for the global view
        fig, axes = plt.subplots(
            nrows=n_panels,
            ncols=1,
            figsize=(6, 3 * n_panels),
            sharex=True,
        )

        # Matplotlib quirk: when nrows == 1, axes is a single Axes object
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        per_sensor = {}

        # ---------- per‑sensor fits ----------
        for ax, (addr, vals) in zip(axes[:-1], sensors_dict.items()):
            med = np.asarray(vals["fwhm_median"])
            std = np.asarray(vals["fwhm_std"])
            foc = focus_positions

            popt = fit_parabola(foc, med, std)  # [best_focus, a, b] etc.
            xgrid = np.linspace(foc.min(), foc.max(), 200)

            ax.errorbar(foc, med, yerr=std, fmt="o", label=f"{addr}")
            ax.plot(xgrid, parabola(xgrid, *popt), label=f"best = {popt[0]:.1f}")
            ax.set_ylabel("FWHM [arcsec]")
            ax.legend(frameon=False, loc="upper center", ncols=2)

            per_sensor[addr] = {
                "best_focus": float(popt[0]),
                "focus": foc.tolist(),
                "fwhm_median": med.tolist(),
                "fwhm_std": std.tolist(),
            }

        # ------------------------------------------------------------------
        # 4. global (stacked) fit  +  plot
        # ------------------------------------------------------------------
        global_med = np.asarray(fwhm_dict["median"]["fwhm_median"])
        global_std = np.asarray(fwhm_dict["median"]["fwhm_std"])

        p_global = fit_parabola(focus_positions, global_med, global_std)

        ax_global = axes[-1]  # last panel
        xgrid = np.linspace(focus_positions.min(), focus_positions.max(), 200)

        ax_global.errorbar(
            focus_positions, global_med, yerr=global_std, fmt="o", label="all sensors"
        )
        ax_global.plot(
            xgrid, parabola(xgrid, *p_global), label=f"global best = {p_global[0]:.1f}"
        )
        ax_global.set_ylabel("FWHM [arcsec]")
        ax_global.set_xlabel("FOCPOS")
        ax_global.legend(frameon=False, loc="upper center", ncols=2)

        plt.suptitle(f"Focus loop – global best = {p_global[0]:.1f}")

        plot_path = outdir / "focusloop_all_sensors.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ------------------------------------------------------------------

        results = {
            "best_focus": float(p_global[0]),
            "per_sensor": per_sensor,
            "plot": str(plot_path),
            "results": fwhm_dict,
        }

        if post_plot_to_slack:
            try:
                notifier = SlackNotifier()
                notifier.post_image(
                    plot_path,
                    text=f"Ran the focus script and got best focus = {results['best_focus']:.1f}",
                )
            except Exception as e:
                log.error("Failed to post focus loop results to Slack: %s", e)
        # ------------------------------------------------------------------

        # clean up the results dictionary so that it only contains native python types and
        # that the results are lists not np arrays. this is important for pyro serialization
        results = sanitize_for_serialization(results)
        return results

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

        def stats(stage, arr):
            nfin = np.isfinite(arr).sum()
            print(
                f"[DEBUG][{addr}] {stage}: finite={nfin}/{arr.size}, min={np.nanmin(arr):.3g}, max={np.nanmax(arr):.3g}"
            )

        print(f"running _calibrate_data with addr={addr}")
        stats("raw", data)
        precomputed = precomputed or {}
        log.debug("Applying calibration steps: %s", steps)
        if addr is not None:
            log.debug("Sensor address: %s", addr)
        if steps["mask"]:
            log.debug("Applying mask")
            data = calibration.apply_mask(data, mask)
            stats("after mask", data)

        if steps["dark"]:
            log.debug("Applying dark subtraction")
            data = calibration.subtract_dark(data, self._load_dark(exptime, addr).data)
            stats("after dark", data)
        if steps["lab_flat"]:
            log.debug("pulling lab flat")
            data = calibration.flat_correct(data, self._load_lab_flat(addr).data)
            stats("after lab flat", data)
        if steps["dither_flat"]:
            log.debug("building dither flat")
            try:
                flat = precomputed["dither_flat"]
            except KeyError:
                flat = self._make_flat(
                    bkg_images or [], exptime=exptime, addr=addr
                ).data

            data = calibration.flat_correct(data, flat)
            stats("after dither flat", data)
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
            stats("after sky flat", data)

        if steps["mask_hot_pixels"]:
            data = calibration.mask_hot_pixels(
                data, threshold=self._hot_pixel_threshold()
            )
            stats("after hot pixel mask", data)
        if steps["remove_horizontal_stripes"]:
            data = calibration.remove_horizontal_stripes(data, mask)
            stats("after horizontal stripes", data)

        if steps["replace_nans_with_median"]:
            data = calibration.replace_nans_with_median(data)
            stats("after NaN replacement", data)

        if steps["replace_nans_with_local_median"]:
            data = calibration.replace_nans_with_local_median(data)
            stats("after NaN replacement", data)

        return data

    def _weight_image_exists(self, addr: str | None) -> bool:
        """
        Check if a weight image exists for the given sensor address.

        Default: no weight image.
        """
        # the weight image path is stored in the camera meta
        weight_image_path = self.meta.weight_file_path(addr)
        if Path(weight_image_path).exists():
            return True
        else:
            log.warning("No weight image found for sensor %s", addr)
            # if no weight image is found, return False
            # this will be overridden in the camera subclass
        return False

    def _get_default_mask(self, img: Image, addr: str | None) -> np.ndarray:
        """
        Return the default mask for the given sensor address.

        Default: no mask.
        """
        # default mask is none
        mask = np.zeros_like(img.data, dtype=bool)

        return mask

    def _get_focus_position(self, img: Image) -> float:
        """
        Return the focus position of the image.

        Default: read from the header.
        """
        return img.header["FOCPOS"]

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
        try:
            dark_img = self._load_dark(exptime, addr)
            dark_data = dark_img.data
        except Exception as e:
            # if we can't load a dark, just use a zero array
            dark_data = np.zeros_like(median_data)
            log.warning(
                "No dark frame available for %s, using zero array instead", addr
            )
        median_dark_sub_data = calibration.subtract_dark(median_data, dark_data)

        # normalize the flat to 1.0
        flat_data = calibration.normalize(median_dark_sub_data)

        # create a new Image object with the flat data
        flat_img = Image(flat_data, header=images[0].header, mask=images[0].mask)

        return flat_img

    # ----- astrometry helpers ---------------------------------------------
    def _get_radeg_from_header(self, header) -> float:
        """
        Get the RA from the header.
        """
        return header["RADEG"]

    def _get_decdeg_from_header(self, header) -> float:
        """
        Get the Dec from the header.
        """
        return header["DECDEG"]

    # ----- raw image loading ---------------------------------------------
    def _load_raw_image(
        self, path: str | Path, *, addr: list[str] | str | None
    ) -> Image:
        """Default: just read the FITS; no sensor selection."""
        return Image(path)

    def _load_focus_images(
        self,
        image_list: list[str | Path | Image],
    ) -> dict:  # make a list of Image objects
        images = [Image(path) if isinstance(path, str) else path for path in image_list]
        if not images:
            raise ValueError("No images to calibrate")
        if len(images) < 4:
            raise ValueError("Need at least 4 images to fit a best focus parabola")
        image_dict = {"": images}
        return image_dict

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
