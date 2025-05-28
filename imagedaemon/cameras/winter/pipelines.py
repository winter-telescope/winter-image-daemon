from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from imagedaemon.cameras.winter.winter_image import WinterImage
from imagedaemon.pipelines.base import BasePipelines
from imagedaemon.processing import calibration
from imagedaemon.processing.focus import fit_parabola, parabola
from imagedaemon.utils.image import Image
from imagedaemon.utils.notify import SlackNotifier
from imagedaemon.utils.paths import CAL_DATA_DIR
from imagedaemon.utils.serialization import sanitize_for_serialization

# ----------------------------------------------------------------------
# camera‑specific logger
# ----------------------------------------------------------------------
log = logging.getLogger("imagedaemon.camera.winter")


class WinterPipelines(BasePipelines):
    """
    Sub‑sensor‑aware adapter that re‑uses BasePipelines but overrides
    the few helper methods Winter needs.
    """

    # focus method

    # ----- raw image loader with subsensor support -------------------
    def _load_raw_image(self, path: str | Path, *, addr: str | None) -> Image:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[Winter] raw image not found: {path}")
        win_img = WinterImage(path)

        if not addr:
            raise ValueError(
                "WinterPipelines requires a sub‑sensor addr ('pa', 'pb', …')."
            )

        img = win_img.get_sensor_image(addr)
        log.debug("Loaded raw sensor %s from %s", addr, path.name)
        return img

    def _load_focus_images(
        self,
        image_list: list[str | Path | WinterImage],
    ) -> dict[str, list[Image]]:
        """
        Turn a list of MEF paths (or WinterImage instances) into a dict
        addr -> list of single‐sensor Image instances.
        """
        # first wrap any raw paths
        winter_imgs = [
            img if isinstance(img, WinterImage) else WinterImage(img)
            for img in image_list
        ]
        if not winter_imgs:
            raise ValueError("No images to calibrate")
        if len(winter_imgs) < 4:
            raise ValueError("Need at least 4 images to fit a best focus parabola")

        image_dict: dict[str, list[Image]] = defaultdict(list)

        # for each MEF, pull out every sensor that actually exists
        for wimg in winter_imgs:
            for addr in WinterImage.get_addrs():
                try:
                    sensor_img = wimg.get_sensor_image(addr)
                except KeyError:
                    # that sub-image wasn’t in this file, skip it
                    continue
                image_dict[addr].append(sensor_img)

        # you might want to cast back to a normal dict
        return dict(image_dict)

    # ----- focus analysis and plotting helpers ----------------------------
    def _analyze_and_plot_fwhm_results(
        self,
        fwhm_dict: dict,
        outdir: Path | str,
        post_plot_to_slack: bool = False,
    ) -> dict:
        """
        Analyze the FWHM results and plot them.
        Returns a dictionary with the results and paths to the plots.
        """

        # ------------------------------------------------------------------
        # fit each sensor separately  +  plot
        # ------------------------------------------------------------------

        focus_positions = np.asarray(fwhm_dict["focus_positions"])
        sensors_dict = fwhm_dict["sensors"]  # {'pa': {...}, 'pb': {...}, …}

        fig, axes = plt.subplots(
            nrows=3,
            ncols=2,
            figsize=(12, 12),
        )

        # Matplotlib quirk: when nrows == 1, axes is a single Axes object
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        per_sensor = {}

        # channel mapping

        # ---------- per‑sensor fits ----------
        # iterate over all winter sensors
        all_addrs = WinterImage.get_addrs()
        for i, addr in enumerate(all_addrs):
            rowcol = WinterImage._rowcol_locs_by_addr[addr]
            row, col = rowcol
            ax = axes[row, col]

            vals = sensors_dict.get(addr, None)
            if vals is None:
                log.warning("No FWHM data for sensor %s, skipping", addr)
                continue
            med = np.asarray(vals["fwhm_median"])
            std = np.asarray(vals["fwhm_std"])
            foc = focus_positions

            popt = fit_parabola(foc, med, std)  # [best_focus, a, b] etc.
            xgrid = np.linspace(foc.min(), foc.max(), 200)

            ax.errorbar(foc, med, yerr=std, fmt="o", label=f"{addr}")
            ax.plot(xgrid, parabola(xgrid, *popt), label=f"best = {popt[0]:.1f}")
            ax.set_ylabel("FWHM [arcsec]")
            ax.legend(frameon=False, loc="upper center", ncols=2)
            ax.set_xlabel("Focus Position [micron]")

            per_sensor[addr] = {
                "best_focus": float(popt[0]),
                "focus": foc.tolist(),
                "fwhm_median": med.tolist(),
                "fwhm_std": std.tolist(),
            }
        plt.tight_layout()
        # save the per-sensor results
        per_sensor_plot_path = Path(outdir) / "focusloop_per_sensor.png"
        plt.suptitle("Focus loop – per sensor best focus")
        plt.savefig(per_sensor_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ------------------------------------------------------------------
        # calculate and plot median focus results
        # ------------------------------------------------------------------

        focus_positions = np.asarray(fwhm_dict["focus_positions"])
        fig, ax_global = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(12, 12),
            sharex=True,
        )

        global_med = np.asarray(fwhm_dict["median"]["fwhm_median"])
        global_std = np.asarray(fwhm_dict["median"]["fwhm_std"])

        p_global = fit_parabola(focus_positions, global_med, global_std)

        xgrid = np.linspace(focus_positions.min(), focus_positions.max(), 200)

        ax_global.errorbar(
            focus_positions, global_med, yerr=global_std, fmt="o", label="all sensors"
        )
        ax_global.plot(
            xgrid, parabola(xgrid, *p_global), label=f"global best = {p_global[0]:.1f}"
        )
        ax_global.set_ylabel("FWHM [arcsec]")
        ax_global.set_xlabel("Focus Position [micron]")
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
                    per_sensor_plot_path,
                    text="Best focus per sensor",
                )
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

    # ----- calibration assets ---------------------------------------
    def _load_dark(self, exptime: float, addr: str | None) -> Image:
        dark_path = Path(self.meta.dark_dir) / f"winter_masterdark_{exptime:.3f}s.fits"
        if not dark_path.exists():
            raise FileNotFoundError(f"[Winter] master‑dark missing: {dark_path}")

        if not addr:
            raise ValueError("Must supply addr for Winter dark")

        img = WinterImage(dark_path).get_sensor_image(addr)
        log.debug("Loaded master‑dark  exp=%.3fs  addr=%s", exptime, addr)
        return img

    def _load_lab_flat(self, addr: str | None) -> Image:
        if not self.meta.lab_flat_file.exists():
            raise FileNotFoundError(
                f"[Winter] lab flat missing: {self.meta.lab_flat_file}"
            )

        if not addr:
            raise ValueError("Must supply addr for Winter lab flat")

        img = WinterImage(self.meta.lab_flat_file).get_sensor_image(addr)
        log.debug("Loaded lab flat addr=%s", addr)
        return img

    # ----- make (dither) flat ----------------------------------------------
    """
    def _make_dither_flat(
        self,
        frames: List[str | Path],
        exptime: float,
        addr: str | None,
    ) -> Image:
        if not frames:
            raise ValueError("No frames provided for dither‑flat construction")
        if not addr:
            raise ValueError("Must supply addr for Winter dither flat")

        win_imgs = [WinterImage(f).get_sensor_image(addr) for f in frames]
        bkg_data = [w.data for w in win_imgs]
        mask = win_imgs[0].mask
        dark = self._load_dark(exptime, addr=addr).data

        flat_data = calibration.build_dither_flat(bkg_data, dark, mask=mask)
        log.info("Built dither flat from %d frames  addr=%s", len(frames), addr)

        # Wrap ndarray in Image (empty header is fine for flats)
        return Image(flat_data, header={})
        """

    def _get_default_mask(self, img: Image, addr: str | None) -> np.ndarray:

        # get the default mask for this image
        mask = WinterImage.get_raw_winter_mask(
            img.data, WinterImage._board_id_by_addr[addr]
        )

        return mask

    def _make_flat(
        self,
        image_list: list[Image] | list[str | Path],
        exptime: float,
        addr: str | None,
    ) -> Image:
        if not image_list:
            raise ValueError("No frames provided for flat construction")

        # handle some different cases:
        match image_list:
            case [Image(), *_]:
                images = image_list

            case [str() | Path(), *_]:
                # addr is required
                if not addr:
                    raise ValueError("Must supply addr for a list of paths")
                images = [
                    WinterImage(path).get_sensor_image(addr) for path in image_list
                ]

            case [WinterImage(), *_]:
                # addr is required
                if not addr:
                    raise ValueError("Must supply addr a list of WinterImages")
                images = [img.get_sensor_image(addr) for img in image_list]

            case _:
                raise ValueError(
                    "Invalid image list: Must contain Image or WinterImage objects or paths."
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
