from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from imagedaemon.cameras.winter.winter_image import WinterImage
from imagedaemon.pipelines.base import BasePipelines
from imagedaemon.processing import calibration
from imagedaemon.utils.image import Image
from imagedaemon.utils.paths import CAL_DATA_DIR

# ----------------------------------------------------------------------
# camera‑specific logger
# ----------------------------------------------------------------------
log = logging.getLogger("imagedaemon.camera.winter")


class WinterPipelines(BasePipelines):
    """
    Sub‑sensor‑aware adapter that re‑uses BasePipelines but overrides
    the few helper methods Winter needs.
    """

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

    # ----- dither flat ----------------------------------------------
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
