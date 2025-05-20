# qCMOS camera definition

from pathlib import Path

from pydantic import BaseModel

from imagedaemon.utils.paths import CAL_DATA_DIR


class SummerCCDMeta(BaseModel):
    name: str = "summer-ccd"
    pixel_scale: float = 0.157
    scale_margin: float = 0.05
    hot_pixel_threshold: int = 200

    # calibration steps
    cal_steps: dict[str, bool] = {
        "dark": False,
        "lab_flat": False,
        "dither_flat": False,
        "sky_flat": False,
        "remove_horizontal_stripes": False,
        "mask": False,
        "mask_hot_pixels": True,
        "replace_nans_with_median": True,
    }

    # reference‑frame paths
    dark_dir: Path = Path(CAL_DATA_DIR, "summer-ccd", "masterdarks")
    lab_flat_file: Path = Path(
        CAL_DATA_DIR, "summer-ccd", "masterflats", "masterflat.fits"
    )
