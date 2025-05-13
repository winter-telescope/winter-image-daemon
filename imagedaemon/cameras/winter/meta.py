from pathlib import Path

from pydantic import BaseModel

from imagedaemon.utils.paths import CAL_DATA_DIR


class WinterMeta(BaseModel):
    name: str = "winter"
    pixel_scale: float = 1.12
    scale_margin: float = 0.05
    sensors: tuple[str, ...] = ("pa", "pb", "pc", "sa", "sb", "sc")
    hot_pixel_threshold: int = 60000

    # calibration steps
    cal_steps: dict[str, bool] = {
        "dark": True,
        "lab_flat": False,
        "dither_flat": True,
        "sky_flat": False,
        "remove_horizontal_stripes": True,
        "mask": True,
        "mask_hot_pixels": True,
        "replace_nans_with_median": True,
    }

    # reference‑frame paths
    dark_dir: Path = Path(CAL_DATA_DIR, "winter", "masterdarks")
    lab_flat_file: Path = Path(CAL_DATA_DIR, "winter", "masterflats", "masterflat.fits")
