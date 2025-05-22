# qCMOS camera definition

from imagedaemon.meta.base import BaseMeta


class QcmosMeta(BaseMeta):
    name: str = "qcmos"
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
    # focus steps
    focus_addrs: list[str] = []  # just a winter thing
    focus_cal_steps: dict[str, bool] = {
        "dark": True,
        "lab_flat": False,
        "dither_flat": True,
        "sky_flat": False,
        "remove_horizontal_stripes": False,
        "mask": False,
        "mask_hot_pixels": False,
        "replace_nans_with_median": True,
    }
