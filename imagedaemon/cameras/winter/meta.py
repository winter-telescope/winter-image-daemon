from imagedaemon.meta.base import BaseMeta


class WinterMeta(BaseMeta):
    name: str = "winter"
    pixel_scale: float = 1.12
    scale_margin: float = 0.05
    sensors: tuple[str, ...] = ("pa", "pb", "pc", "sa", "sb", "sc")
    hot_pixel_threshold: int = 60000

    # calibration steps
    cal_steps: dict[str, bool] = {
        "dark": True,
        "lab_flat": False,
        "dither_flat": False,
        "sky_flat": False,
        "remove_horizontal_stripes": True,
        "mask": True,
        "mask_hot_pixels": True,
        "replace_nans_with_median": True,
    }

    # focus steps
    focus_addrs: list[str] = ["pa", "pb", "pc", "sa", "sb", "sc"]
    focus_cal_steps: dict[str, bool] = {
        "dark": True,
        "lab_flat": False,
        "dither_flat": True,
        "sky_flat": False,
        "remove_horizontal_stripes": True,
        "mask": True,
        "mask_hot_pixels": True,
        "replace_nans_with_median": True,
    }
