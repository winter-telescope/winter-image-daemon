import numpy as np
from scipy.optimize import curve_fit


# helper functions
def parabola(x, x0, A, B):
    return A + B * (x - x0) ** 2


def fit_parabola(x, y, yerr):
    p0 = [np.mean(x), np.min(y), np.std(y)]
    popt, pcov = curve_fit(parabola, x, y, p0=p0, sigma=yerr, absolute_sigma=True)
    return popt  # [x0_best, A, B]


def run_focus_loop(image_list, *, pipeline, **opts):
    """
    Generic driver that calls pipeline‑specific helpers and returns a dict:
      {best_focus, focus_positions, fwhm_median, fwhm_std, plot}
    """
    # 1. load in the data
    # in nearly all cases this is just a list of Image objects,
    # for WINTER it will be a dictionary of addr to lists of Image objects
    images = pipeline.load_images_for_focus(image_list, **opts)

    # 2. quick calibration
    calibrated_images = pipeline.calibrate_for_focus(
        images, opts.get("output_dir"), **opts
    )

    # 3. measure FWHM
    med_fwhm, std_fwhm, focus_vals = pipeline.measure_fwhm(calibrated_images, **opts)

    # 4. fit & plot
    best_focus, figpath = pipeline.fit_focus_curve(
        focus_vals, med_fwhm, std_fwhm, opts.get("output_dir")
    )

    return {
        "best_focus": float(best_focus),
        "focus_positions": focus_vals.tolist(),
        "fwhm_median": med_fwhm.tolist(),
        "fwhm_std": std_fwhm.tolist(),
        "plot": str(figpath),
    }
