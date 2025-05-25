from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# ------------ generic math --------------------------------------------
def parabola(x, x0, A, B):
    return A + B * (x - x0) ** 2


def fit_parabola(x, y, yerr):
    p0 = [np.mean(x), np.min(y), np.std(y)]
    popt, _ = curve_fit(parabola, x, y, p0=p0, sigma=yerr, absolute_sigma=True)
    return popt  # [best_focus, A, B]


# ------------ orchestrator --------------------------------------------
def run_focus_loop(image_list, *, pipeline, addrs=None, output_dir=None, **opts):
    """
    Generic driver – *pipeline* is a {Winter,Qcmos,…}Pipelines instance.

    Parameters
    ----------
    image_list   list[str]       raw images belonging to one focus sweep
    addrs        list[str]|None  subset of sensors (WINTER) or None
    output_dir   str|Path|None   where to drop plots & intermediates
    """
    outdir = Path(output_dir or pipeline.meta.focus_output_dir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. load & quick‑calibrate
    # ------------------------------------------------------------------
    images = pipeline.load_images_for_focus(image_list, addrs=addrs, out_dir=outdir)
    calibrated_images_dict = pipeline.calibrate_for_focus(
        images, out_dir=outdir, **opts
    )

    # ------------------------------------------------------------------
    # 2. measure FWHM   ->  dict {addr: (focus, med, std)}
    # ------------------------------------------------------------------
    fwhm_dict = pipeline.measure_fwhm_of_calibrated_images(
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
    return {
        "best_focus": float(p_global[0]),
        "per_sensor": per_sensor,
        "plot": str(plot_path),
    }
