import os
import subprocess
import sys
import warnings
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from imagedaemon.utils.ldactools import (
    get_table_from_ldac,
)
from imagedaemon.utils.paths import (
    SEXTRACTOR_CONV_FILE,
    SEXTRACTOR_NNW_FILE,
    SEXTRACTOR_PARAM_FILE,
    SEXTRACTOR_SEX_FILE,
)

"""
sex_config_path = os.path.join(wsp_path, "focuser", "sex_config")
astrom_scamp = os.path.join(sex_config_path, "scamp.conf")
astrom_sex = os.path.join(sex_config_path, "astrom.sex")
astrom_param = os.path.join(sex_config_path, "astrom.param")
astrom_filter = os.path.join(sex_config_path, "default.conv")
astrom_swarp = os.path.join(sex_config_path, "/onfig.swarp")
astrom_nnw = os.path.join(sex_config_path, "default.nnw")
photom_sex = os.path.join(sex_config_path, "photomCat.sex")"""


def run_sextractor(
    imgname: str | Path,
    pixscale: float,
    weightimg: str | Path,
    regions: bool = True,
    sex_config: str | Path = SEXTRACTOR_SEX_FILE,
    sex_param: str | Path = SEXTRACTOR_PARAM_FILE,
    sex_filter: str | Path = SEXTRACTOR_CONV_FILE,
    sex_nnw: str | Path = SEXTRACTOR_NNW_FILE,
):
    # Run sextractor on the proc image file

    # normalize all file inputs by converting to strings
    imgname = str(imgname)
    weightimg = str(weightimg)
    sex_config = str(sex_config)
    sex_param = str(sex_param)
    sex_filter = str(sex_filter)
    sex_nnw = str(sex_nnw)

    print("Running SExtractor on %s" % (imgname))

    try:
        command = (
            "sex -c "
            + sex_config
            + " "
            + imgname
            + " "
            + "-CATALOG_NAME "
            + imgname
            + ".cat"
            + " -CATALOG_TYPE FITS_LDAC "
            + "-PARAMETERS_NAME "
            + sex_param
            + " "
            + "-FILTER_NAME "
            + sex_filter
            + " "
            + "-STARNNW_NAME "
            + sex_nnw
            + " "
            + "-WEIGHT_TYPE NONE -CHECKIMAGE_TYPE NONE -PIXEL_SCALE "
            + str(pixscale)
            + " -DETECT_THRESH 10 -ANALYSIS_THRESH 10 -SATUR_LEVEL 60000 -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE "
            + weightimg
            + " -CHECKIMAGE_NAME "
            + imgname
            + ".seg,"
            + imgname
            + ".bkg,"
            + imgname
            + ".bkg.rms"
        )
        print("Executing command : %s" % (command))
        rval = subprocess.run(command.split(), check=True, capture_output=True)
        print("Process completed")
        print(rval.stdout.decode())

    except subprocess.CalledProcessError as err:
        print("Could not run sextractor with error %s." % (err))
        return

    if regions:
        print(f"")
        t = get_table_from_ldac(imgname + ".cat")
        with open(imgname + ".cat" + ".stats.reg", "w") as f:
            f.write("image\n")
            for row in t:
                f.write(
                    "CIRCLE(%s,%s,%s) # text={%.2f,%.2f}\n"
                    % (
                        row["X_IMAGE"],
                        row["Y_IMAGE"],
                        row["FWHM_IMAGE"] / 2,
                        row["FWHM_IMAGE"],
                        row["SNR_WIN"],
                    )
                )


def get_img_fluxdiameter(
    imgname: str | Path,
    pixscale: float,
    weightimg: str | Path,
    xlolim=10,
    xuplim=2000,
    ylolim=10,
    yuplim=2000,
    exclude=False,
):
    if not os.path.exists(imgname + ".cat"):
        run_sextractor(imgname, pixscale, weightimg=weightimg)

    img_cat = get_table_from_ldac(imgname + ".cat")
    print("Found %s sources" % (len(img_cat)))

    center_mask = (
        (img_cat["X_IMAGE"] < xuplim)
        & (img_cat["X_IMAGE"] > xlolim)
        & (img_cat["Y_IMAGE"] < yuplim)
        & (img_cat["Y_IMAGE"] > ylolim)
    )
    if exclude:
        center_mask = np.invert(center_mask)
    n_sources = len(img_cat[center_mask])
    print("Using %s sources" % (n_sources))
    mean, median, std = sigma_clipped_stats(2 * img_cat[center_mask]["FLUX_RADIUS"])
    stderr_mean = std / (n_sources) ** 0.5
    stderr_med = (np.pi / 2) ** 0.5 * stderr_mean
    return mean, median, std, stderr_mean, stderr_med


def get_img_fwhm(
    imgname: str | Path,
    pixscale: float,
    weightimg: str | Path,
    xlolim: int = 10,
    xuplim: int = 2000,
    ylolim: int = 10,
    yuplim: int = 2000,
    exclude: bool = False,
) -> tuple[float, float, float]:
    """
    Run SExtractor on `imgname` (if needed), load the catalog, and return
    sigma-clipped FWHM stats for sources within the core region.

    Parameters
    ----------
    imgname
        Path to the image (fits) file.
    pixscale
        Pixel scale in arcsec/pix, passed to SExtractor.
    weightimg
        Path to the weight map for SExtractor.
    xlolim, xuplim, ylolim, yuplim
        Bounding box (in pixels) around the center to include sources.
    exclude
        If True, invert that box (i.e. mask the core and use the edges).

    Returns
    -------
    mean, median, std of the FWHM_IMAGE column.
    """
    # normalize inputs
    imgpath = Path(imgname)
    weightpath = Path(weightimg)

    # where SExtractor will write the catalog
    catpath = Path(str(imgpath) + ".cat")

    # run SExtractor if the catalog doesn't exist yet
    if not catpath.exists():
        print(f"Running SExtractor on {imgpath.name}...")
        run_sextractor(
            str(imgpath),
            pixscale,
            weightimg=str(weightpath),
        )

    # read the LDAC catalog
    img_cat = get_table_from_ldac(str(catpath))
    print(f"Found {len(img_cat)} sources in {catpath.name}")

    # build the mask of “central” sources
    center_mask = (
        (img_cat["X_IMAGE"] < xuplim)
        & (img_cat["X_IMAGE"] > xlolim)
        & (img_cat["Y_IMAGE"] < yuplim)
        & (img_cat["Y_IMAGE"] > ylolim)
    )
    if exclude:
        center_mask = ~center_mask

    n_used = np.count_nonzero(center_mask)
    print(f"Using {n_used} sources for FWHM stats")

    # compute clipped stats on just the FWHM_IMAGE column
    data = img_cat["FWHM_IMAGE"][center_mask]
    mean, median, std = sigma_clipped_stats(data, sigma=2.0)

    # convert to arcsec from pixels
    # (SExtractor returns FWHM in pixels, not arcsec)
    mean *= pixscale
    median *= pixscale
    std *= pixscale

    return mean, median, std


def gen_map(
    imgname: str | Path,
    pixscale: float,
    weightimg: str | Path,
    regions=False,
):
    npix = 2000
    x_inds = np.linspace(0, npix, 5)
    y_inds = np.linspace(0, npix, 5)

    if not os.path.exists(imgname + ".cat"):
        run_sextractor(imgname)
    t = get_table_from_ldac(imgname + ".cat")

    fwhms = np.zeros((4, 4))
    elongs = np.zeros((4, 4))
    for i in range(len(x_inds) - 1):
        for j in range(len(y_inds) - 1):
            cut_t = (
                (x_inds[i] < t["X_IMAGE"])
                & (t["X_IMAGE"] < x_inds[i + 1])
                & (y_inds[j] < t["Y_IMAGE"])
                & (t["Y_IMAGE"] < y_inds[j + 1])
            )
            mean, median, std = sigma_clipped_stats(t[cut_t]["FWHM_IMAGE"])
            fwhms[j][i] = median
            mean, median, std = sigma_clipped_stats(t[cut_t]["ELONGATION"])
            elongs[j][i] = median

    fig = plt.figure()
    gs = GridSpec(1, 2, wspace=0.5)
    ax1 = fig.add_subplot(gs[0])
    fmap = ax1.imshow(
        fwhms * pixscale, origin="lower", extent=(0, npix, 0, npix), vmin=1.8, vmax=3.5
    )
    ax1.set_xticks(x_inds)
    ax1.set_yticks(y_inds)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(fmap, cax=cax)
    ax1.set_title("FWHM")

    ax2 = fig.add_subplot(gs[1])
    emap = ax2.imshow(1 - 1 / elongs, origin="lower", extent=(0, npix, 0, npix))
    ax2.set_xticks(x_inds)
    ax2.set_yticks(y_inds)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax2.set_title("Ellipticity")
    fig.colorbar(emap, cax=cax)

    plt.savefig(
        "%s_fwhm_ellipticity_maps.pdf" % (imgname.split("/")[-1]), bbox_inches="tight"
    )


def gen_weight_img(imgname):
    img = fits.open(imgname)
    data = img[0].data
    img.close()
    weight_img = np.ones(data.shape)
    weight_img[np.isnan(data)] = 0
    # weight_img[2048,:] = 0	 #Better way to flag this out
    weight_hdu = fits.PrimaryHDU(weight_img)
    weight_hdu.writeto(imgname + ".weight", overwrite=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--imgname", type=str, help="Imagename")
    parser.add_argument("--pixscale", type=float, default=0.47, help="Pixelscale")
    parser.add_argument("--plot", action="store_true", help="Plot?")
    parser.add_argument(
        "--weight", type=str, default="weight.fits", help="weight image"
    )

    args = parser.parse_args()

    mean_fwhm, med_fwhm, std = get_img_fwhm(
        args.imgname, pixscale=args.pixscale, weightimg=args.weight
    )
    print(
        "Mean, Median, Std : %.2f,%.2f,%.2f"
        % (mean_fwhm * args.pixscale, med_fwhm * args.pixscale, std * args.pixscale)
    )

    if args.plot:
        gen_map(args.imgname, pixscale=args.pixscale)
