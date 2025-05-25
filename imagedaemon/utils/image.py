import os
from os import fspath
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, SqrtStretch, ZScaleInterval
from mpl_toolkits.axes_grid1 import make_axes_locatable

from imagedaemon.utils.mask import mask_datasec


class Image:
    "Class to hold an image and its header, and provide basic plotting/saving functions"

    def __init__(
        self,
        data: str | np.ndarray,  # can be a filename or a numpy array
        header: Optional[fits.Header] = None,
        mask: Optional[str | Path | np.ndarray] = None,
        verbose: bool = False,
    ) -> None:

        self.filename: Optional[str] = None
        self.filepath: Optional[str | Path] = None
        self.data: Optional[np.ndarray] = None
        self.mask: Optional[str | Path | np.ndarray] = None
        self.header: Optional[fits.Header] = None

        match data:
            # if data is a filename/filepath load the image data
            case str() | Path():
                self.load_image(data)
            case np.ndarray():
                self.load_data(data, header=header)
            case _:
                raise ValueError(
                    f"data must be a filename or a numpy array, not {type(data)}"
                )

        # load the mask
        match mask:
            case str() | Path():
                self.mask = self.load_mask_from_file(mask)
            case np.ndarray():
                self.mask = mask
            case None:
                self.mask = np.zeros_like(self.data, dtype=bool)

        # no matter what, let's mask the datasec region
        # datasec_mask = mask_datasec(self.data, self.header, fill_value=1)
        # self.mask = np.logical_or(self.mask, datasec_mask)

    def load_image(self, filepath: str | Path) -> None:
        """
        Load an image from a fits file
        :param filename:
        :return:
        """
        self.filename = os.path.basename(filepath)
        self.filepath = filepath

        # use astropy to load the fits image
        with fits.open(filepath) as hdu:
            self.header = hdu[0].header
            self.data = hdu[0].data

        # make the data a float32 array
        self.data = np.array(self.data, dtype=np.float32)

    def load_data(
        self, data: np.ndarray, header: Optional[fits.Header | dict] = None
    ) -> None:
        self.data = data
        self.header = header

    def plot_image(
        self,
        title: Optional[str] = None,
        cbar: bool = False,
        cmap: str = "gray",
        norm: Optional[str] = "zscale",
        apply_mask: bool = False,
        post_to_slack: bool = False,  # pylint: disable=unused-argument
        savepath: Optional[str] = None,
        single_image_width: Optional[float] = 3,
        **kwargs: Any,
    ) -> None:
        # use plot_image

        if apply_mask:
            # apply mask to the data
            mask = self.mask
        else:
            mask = None
        ax = plot_image(
            self.data,
            title=title,
            cbar=cbar,
            cmap=cmap,
            norm=norm,
            mask=mask,
            **kwargs,
        )
        return ax

    def save_image(self, filename: str | Path, overwrite: bool = True) -> None:
        """
        Write the image to a FITS file.

        Parameters
        ----------
        filename : str | Path
            Destination file path. If missing the ``.fits`` suffix it will
            be added automatically.
        overwrite : bool
            Forwarded to ``astropy.io.fits.writeto``.
        """
        # 1. normalise to Path object
        filename = Path(filename).with_suffix(".fits")

        # 2. ensure parent directory exists (skip if `.` / current dir)
        if filename.parent != Path():
            filename.parent.mkdir(parents=True, exist_ok=True)

        # 3. write the file
        hdu = fits.PrimaryHDU(self.data, header=self.header)
        hdu.writeto(fspath(filename), overwrite=overwrite)

    def save_mask_image(self, filename: str | Path, overwrite: bool = True) -> None:
        """
        Write the mask to a FITS file.

        Parameters
        ----------
        filename : str | Path
            Destination file path. If missing the ``.fits`` suffix it will
            be added automatically.
        overwrite : bool
            Forwarded to ``astropy.io.fits.writeto``.
        """
        # 1. normalise to Path object
        filename = Path(filename).with_suffix(".fits")

        # 2. ensure parent directory exists (skip if `.` / current dir)
        if filename.parent != Path():
            filename.parent.mkdir(parents=True, exist_ok=True)

        # 3. write the file
        # convert the mask to 1s and 0s instead of True and False
        mask_image_data = np.zeros_like(self.mask, dtype=np.uint8)
        mask_image_data[self.mask] = 1
        hdu = fits.PrimaryHDU(mask_image_data, header=self.header)
        hdu.writeto(fspath(filename), overwrite=overwrite)

    def save_weight_image(self, filename: str | Path, overwrite: bool = True) -> None:
        """
        Write the weight image to a FITS file.

        Parameters
        ----------
        filename : str | Path
            Destination file path. If missing the ``.fits`` suffix it will
            be added automatically.
        overwrite : bool
            Forwarded to ``astropy.io.fits.writeto``.
        """
        # 1. normalise to Path object
        filename = Path(filename).with_suffix(".fits")

        # 2. ensure parent directory exists (skip if `.` / current dir)
        if filename.parent != Path():
            filename.parent.mkdir(parents=True, exist_ok=True)

        # 3. write the file
        # convert the mask to 1s and 0s instead of True and False
        # note that this is the inverse of the mask
        # so that the weight image is 1 where the mask is 0
        mask_image_data = np.ones_like(self.mask, dtype=np.uint8)
        mask_image_data[self.mask] = 0
        hdu = fits.PrimaryHDU(mask_image_data, header=self.header)
        hdu.writeto(fspath(filename), overwrite=overwrite)


def write_image(image, filename, header=None, overwrite=True):
    """
    Write an image to a fits file
    :param image:
    :param filename:
    :param header:
    :param overwrite:
    :return:
    """
    # make parent directories if they don't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    hdu = fits.PrimaryHDU(image, header=header)
    hdu.writeto(filename, overwrite=overwrite)


def plot_image(
    data,
    thresh=3.0,
    ax=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    norm=None,
    return_norm=False,
    mask=None,
    title=None,
    cbar=False,
    figsize=(6, 6),
    **kwargs: Any,
):
    """
    Plot an image with plt.imshow, auto-thresholded via sigma_clipped_stats,
    and optionally restrict to a window defined by (x_min:x_max, y_min:y_max).

    Parameters
    ----------
    data : 2D np.ndarray
        The input image to plot.
    thresh : float, optional
        The sigma threshold to use in sigma_clipped_stats. Default is 3.0.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, a new figure and axes will be created.
    x_min : int, optional
        Minimum x index (column) for the window. If None, defaults to 0.
    x_max : int, optional
        Maximum x index (column) for the window (non-inclusive). If None, defaults to data.shape[1].
    y_min : int, optional
        Minimum y index (row) for the window. If None, defaults to 0.
    y_max : int, optional
        Maximum y index (row) for the window (non-inclusive). If None, defaults to data.shape[0].
    norm : matplotlib.colors.Normalize, optional
        A normalization object to pass to imshow. If None, defaults to a linear normalization.
    return_norm : bool, optional
        If True, return the normalization object. Default is False
    title : str, optional
        Title for the plot. Default is None.
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the image was plotted.
    norm : astropy.visualization.ImageNormalize, optional
        The normalization object used for the plot. Only returned if return_norm is True
    """
    # Create ax if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Handle default window boundaries
    if x_min is None:
        x_min = 0
    if x_max is None:
        x_max = data.shape[1]
    if y_min is None:
        y_min = 0
    if y_max is None:
        y_max = data.shape[0]

    # apply mask if provided: it must be the same shape as the data
    if mask is not None:
        if mask.shape != data.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match data shape {data.shape}"
            )
        # Apply the mask to the data
        data[mask] = np.nan

    # Slice the data to the desired window
    windowed_data = data[y_min:y_max, x_min:x_max]

    # Apply mask if provided

    # Compute stats on the windowed region
    _, med, std = sigma_clipped_stats(windowed_data, sigma=thresh)

    # Create a normalization object if not provided
    if norm is None:
        # Plot
        im = ax.imshow(
            windowed_data,
            vmin=med - 3 * std,
            vmax=med + 3 * std,
            origin="lower",
            **kwargs,
        )
    else:
        if norm == "zscale":
            norm = ImageNormalize(
                windowed_data,
                interval=ZScaleInterval(),
                stretch=SqrtStretch(),
            )
        elif norm == "minmax":
            norm = ImageNormalize(
                windowed_data,
                vmin=np.nanmin(windowed_data),
                vmax=np.nanmax(windowed_data),
            )
        # Plot
        im = ax.imshow(
            windowed_data,
            origin="lower",
            norm=norm,
            **kwargs,
        )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)  # Similar to fig.colorbar(im, cax = cax)
    # cbar = plt.colorbar(im, ax=ax)

    # Set the title if provided
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    # Return the normalization object if requested
    if return_norm:
        return ax, norm
    else:
        return ax


def median_combine_images(imglist: list):
    """
    Function to return median combined data
    :param imglist:
    :return:
    """
    data_list = []
    for imgname in imglist:
        with fits.open(imgname) as hdul:
            data = hdul[0].data
            data_list.append(data)

    data_array = np.array(data_list)
    return np.nanmedian(data_array, axis=0)


def normalize_and_median_combine_images(imglist: list):
    """
    Function to normalize and median combine images
    :param imglist:
    :return:
    """
    data_list = []
    for imgname in imglist:
        with fits.open(imgname) as hdul:
            data = hdul[0].data
            data_list.append(data / np.nanmedian(data))

    data_array = np.array(data_list)
    return np.nanmedian(data_array, axis=0)


def get_split_mef_fits_data(
    fits_filename: str,
) -> (list[np.ndarray], list[fits.Header]):
    """
    Get the data from a MEF fits file as a numpy array
    :param fits_filename:
    :return:
    """
    split_data, split_headers = [], []
    with fits.open(fits_filename) as hdu:
        num_ext = len(hdu)
        for ext in range(1, num_ext):
            split_data.append(hdu[ext].data)
            split_headers.append(hdu[ext].header)

    return split_data, split_headers


def join_files_to_mef(
    fits_data: list[np.ndarray],
    fits_headers: list[fits.Header],
    primary_hdu: fits.hdu.image.PrimaryHDU,
    write: bool = False,
    write_filename: str = None,
) -> fits.HDUList:
    """

    :param fits_hdus:
    :param fits_headers:
    :param primary_hdu:
    :return:
    """
    hdu_list = [primary_hdu]
    for ind, data in enumerate(fits_data):
        hdu_list.append(fits.ImageHDU(data=data, header=fits_headers[ind]))

    hdulist = fits.HDUList(hdu_list)
    if write:
        if write_filename is None:
            raise ValueError(f"Please provide a name for the output file")
        hdulist.writeto(write_filename, overwrite=True)
    return hdulist


def package_image_list_into_mef(imglist: list, output_mef: str):
    """
    Function to package a list of images into a multi-extension fits file
    :param imglist:
    :param output_mef:
    :return:
    """
    data_list = []
    for imgname in imglist:
        with fits.open(imgname) as hdul:
            data = hdul[0].data
            data_list.append(data)

    data_array = np.array(data_list)
    hdu = fits.PrimaryHDU(data_array)
    hdu.writeto(output_mef, overwrite=True)
    return output_mef


def package_image_data_into_mef(data_list: list, output_mef: str):
    """
    Function to package a list of images into a multi-extension fits file
    :param data_list:
    :param output_mef:
    :return:
    """
    data_array = np.array(data_list)
    hdu = fits.PrimaryHDU(data_array)
    hdu.writeto(output_mef, overwrite=True)
    return output_mef


def calibrate_mef_files(fits_filename: str, master_darkname: str, master_flatname: str):
    """
    Calibrate a MEF fits file
    :param fits_filename:
    :param master_darkname:
    :param master_flatname:
    :return:
    """
    split_fits_data, split_fits_headers = get_split_mef_fits_data(fits_filename)
    split_dark_data, _ = get_split_mef_fits_data(master_darkname)
    split_flat_data, _ = get_split_mef_fits_data(master_flatname)
    for i in range(len(split_fits_data)):
        split_fits_data[i] = subtract_dark(split_fits_data[i], split_dark_data[i])
        split_fits_data[i] = flat_correct(split_fits_data[i], split_flat_data[i])

    # with fits.open(fits_filename, 'update') as hdu:
    #     primary_hdu = hdu[0]

    hdulist = fits.open(fits_filename)
    primary_hdu = hdulist[0]
    calibrated_mef_hdulist = join_files_to_mef(
        split_fits_data,
        split_fits_headers,
        primary_hdu=primary_hdu,
        write=True,
        write_filename=fits_filename.replace(".fits", "_calibrated.fits"),
    )
    hdulist.close()
    return calibrated_mef_hdulist
