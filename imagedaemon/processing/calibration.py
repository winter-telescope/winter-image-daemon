# calibration.py


# methods for calibrating (dark, flat, etc) images that are used between multiple cameras


from typing import List, Optional

import numpy as np


class CalibrationError(Exception):
    """
    Exception raised when calibration fails
    """

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)
        self.message = message
        self.args = args


def subtract_dark(data: np.ndarray, dark_data: np.ndarray) -> np.ndarray:
    """
    Subtract a master dark from an image

    :param image: image to subtract dark from
    :param masterdark: master dark to subtract
    :return: image with dark subtracted
    """
    return data - dark_data


def flat_correct(data: np.ndarray, flat_data: np.ndarray) -> np.ndarray:
    """
    Flat correct an image

    :param image: image to flat correct
    :param masterflat: master flat to use
    :return: flat corrected image
    """
    return data / flat_data


def median_combine(data: List[np.ndarray] | np.ndarray) -> np.ndarray:
    # what is coming in?
    print(f"data: {data}")
    print(f"data type: {type(data)}")
    print(f"type(data[0]): {type(data[0])}")
    print(f"data[0].shape: {data[0].shape}")
    match data:
        case list():
            data = np.array(data)
        case np.ndarray():
            pass
        case _:
            raise TypeError("data must be a list or a numpy array")

    # median combine the data
    print(f"data.shape: {data.shape}")

    median_data = np.nanmedian(data, axis=0)
    print(f"median_data.shape: {median_data.shape}")
    return median_data


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize an image

    :param image: image to normalize
    :return: normalized image
    """
    # normalize the data
    data = data / np.nanmedian(data)
    return data


def apply_mask(
    data: np.ndarray, mask: Optional[np.ndarray] = None, fill_value=np.nan
) -> np.ndarray:
    """
    Apply a mask to an image

    :param image: image to apply mask to
    :param mask: mask to apply to the image
    :return: masked image
    """
    if mask is not None:
        data = np.where(mask, fill_value, data)
    return data


def replace_nans_with_median(data: np.ndarray) -> np.ndarray:
    """
    Replace NaNs with the median of the image

    :param image: image to replace NaNs in
    :return: image with NaNs replaced with the median
    """
    # replace NaNs with the median of the image
    data = np.nan_to_num(data, nan=np.nanmedian(data))
    return data


def remove_horizontal_stripes(
    data: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Destripe an image

    :param image: image to destripe
    :param mask: mask to apply to the image
    :return: destriped image
    """
    if mask is not None:
        data = np.where(mask, np.nan, data)

    # destripe the image
    # by subtracting the median of each row from each pixel in that row
    row_medians = np.nanmedian(data, axis=1)
    data -= np.outer(row_medians, np.ones(data.shape[1]))
    return data


def mask_hot_pixels(data: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """
    Mask hot pixels in an image

    :param image: image to mask hot pixels in
    :param threshold: threshold to use for masking hot pixels
    :return: image with hot pixels masked
    """
    if threshold is None:
        return data

    # mask hot pixels
    data[data > threshold] = np.nan
    return data


# deprecate this:
def build_flat(
    bkg_data: List[np.ndarray] | np.ndarray, dark_data: np.ndarray, mask=None
):
    """
    bkg_data is going to be a list of np.ndarrays corresponding image data
    from multiple dithered images of the same field, or it could be a single
    np.ndarray which is N x W x H, where N is the number of dithered images
    W is the width of the image and H is the height of the image.
    If bkg_data is a list, it will be converted to a single np.ndarray
    If bkg_data is a single np.ndarray, it will be used as is.
    dark_data is going to be a single np.ndarray of the master dark
    """

    match bkg_data:
        case list():
            bkg_data = np.array(bkg_data)
        case np.ndarray():
            pass
        case _:
            raise TypeError("bkg_data must be a list or a numpy array")

    # if there is a mask, apply it to all the data, including to all layers of the background
    # and the dark data
    if mask is not None:
        mask = mask.astype(bool)
        if mask.shape != bkg_data.shape[-2:]:
            raise ValueError("mask shape must match image shape")

        # broadcast mask over the stack: (1, H, W) → (N, H, W)
        bkg_data = np.where(mask[None, ...], np.nan, bkg_data)
        dark_data = np.where(mask, np.nan, dark_data)

    # median combine the background data
    median_bkg = np.nanmedian(bkg_data, axis=0)

    # subtract the master dark from the median background
    median_bkg_darksub_data = subtract_dark(median_bkg, dark_data)

    # normalize the median background
    dither_flat_data = median_bkg_darksub_data / np.nanmedian(median_bkg_darksub_data)

    return dither_flat_data
