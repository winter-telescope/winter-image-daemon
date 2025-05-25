import astropy
import numpy as np
from astropy.io import fits


def mask_datasec(
    data: np.ndarray,
    header: astropy.io.fits.Header,
    fill_value: int | bool | float,  # eg, 1, True, np.nan
) -> np.ndarray:
    """
    Function to mask the data section of an image
    """
    datasec = header["DATASEC"].replace("[", "").replace("]", "").split(",")
    datasec_xmin = int(datasec[0].split(":")[0])
    datasec_xmax = int(datasec[0].split(":")[1])
    datasec_ymin = int(datasec[1].split(":")[0])
    datasec_ymax = int(datasec[1].split(":")[1])

    data[:, :datasec_xmin] = fill_value
    data[:, datasec_xmax:] = fill_value
    data[:datasec_ymin, :] = fill_value
    data[datasec_ymax:, :] = fill_value
    return data
