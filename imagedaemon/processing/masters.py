# masters.py

# methods for creating master calibration images (dark, flat, etc) that are used between multiple cameras
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from astropy.io import fits

# we will import the median_combine function from the image_operations module
from imagedaemon.processing.calibration import median_combine

# --------------------------------------------------------------------------

# Things we will make


def _intersect_headers(headers: List["fits.Header"]) -> "fits.Header":
    """Return a header containing only cards identical in every header."""
    out = headers[0].copy()
    for key in list(out.keys()):
        if any(h.get(key) != out[key] for h in headers[1:]):
            del out[key]
    return out
