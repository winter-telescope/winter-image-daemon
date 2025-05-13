from typing import Iterable, Mapping, Union

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS


def wcs_from_header(
    header: Mapping[str, Union[str, float]],
    *,
    default_projection: str = "TAN",
) -> WCS:
    """
    Create an astropy.wcs.WCS object from a FITS header or dict.

    The function understands both the modern CD matrix formalism
    and the older PC + CDELT representation.

    Parameters
    ----------
    header
        astropy.io.fits.Header *or* plain dict with WCS keywords.
    default_projection
        Fallback three‑letter code (e.g. 'TAN', 'SIN', 'AIT') if
        CTYPE1/2 are missing.  Ignored when they are present.

    Returns
    -------
    astropy.wcs.WCS
    """
    # make a case‑insensitive lookup
    h = {k.upper(): v for k, v in dict(header).items()}

    # mandatory keywords
    crval1 = h["CRVAL1"]
    crval2 = h["CRVAL2"]
    crpix1 = h["CRPIX1"]
    crpix2 = h["CRPIX2"]

    # CD matrix (preferred) or PC + CDELT
    if all(k in h for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
        cd = np.array([[h["CD1_1"], h["CD1_2"]], [h["CD2_1"], h["CD2_2"]]])
    else:
        # fall back to PC * diag(CDELT)
        cdelt1 = h["CDELT1"]
        cdelt2 = h["CDELT2"]
        pc11 = h.get("PC1_1", 1.0)
        pc12 = h.get("PC1_2", 0.0)
        pc21 = h.get("PC2_1", 0.0)
        pc22 = h.get("PC2_2", 1.0)
        cd = np.array([[pc11 * cdelt1, pc12 * cdelt2], [pc21 * cdelt1, pc22 * cdelt2]])

    # projection type
    ctype1 = h.get("CTYPE1", f"RA---{default_projection}")
    ctype2 = h.get("CTYPE2", f"DEC--{default_projection}")

    w = WCS(naxis=2)
    w.wcs.crval = [crval1, crval2]
    w.wcs.crpix = [crpix1, crpix2]
    w.wcs.cd = cd
    w.wcs.ctype = [ctype1, ctype2]
    return w


# ───────────────────────────────────────────────────────────────────
# 2. pixel → sky (unchanged) ----------------------------------------
# ───────────────────────────────────────────────────────────────────
def pix2sky(
    x: Union[float, Iterable[float]],
    y: Union[float, Iterable[float]],
    wcs: WCS,
    frame: str = "icrs",
) -> SkyCoord:
    """
    Convert DS9 *Physical* pixel coordinates (1‑based) to a SkyCoord.

    Parameters
    ----------
    x, y
        Pixel positions (scalar or array‑like).
    wcs
        A 2‑D astropy.wcs.WCS object.
    frame
        Desired celestial frame for the output (default 'icrs').

    Returns
    -------
    astropy.coordinates.SkyCoord
    """
    ra, dec = wcs.all_pix2world(x, y, 1)  # origin=1 → FITS/DS9 convention
    return SkyCoord(ra * u.deg, dec * u.deg, frame=frame)
