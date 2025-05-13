"""
master_frame_builder.py
-----------------------

Scan a directory of WINTER FITS files, group them by OBSTYPE and EXP_ACT,
median–combine each group into a master frame, and save the result as a
multi-extension FITS file.

Requires:
    • numpy, pandas, tqdm
    • WinterImage class (the version we just fixed)

Author : Nate Lourie (edited by ChatGPT)
Date   : 2025-04-30
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

# --------------------------------------------------------------------------
# 0.  Bring WinterImage into scope
# --------------------------------------------------------------------------
# Either import from its own module …
# from winter_image import WinterImage
# … or, if the class lives in the same notebook/file, ensure it’s defined
# earlier in the execution order.
from imagedaemon.image.winter_image import WinterImage  # adjust to your project layout

# --------------------------------------------------------------------------
# 1.  Discover files & header keywords
# --------------------------------------------------------------------------
_KEYSETS = {
    "obstype": ["OBSTYPE"],  # common variants
    "exptime": ["EXP_ACT", "EXPOSURE", "EXPTIME"],
}


def _find_key(header: "fits.Header", names: Iterable[str]) -> Any | None:
    """
    Return the value for the first keyword in *names* that exists in *header*.
    Comparison is case-insensitive.

    Parameters
    ----------
    header : astropy.io.fits.Header
    names  : any iterable of strings  (list, tuple, set, …)

    Returns
    -------
    The header value, or None if none of the names are found.
    """
    upper_map = {k.upper(): k for k in header.keys()}  # UPPER → real key

    for name in names:
        real = upper_map.get(name.upper())
        if real is not None:
            return header[real]

    return None


def _first_ext_header(img: "WinterImage") -> "fits.Header":
    """Return the header of the first extension; assumes at least one exists."""
    # dict preserves insertion order since Py 3.7+
    first_key = next(iter(img.headers))
    return img.headers[first_key]


def scan_directory(path: str | Path, *, pattern: str = "*.fits") -> pd.DataFrame:
    """
    Index *path* and return a DataFrame with columns
        filepath | filename | obstype | exposure | top_level_header | headers
    """
    path = Path(path)
    records: List[Dict] = []

    for fits_path in path.glob(pattern):
        try:
            img = WinterImage(str(fits_path))  # only headers read
            prim = img.top_level_header
            ext0 = _first_ext_header(img) if img.headers else prim

            # ---- OBSTYPE -------------------------------------------------
            obstype = (
                _find_key(prim, _KEYSETS["obstype"])
                or _find_key(ext0, _KEYSETS["obstype"])
                or "UNKNOWN"
            )
            obstype = str(obstype).upper()

            # ---- EXPOSURE -----------------------------------------------
            exposure = _find_key(prim, _KEYSETS["exptime"]) or _find_key(
                ext0, _KEYSETS["exptime"]
            )

            records.append(
                dict(
                    filepath=fits_path,
                    filename=fits_path.name,
                    obstype=obstype,
                    exposure=exposure,
                    top_level_header=prim,
                    headers=img.headers,
                )
            )

        except Exception as exc:
            logging.warning("Could not open %s: %s", fits_path.name, exc)

    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# 2.  Header-intersection helper
# --------------------------------------------------------------------------


def intersect_headers(headers: List["fits.Header"]) -> "fits.Header":
    """Return a header containing only cards identical in every header."""
    out = headers[0].copy()
    for key in list(out.keys()):
        if any(h.get(key) != out[key] for h in headers[1:]):
            del out[key]
    return out


# --------------------------------------------------------------------------
# 3.  Median-combine a group of WinterImages
# --------------------------------------------------------------------------


def median_combine_group(imgs: List[WinterImage]) -> WinterImage:
    """
    Median-combine a list of WinterImages (assumed same OBSTYPE & EXP_ACT)
    and return a new WinterImage whose headers are the intersection of
    the input headers.
    """
    addrs = imgs[0]._mef_addr_order  # ['sa','sb',…,'pc']
    median_data: Dict[str, np.ndarray] = {}
    median_hdrs: Dict[str, "fits.Header"] = {}

    # ----- sensor planes --------------------------------------------------
    for addr in addrs:
        if addr not in imgs[0].data:
            continue
        stack = np.stack([im.data[addr] for im in imgs])  # shape (N,H,W)
        median_data[addr] = np.median(stack, axis=0)

        hdrs = [im.headers[addr] for im in imgs]
        median_hdrs[addr] = intersect_headers(hdrs)

    # ----- primary header -------------------------------------------------
    top_hdr = intersect_headers([im.top_level_header for im in imgs])

    return WinterImage(
        data=median_data,
        headers=median_hdrs,
        top_level_header=top_hdr,
        comment=f"MEDIAN_COMBINED_{len(imgs)}",
    )


# --------------------------------------------------------------------------
# 4.  Orchestrator: build master frames for one OBSTYPE
# --------------------------------------------------------------------------


def build_master_frames(
    src_dir: str | Path,
    dst_dir: str | Path,
    *,
    obstype: str,
    camera_name: str = "winter",
) -> None:
    """
    Make master frames for all EXP_ACT values of a given OBSTYPE
    (e.g. 'DARK' or 'BIAS') found in *src_dir*.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    df = scan_directory(src_dir)
    print(df)
    df = df[df.obstype == obstype.upper()]

    if df.empty:
        logging.warning("No %s frames found in %s", obstype, src_dir)
        return

    for exp_val, grp in df.groupby("exposure"):
        logging.info(
            "Combining %d %s frames at EXP_ACT=%s", len(grp), obstype.upper(), exp_val
        )

        imgs = [
            WinterImage(str(p)) for p in tqdm(grp.filepath, desc=f"{obstype}_{exp_val}")
        ]
        master = median_combine_group(imgs)

        outname = f"{camera_name}_master{obstype.lower()}_{exp_val:.3f}s.fits"
        master.save_mef(str(dst_dir / outname))
        logging.info("  --> %s", outname)


# --------------------------------------------------------------------------
# 5.  Convenience: build darks *and* biases in one call
# --------------------------------------------------------------------------


def build_all_masters(
    src_dir: str | Path, dst_dir: str | Path, camera_name: str = "winter"
) -> None:
    """
    Build master DARK and BIAS frames (add more OBSTYPEs if needed).
    """
    for typ in ("DARK", "BIAS"):
        build_master_frames(src_dir, dst_dir, obstype=typ, camera_name=camera_name)


# --------------------------------------------------------------------------
# 6.  CLI entry-point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    RAW_DATA_DIR = Path("/path/to/raw")  # <-- EDIT
    CAL_DATA_DIR = Path("/path/to/cal")  # <-- EDIT

    build_all_masters(
        src_dir=RAW_DATA_DIR / "winter" / "darks",  # all dark/bias frames live here
        dst_dir=CAL_DATA_DIR / "winter" / "master",
        camera_name="winter",
    )
