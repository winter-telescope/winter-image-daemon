import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import List, Union

import numpy as np
from astropy.io import fits

from imagedaemon.utils.image import Image
from imagedaemon.utils.wcs_utils import wcs_from_header  # your helper

PathLike = Union[str, Path]


def run_astrometry(
    fits_path: PathLike,
    ra: float,
    dec: float,
    radius: float = 2.0,
    scale_low: float = 1.1,
    scale_high: float = 1.2,
    downsample: int = 2,
    output_dir: PathLike | None = None,  # None, Path, or "tmp"
):
    """
    Call solve‑field and return an astropy.wcs.WCS built from the .new file.

    Parameters
    ----------
    fits_path
        Path to the science FITS to solve.
    ra, dec, radius
        Sky search constraints (deg).
    scale_low, scale_high
        Pixel‑scale bounds (arcsec/px).
    downsample
        Downsampling factor fed to solve‑field.
    output_dir
        • None  → run where FITS lives.
        • Path/str → copy FITS there and keep all solve‑field outputs.
        • "tmp" → run in a TemporaryDirectory that is deleted on return.
    """
    fits_path = Path(fits_path)

    # ------------------------------------------------------------------
    # choose working directory & input copy
    # ------------------------------------------------------------------
    if output_dir is None:
        work_dir = fits_path.parent
        input_path = fits_path
        cleanup_ctx = nullcontext()  # <- no clean‑up needed

    elif str(output_dir).lower() == "tmp":
        tmpobj = tempfile.TemporaryDirectory()
        work_dir = Path(tmpobj.name)
        input_path = work_dir / fits_path.name
        shutil.copy2(fits_path, input_path)
        cleanup_ctx = tmpobj  # <- auto‑delete temp dir

    else:
        work_dir = Path(output_dir).expanduser()
        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / fits_path.name
        shutil.copy2(fits_path, input_path)
        cleanup_ctx = nullcontext()  # <- keep artefacts, no clean‑up

    # ------------------------------------------------------------------
    # assemble solve‑field command
    # ------------------------------------------------------------------
    cmd = (
        f"solve-field {input_path} "
        f"--scale-units arcsecperpix --scale-low {scale_low} --scale-high {scale_high} "
        f"--ra {ra} --dec {dec} --radius {radius} "
        f"--downsample {downsample} --overwrite "
        "--no-plots "
        "--cpulimit 30 "
    )
    print(f"[astrometry] {cmd}\n")

    # ------------------------------------------------------------------
    # run solve‑field inside chosen working dir
    # ------------------------------------------------------------------
    with cleanup_ctx:  # may be tmpdir or null‑context
        proc = subprocess.Popen(
            shlex.split(cmd),
            cwd=work_dir,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
        proc.wait()
        if proc.returncode:
            raise RuntimeError(f"solve-field failed (exit {proc.returncode})")

        # ------------------------------------------------------------------
        # 4. build WCS from the .new file
        # ------------------------------------------------------------------
        solved_path = input_path.with_suffix(".new")
        solved_image = Image(solved_path)
        wcs = wcs_from_header(solved_image.header)

    # At this point:
    #   * tmp mode: tmpdir is gone, but we already parsed the WCS.
    #   * explicit output_dir: all artefacts kept there.
    #   * output_dir is None: artefacts live next to original FITS.

    return wcs
