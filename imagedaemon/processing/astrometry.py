import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import List, Union

import numpy as np
from astropy.io import fits
from astropy.wcs.utils import proj_plane_pixel_scales

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
    timeout: int = 30,
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
            start_new_session=True,  # ← each child inherits the new pgid
        )
        """
        # realtime streaming of output
        # works but doesn't handle any timeout
        
        for line in proc.stdout:
            sys.stdout.write(line)
        proc.wait()
        """
        try:
            out, _ = proc.communicate(timeout=timeout)
            sys.stdout.write(out)
        except subprocess.TimeoutExpired:
            # terminate the whole process group (SIGTERM then SIGKILL fallback)
            pgid = proc.pid  # leader’s pid == pgid because of start_new_session
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.communicate(timeout=5)  # wait a moment for clean exit
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.communicate()
            raise RuntimeError(f"solve‑field exceeded {timeout}s wall‑clock limit")
        # still check exit status
        if proc.returncode:
            raise RuntimeError(f"solve-field failed (exit {proc.returncode})")

        # ------------------------------------------------------------------
        # 4. build WCS from the .new file
        # ------------------------------------------------------------------
        solved_path = input_path.with_suffix(".new")
        solved_image = Image(solved_path)
        wcs = wcs_from_header(solved_image.header)

        # ---------------------- extra info to return ---------------------
        hdr = solved_image.header
        width = hdr.get("IMAGEW") or hdr.get("NAXIS1")
        height = hdr.get("IMAGEH") or hdr.get("NAXIS2")

        # pixel scale (arcsec/px)  — returns array([scale_x, scale_y]) in degrees
        scale_deg = proj_plane_pixel_scales(wcs)  # deg / pix
        pixel_scale = float(scale_deg.mean() * 3600.0)  # arcsec / pix

        # rotation angle (deg, "east of north")
        cd = wcs.wcs.cd if wcs.wcs.has_cd() else wcs.wcs.pc
        rot_rad = np.arctan2(-cd[0, 1], cd[1, 1])  # radians
        rotation_deg = float(np.degrees(rot_rad))

    # At this point:
    #   * tmp mode: tmpdir is gone, but we already parsed the WCS.
    #   * explicit output_dir: all artefacts kept there.
    #   * output_dir is None: artefacts live next to original FITS.

    return {
        "wcs": wcs,  # full astropy.wcs.WCS object
        "image_width": width,
        "image_height": height,
        "pixel_scale": pixel_scale,  # arcsec / pixel
        "rotation_deg": rotation_deg,
        "solved_fits": str(solved_path),
    }
