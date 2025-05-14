# daemon/camera_worker.py
import logging

import Pyro5.server
from astropy.coordinates import SkyCoord
from PySide6 import QtCore

from imagedaemon.utils.wcs_utils import pix2sky

from .rpc import register_object  # helper we'll define below


class CameraWorker(QtCore.QObject):
    def __init__(self, name: str, pipelines, *, ns_host):
        super().__init__()
        self.name = name
        self.pipelines = pipelines
        self.log = logging.getLogger(f"imagedaemon.{name}")
        self.daemon_thread = register_object(self, f"{name}_daemon", ns_host)

    def shutdown(self):
        self.daemon_thread.stop()
        self.daemon_thread.wait()  # block until fully finished
        self.log.info("Daemon thread stopped")

    # ---------- RPC‑exposed methods ----------
    @Pyro5.server.expose
    def solve_astrometry(
        self,
        science_image: str,
        addr: str | None = None,
        background_image_list: list[str] = None,
        output_dir: str | None = None,
        pix_coords: tuple[int, int] | None = None,
        timeout: int = 30,
        **astrometry_opts,
    ):
        """RPC call: solve astrometry for the specified image.
        takes in optional list of backgroudn images to use for
        background subtraction.
        can also take in xpix and ypix to specify the pixel location
        to report back the astrometric solution by applying the
        WCS. If none is provided, the center of the image will be used.
        """

        info = self.pipelines.get_astrometric_solution(
            science_image=science_image,
            addr=addr,
            background_image_list=background_image_list,
            output_dir=output_dir,
            timeout=timeout,
            **astrometry_opts,
        )

        if pix_coords is None:
            # use the center of the image
            pix_coords = (info["image_width"] // 2, info["image_height"] // 2)

        sky: SkyCoord = pix2sky(pix_coords[0], pix_coords[1], info["wcs"])
        ra_deg = float(sky.ra.deg)
        dec_deg = float(sky.dec.deg)

        return {
            "ra": ra_deg,
            "dec": dec_deg,
            "ra_guess": info["ra_guess"],
            "dec_guess": info["dec_guess"],
            "pix_coords": pix_coords,
            "pixel_scale": info["pixel_scale"],
            "rotation_deg": info["rotation_deg"],
            "image_width": info["image_width"],
            "image_height": info["image_height"],
        }

    @Pyro5.server.expose
    def run_focus_loop(self, image_list: list[str], **kwargs):
        if hasattr(self.pipelines, "run_focus_loop"):
            return self.pipelines.run_focus_loop(image_list, **kwargs)
        raise NotImplementedError("Focus loop not implemented for this camera")

    @Pyro5.server.expose
    def validate_startup(self, bias_image: str, **kwargs):
        if hasattr(self.pipelines, "validate_startup"):
            return self.pipelines.validate_startup(bias_image, **kwargs)
        raise NotImplementedError("Startup validation not implemented")
