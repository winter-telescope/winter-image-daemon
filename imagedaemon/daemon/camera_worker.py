# daemon/camera_worker.py
import logging
from typing import Optional

import Pyro5.server
from astropy.coordinates import SkyCoord
from PySide6 import QtCore

from imagedaemon.utils.notify import SlackNotifier
from imagedaemon.utils.paths import ENV_FILE
from imagedaemon.utils.wcs_utils import pix2sky

from .rpc import register_object  # helper we'll define below


class CameraWorker(QtCore.QObject):
    def __init__(self, name: str, pipelines, *, ns_host):
        super().__init__()
        self.name = name
        self.pipelines = pipelines
        self.log = logging.getLogger(f"imagedaemon.{name}")
        self.daemon_thread = register_object(self, f"{name}_daemon", ns_host)
        self.notifier = SlackNotifier(env_file=ENV_FILE)

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
    def run_focus_loop(
        self,
        image_list: list[str],
        addrs: Optional[str] = None,
        output_dir: Optional[str] = None,
        post_plot: bool = False,
        **kwargs,
    ):
        """run a focus loop on the specified image list.
        For multiple sensor images (e.g. WINTER), optional
        addrs argument can be used to specify which sensors to
        use in the aggregated median best focus."""

        results = self.pipelines.run_focus_loop(
            image_list=image_list,
            addrs=addrs,
            output_dir=output_dir,
            **kwargs,
        )

        if post_plot:
            try:
                image_path = results.get("plot", None)
                if image_path:
                    text = f"Focus Results: Best Focus = {results.get('best_focus', 0):0.1f})"
                    self.notifier.post_image(image_path=image_path, text=text)
                else:
                    self.log.warning("No focus plot found to post to Slack")
            except Exception as e:
                self.log.error(f"Error posting focus plot to Slack: {e}")
        return results

    @Pyro5.server.expose
    def validate_startup(self, bias_image: str, **kwargs):
        if hasattr(self.pipelines, "validate_startup"):
            return self.pipelines.validate_startup(bias_image, **kwargs)
        raise NotImplementedError("Startup validation not implemented")
