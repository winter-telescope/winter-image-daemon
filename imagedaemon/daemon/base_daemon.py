# daemon/base_daemon.py
import logging
import signal
import sys

from PySide6 import QtCore, QtWidgets


class ImageDaemon(QtWidgets.QApplication):
    def __init__(self, camera_names: list[str], *, ns_host: str | None, log=None):
        super().__init__(sys.argv)

        # set up garbage collection for clean shutdown
        self.aboutToQuit.connect(self._cleanup_threads)

        self.log = log or logging.getLogger("imagedaemon.daemon")
        self.workers = []

        for cam in camera_names:
            try:
                from imagedaemon.registry import get

                pipelines = get(cam)
            except KeyError:
                self.log.error("Camera %s not found in registry", cam)
                continue

            from .camera_worker import CameraWorker

            worker = CameraWorker(cam, pipelines, ns_host=ns_host)
            self.workers.append(worker)

        # ----- graceful Ctrl‑C ---------------------------------------

        signal.signal(signal.SIGINT, lambda *_: self.quit())

        # SIGINT helper: tick Python every 200 ms so the handler fires
        self._sigint_timer = QtCore.QTimer(self)
        self._sigint_timer.setInterval(200)  # ms
        self._sigint_timer.timeout.connect(lambda: None)
        self._sigint_timer.start()

    def _cleanup_threads(self):
        """Clean up threads before exiting."""
        for worker in self.workers:
            worker.shutdown()
        self._sigint_timer.stop()
