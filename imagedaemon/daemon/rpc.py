import logging
import threading
import time

import Pyro5.core
import Pyro5.socketutil
from PySide6 import QtCore

log = logging.getLogger("imagedaemon.rpc")


class PyroThread(QtCore.QThread):
    PING_SEC = 5
    RETRY_SEC = 1

    def __init__(
        self, obj, name: str, ns_host: str | None = None, daemon_host: str | None = None
    ):
        super().__init__()
        self.obj = obj
        self.name = name

        # ---------- choose IPs -----------------------------------------
        def my_ip_for(target: str | None) -> str:
            """Return non‑loopback IP chosen for packets toward *target*."""
            return Pyro5.socketutil.get_ip_address(target, workaround127=True)

        self.ns_host = ns_host or my_ip_for(None)  # broadcast OK
        self.daemon_host = daemon_host or my_ip_for(self.ns_host)

        log.debug(
            "NameServer host = %s | Daemon host = %s", self.ns_host, self.daemon_host
        )

        self._stopping = threading.Event()
        self._daemon = None
        self._uri = None  # filled in run()

    # ------------------------------------------------------------------
    def stop(self):
        self._stopping.set()
        if self._daemon:
            self._daemon.shutdown()  # unblocks requestLoop

    # ------------------------------------------------------------------
    def run(self):
        # 1.  create the daemon on the chosen interface
        self._daemon = Pyro5.server.Daemon(host=self.daemon_host)
        self._uri = self._daemon.register(self.obj)

        # 2.  wait for NameServer, then register once
        while not self._stopping.is_set():
            try:
                self._ns = Pyro5.core.locate_ns(host=self.ns_host)
                self._ns.register(self.name, self._uri)
                log.info("Registered %s [%s]", self.name, self._uri)
                break
            except Exception as e:
                log.warning(
                    "NameServer unavailable (%s). retry in %ds", e, self.RETRY_SEC
                )
                time.sleep(self.RETRY_SEC)

        if self._stopping.is_set():  # interrupted during wait
            self._daemon.close()
            return

        # 3.  main request loop with periodic re‑registration
        last_ping = time.time()

        def loop_cond():
            nonlocal last_ping
            if self._stopping.is_set():
                return False

            if time.time() - last_ping > self.PING_SEC:
                last_ping = time.time()
                try:
                    self._ns.lookup(self.name)
                except Exception:
                    try:
                        self._ns.register(self.name, self._uri)
                        log.info("Re‑registered %s", self.name)
                    except Exception:
                        # Ns probably down again – locate on next ping
                        try:
                            self._ns = Pyro5.core.locate_ns(host=self.ns_host)
                        except Exception:
                            pass
            return True

        self._daemon.requestLoop(loopCondition=loop_cond)
        self._daemon.close()


# ----------------------------------------------------------------------
def register_object(obj, name, *, ns_host=None, daemon_host=None):
    """Spawn a background thread exporting *obj* under *name* in Pyro."""
    t = PyroThread(obj, name, ns_host=ns_host, daemon_host=daemon_host)
    t.start()
    return t
