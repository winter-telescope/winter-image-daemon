# daemon/rpc.py
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import Pyro5.core
import Pyro5.errors
import Pyro5.server
import Pyro5.socketutil
from PySide6 import QtCore

log = logging.getLogger("imagedaemon.rpc")


class PyroThread(QtCore.QThread):
    """
    Runs a Pyro5 requestLoop *and* guarantees the object is (re)registered
    in the NameServer.  It will:

        • retry every second if Ns is down at startup
        • ping Ns every 5 s; if registration is lost it re‑registers
    """

    PING_SEC = 5
    RETRY_SEC = 1

    def __init__(
        self, obj, name: str, ns_host: Optional[str], daemon_host: Optional[str] = None
    ):
        super().__init__()
        self.obj = obj
        self.name = name
        # this trick makes sure we're not using localhost and can be reached externally
        if ns_host is None:
            self.ns_host = Pyro5.socketutil.get_ip_address(
                "localhost", workaround127=True
            )
        else:
            self.ns_host = ns_host
        if daemon_host is None:
            self.daemon_host = Pyro5.socketutil.get_ip_address(
                "localhost", workaround127=True
            )
        else:
            self.daemon_host = daemon_host

        self.ns_host = ns_host
        self._stopping = threading.Event()
        self._daemon: Pyro5.server.Daemon | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def stop(self):
        self._stopping.set()
        if self._daemon:
            self._daemon.shutdown()  # unblocks requestLoop

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------
    def _locate_ns(self):
        return Pyro5.core.locate_ns(host=self.ns_host)

    def _register(self, ns):
        uri = self._daemon.register(self.obj)
        ns.register(self.name, uri)
        log.info("Registered %s [%s]", self.name, uri)

    def run(self):
        self._daemon = Pyro5.server.Daemon(self.daemon_host)
        # store the uri, so that it can be re‑registered
        # in case the NameServer goes down
        self._uri = self._daemon.register(self.obj)  # store once

        # -------- wait for NameServer ---------------------------------
        while not self._stopping.is_set():
            try:
                self._ns = Pyro5.core.locate_ns(host=self.ns_host)
                self._ns.register(self.name, self._uri)  # first registration
                log.info("Registered %s [%s]", self.name, self._uri)
                break
            except Exception as e:
                log.warning(
                    "NameServer unavailable (%s). Retry in %ds", e, self.RETRY_SEC
                )
                time.sleep(self.RETRY_SEC)

        if self._stopping.is_set():
            self._daemon.close()
            return

        # -------- periodic ping & re‑registration ---------------------
        last_ping = time.time()

        def loop_cond() -> bool:
            nonlocal last_ping
            if self._stopping.is_set():
                return False

            if time.time() - last_ping > self.PING_SEC:
                last_ping = time.time()
                try:
                    self._ns.lookup(self.name)  # still present?
                except Exception:
                    try:
                        self._ns.register(self.name, self._uri)
                        log.info("Re‑registered %s", self.name)
                    except Exception as e:
                        log.warning("Re‑registration failed (%s)", e)
                        # maybe Ns went down again → locate it on next ping
                        try:
                            self._ns = Pyro5.core.locate_ns(host=self.ns_host)
                        except Exception:
                            pass
            return True

        self._daemon.requestLoop(loopCondition=loop_cond)
        self._daemon.close()


# convenience wrapper remains unchanged
def register_object(obj, name: str, ns_host: str | None):
    t = PyroThread(obj, name, ns_host)
    t.start()
    return t
