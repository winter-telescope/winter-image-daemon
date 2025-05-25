import argparse
import logging
import sys

from imagedaemon import add_file_logger  # your earlier helper

from .base_daemon import ImageDaemon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cameras",
        required=True,
        help="comma‑separated list of camera names on this host",
    )
    ap.add_argument(
        "-n",
        "--ns-host",
        default=None,
        help="Pyro NameServer host (default: autodetect)",
    )
    ap.add_argument("--logfile", default=None)
    args = ap.parse_args()

    if args.logfile:
        add_file_logger(args.logfile)

    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    app = ImageDaemon(cams, ns_host=args.ns_host, log=logging.getLogger())
    sys.exit(app.exec())
