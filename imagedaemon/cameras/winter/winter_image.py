#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:57:51 2023

This is a sandbox for analyzing bias images from the WINTER camera and
deciding automatically whether each sensor is in a well-behaved state.

@author: nlourie
"""

import logging
import os
import re
import sys
from os import PathLike, fspath
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import astropy.visualization
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

from imagedaemon.utils.image import Image

# Disable LaTeX rendering in matplotlib
matplotlib.rcParams["text.usetex"] = False

# ------------------------------------------------------------------
# FITS keywords that must NOT appear in an extension header
# ------------------------------------------------------------------
_STRUCTURAL_KEYS: set[str] = {
    # Primary‑HDU only
    "SIMPLE",
    "EXTEND",
    "BLOCKED",
    # Primary + ImageHDU bit depth
    "BITPIX",
    # Axis sizes
    "NAXIS",
    "NAXIS1",
    "NAXIS2",
    "NAXIS3",
    "NAXIS4",
    # Table / HDU bookkeeping
    "PCOUNT",
    "GCOUNT",
    "TFIELDS",
    # Random groups (obsolete)
    "GROUPS",
}


class WinterImage:
    """
    A class to handle the WINTER camera data and headers.

    Attributes:
        filepath: Path to the image file.
        filename: Name of the image file.
        comment: Additional comments about the image.
        logger: Logger instance for logging messages.
        verbose: Flag to control verbosity of logs.
        imgs: Dictionary holding sub-images keyed by address.
        headers: List of headers for each sub-image.
        header: Top-level FITS header.
    """

    # Class variables for MEF address and board ID order:
    # these are the SAME for every WinterImage
    _mef_addr_order: ClassVar[List[str]] = ["sa", "sb", "sc", "pa", "pb", "pc"]
    _board_id_order: ClassVar[List[int]] = [2, 6, 5, 1, 3, 4]

    _layer_by_addr: ClassVar[Dict[str, int]] = {
        addr: idx for idx, addr in enumerate(_mef_addr_order)
    }
    _layer_by_board_id: ClassVar[Dict[int, int]] = {
        bid: idx for idx, bid in enumerate(_board_id_order)
    }

    _board_id_by_addr: ClassVar[Dict[str, int]] = dict(
        zip(_mef_addr_order, _board_id_order)
    )
    _addr_by_board_id: ClassVar[Dict[int, str]] = {
        v: k for k, v in _board_id_by_addr.items()
    }

    _rowcol_locs: ClassVar[List[Tuple[int, int]]] = [
        (0, 1),
        (1, 1),
        (2, 1),
        (2, 0),
        (1, 0),
        (0, 0),
    ]
    _rowcol_locs_by_addr: ClassVar[Dict[str, Tuple[int, int]]] = dict(
        zip(_mef_addr_order, _rowcol_locs)
    )

    def __init__(
        self,
        data: Union[str, np.ndarray],
        headers: Optional[Dict[str, Any]] = None,
        top_level_header: Optional[fits.Header] = None,
        comment: str = "",
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the WinterImage object and loads the image or data.

        :param data: File path to a FITS file or a numpy array of image data.
        :param headers: Dictionary of headers corresponding to the image layers. Defaults to None.
        :param comment: Additional comments about the image. Defaults to an empty string.
        :param logger: Logger instance for logging. Defaults to None.
        :param verbose: Enables verbose logging if True. Defaults to False.

        :raises ValueError: If the input data format is invalid.
        """
        self.filepath: str = ""
        self.filename: str = ""
        self.comment: str = comment
        self.logger: Optional[logging.Logger] = logger
        self.verbose: bool = verbose
        self.data: Dict[str, np.ndarray] = {}
        self.masks: Dict[str, np.ndarray] = {}
        self.top_level_header: Optional[fits.Header] = top_level_header
        self.headers: Dict[str, fits.Header] = headers or {}
        self.header: fits.Header = fits.Header()

        match data:
            case str() | Path():
                self.load_image(data, comment)
                self.masks = self.get_masks()
            case dict():
                self.load_data(
                    data, headers_dict=headers, top_level_header=top_level_header
                )
                self.masks = self.get_masks()
            case _:
                raise ValueError(
                    f"Input data format {type(data)} not valid. Must be a numpy array of sub-images or a filepath."
                )

    @classmethod
    def get_addrs(cls) -> List[str]:
        return cls._mef_addr_order

    @classmethod
    def get_board_ids(cls) -> List[int]:
        return cls._board_id_order

    def get_board_id_for(self, addr: str) -> int:
        """
        Get the board ID for a given address.

        :param addr: The address of the sensor.
        :return: The board ID corresponding to the address.
        """
        if addr not in self._board_id_by_addr:
            raise ValueError(f"Address {addr} not found.")
        return self._board_id_by_addr[addr]

    def load_image(self, mef_file_path: str, comment: str = "") -> None:
        """
        Loads the data from a MEF FITS file into a numpy array.

        :param mef_file_path: The file path to the MEF FITS file.
        :param comment: Additional comments about the image. Defaults to an empty string.
        """
        self.filepath = mef_file_path
        self.filename = os.path.basename(self.filepath)
        self.comment = comment
        self.data = {}
        self.headers = {}
        self.top_level_header = None

        with fits.open(self.filepath) as hdu:
            self.top_level_header = hdu[0].header
            for ext in hdu[1:]:
                data = ext.data
                # ensure data is recast as float32
                data = np.array(data, dtype=np.float32)
                # old: trim it down by datasec
                # datasec_str = ext.header["DATASEC"][1:-1]
                # datasec = np.array(re.split(r"[,:]", datasec_str)).astype(int)
                # data = ext.data[datasec[2] : datasec[3], datasec[0] : datasec[1]]

                addr = ext.header.get("ADDR", None)
                if addr in self._mef_addr_order:
                    self.data[addr] = data
                    self.headers[addr] = ext.header
                else:
                    boardid = ext.header.get("BOARD_ID", None)
                    if boardid in self._board_id_order:
                        addr_mapped = self._addr_by_board_id[boardid]
                        self.data[addr_mapped] = data
                        self.headers[addr_mapped] = ext.header

    def log(self, msg: str, level: int = logging.INFO) -> None:
        """
        Logs a message with the specified logging level.

        :param msg: The message to log.
        :param level: The logging level (e.g., logging.INFO). Defaults to logging.INFO.
        """
        formatted_msg = f"WinterImage {msg}"

        if self.logger is None:
            print(formatted_msg)
        else:
            self.logger.log(level=level, msg=formatted_msg)

    def load_data(
        self,
        data_dict: Dict[str, np.ndarray],
        headers_dict: Optional[Dict[str, Any]] = None,
        top_level_header: Optional[fits.Header] = None,
    ) -> None:
        """Load data that’s already in memory (e.g. after arithmetic)."""
        self.data = data_dict
        self.headers = headers_dict or {}  # <-- keep it a dict
        self.top_level_header = top_level_header

    def plot_image(
        self,
        title: Optional[str] = None,
        cbar: bool = False,
        cmap: Union[str, Dict[str, str]] = "gray",
        norm_by: str = "full",
        apply_mask: bool = True,
        post_to_slack: bool = False,  # pylint: disable=unused-argument
        savepath: Optional[str] = None,
        channel_labels: Optional[
            str
        ] = None,  # can be either "addr" or "board_id" or None
        single_image_width: Optional[float] = 3,
        **kwargs: Any,
    ) -> None:
        """
        Plots a mosaic of sub-images with an optional color bar and color map.

        :param title: The title of the plot. Defaults to None.
        :param cbar: Whether to add a color bar. Defaults to False.
        :param cmap: Colormap or a dictionary of colormaps by address. Defaults to "gray".
        :param norm_by: Normalization method ("full", "sensor", or "chan"). Defaults to "full".
        :param post_to_slack: Unused argument. Defaults to False.
        :param savepath: Path to save the plotted mosaic. Defaults to None.
        :param kwargs: Additional keyword arguments passed to `imshow`.
        """
        aspect_ratio = 1920 / 1080
        w = single_image_width
        h = w / aspect_ratio

        fig, axarr = plt.subplots(3, 2, figsize=(4 * h, 2.0 * w), dpi=200)
        # Combine all the data to figure out the full-image normalization
        alldata = np.concatenate([img.flatten() for img in self.data.values()])

        for addr in self._mef_addr_order:
            if addr in self.data:
                img_full = self.data[addr]
                # --- trim just for plotting ---------------------------------
                if apply_mask:
                    mask = self.masks[addr]
                    image = img_full.copy()
                    image[mask] = np.nan

                else:
                    image = img_full
                # -------------------------------------------------------------
                # Rotate starboard images by 180 degrees
                if addr.startswith("s"):
                    image = np.rot90(image, 2)

                # set the normalization
                if norm_by.lower() == "full":
                    normdata = alldata
                elif norm_by.lower() in ["sensor", "chan"]:
                    normdata = image
                else:
                    normdata = alldata

                norm = astropy.visualization.ImageNormalize(
                    normdata,
                    interval=astropy.visualization.ZScaleInterval(),
                    stretch=astropy.visualization.SqrtStretch(),
                )
            else:
                image = np.zeros((1081, 1921))
                # this none helps it render better on some machines
                norm = None

            rowcol = self._rowcol_locs_by_addr[addr]
            row, col = rowcol

            ax0 = axarr[row, col]

            if isinstance(cmap, str):
                current_cmap = cmap
            elif isinstance(cmap, dict):
                current_cmap = cmap.get(addr, "gray")
            else:
                current_cmap = "gray"
            if channel_labels is not None:
                if channel_labels.lower() == "addr":
                    addr = addr
                elif channel_labels.lower() == "board_id":
                    addr = str(self._board_id_by_addr[addr])
                else:
                    raise ValueError(
                        f"channel_labels must be 'addr' or 'board_id', not {channel_labels}"
                    )
                addr = addr.upper()
                ax0.text(
                    990,
                    540,
                    addr,
                    fontsize=60,
                    color="white",
                    ha="center",
                    va="center",
                )
            else:
                pass

            individual_plot = ax0.imshow(
                image, origin="lower", cmap=current_cmap, norm=norm, **kwargs
            )
            ax0.set_xlabel("X [pixels]")
            ax0.set_ylabel("Y [pixels]")

            ax0.grid(False)
            ax0.axis("off")

        plt.subplots_adjust(wspace=0.03, hspace=0.01)
        if title is not None:
            plt.suptitle(title)

        if cbar:
            # Add colorbar to the last axis
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(individual_plot, cax=cax, orientation="vertical")

        if savepath is not None:
            plt.savefig(savepath, dpi=500)
        plt.show()

    def get_sensor_image(
        self,
        chan: Union[str, int],
        index_by: str = "addr",
    ) -> "Image":
        """
        Return a *single* :class:`Image` instance for the requested sensor.

        >>> img = WinterImage("file.fits").get_sensor_image("pc")       # by addr
        >>> img = WinterImage("file.fits").get_sensor_image(4, "board") # by board ID
        >>> img.header["ADDR"]   # still has the combined header cards
        """
        addr = self._resolve_addr(chan, index_by=index_by)
        data = self.data[addr]  # raises KeyError if missing
        sensor_hdr = self.headers.get(addr, fits.Header())

        combined_hdr = self._combine_headers(sensor_hdr)

        # include the pre‑computed mask for that sensor (if present)
        mask = self.masks.get(addr)

        # NB:  Image class is the one you showed earlier
        return Image(data.copy(), header=combined_hdr, mask=mask)

    # save the image to a multi-extension fits file
    def save_mef(self, filename: str | Path) -> None:
        # --- ensure we have a plain str -----------------------------------
        fn = fspath(filename)  # handles str, Path, or PathLike

        if not fn.lower().endswith(".fits"):
            # add .fits extension
            fn = f"{fn}.fits"

        # Primary HDU ------------------------------------------------------
        prim_hdr = (self.top_level_header or fits.Header()).copy()
        hdul = fits.HDUList([fits.PrimaryHDU(header=prim_hdr)])

        # Extension HDUs ---------------------------------------------------
        for addr in self._mef_addr_order:
            if addr not in self.data:
                continue
            data = self.data[addr]
            hdr = self.headers.get(addr, fits.Header()).copy()
            hdr.setdefault("EXTNAME", addr.upper())
            hdr["ADDR"] = addr
            hdul.append(fits.ImageHDU(data=data, header=hdr))

        hdul.writeto(fn, overwrite=True)

    def save_image(self, filename: str | Path) -> None:
        self.save_mef(filename)

    def save_sensors(self, filename: str | Path) -> None:
        fn_base = os.path.splitext(fspath(filename))[0]
        for addr in self._mef_addr_order:
            img = self.get_sensor_image(addr)
            sensor_filename = f"{fn_base}_{addr}.fits"
            img.save_image(sensor_filename)

    def get_masks(self) -> Dict[str, np.ndarray]:
        """Get the masks for each sensor image. Uses the raw winter mask
        definition and applies any DATASEC trimming if available.
        :return: A dictionary of masks for each sensor image.
        """
        masks = {}
        for addr in self._mef_addr_order:
            if addr not in self.data:
                continue

            data = self.data[addr]
            mask = self.get_raw_winter_mask(data, self._board_id_by_addr[addr])

            # Only try DATASEC trimming if we *have* a header for this addr
            hdr = self.headers.get(addr)
            if hdr and "DATASEC" in hdr:
                datasec_str = hdr["DATASEC"][1:-1]
                xmin, xmax, ymin, ymax = map(int, re.split(r"[,:]", datasec_str))
                datasec_mask = np.zeros_like(data, dtype=bool)
                datasec_mask[:, :xmin] = True
                datasec_mask[:, xmax:] = True
                datasec_mask[:ymin, :] = True
                datasec_mask[ymax:, :] = True
                mask = np.logical_or(mask, datasec_mask)

            masks[addr] = mask
        return masks

    @classmethod
    def get_raw_winter_mask(self, data: np.array, board_id: int) -> np.array:
        """
        Get mask for raw winter image.
        data is a numpy array of the image data
        board_id is the board ID of the image sensor
        :return: mask is a numpy array of the same shape as data, with 1s where the data is bad and 0s where the data is good
        """

        mask = np.zeros(data.shape)
        if board_id == 0:
            # Mask the outage in the bottom center
            mask[:500, 700:1600] = 1.0
            mask[1075:, :] = 1.0
            mask[:, 1950:] = 1.0
            mask[:20, :] = 1.0

        if board_id == 1:
            mask[:, 344:347] = 1.0
            mask[:, 998:1000] = 1.0
            mask[:, 1006:1008] = 1.0
            mask[260:262, :] = 1.0
            # Mask entire striped area to the right of the chip
            # mask[:, 1655:] = 1.0

            # Mask the low sensitivity regions around edges
            mask[1070:, :] = 1.0
            mask[:20, :] = 1.0
            mask[:, :75] = 1.0
            mask[:, 1961:] = 1.0

        if board_id == 2:
            mask[1060:, :] = 1.0
            mask[:, 1970:] = 1.0
            mask[:55, :] = 1.0
            mask[:, :20] = 1.0
            mask[:475, 406:419] = 1.0
            mask[350:352, :] = 1.0
            mask[260:287, :66] = 1.0
            mask[:, 1564:1567] = 1.0
            mask[:, 1931:] = 1.0

        if board_id == 3:
            mask[1085:, :] = 1.0

            # mask the right side
            # mask[:, 1970:] = 1.0
            mask[:, -50:] = 1.0

            mask[:55, :] = 1.0

            # mask the left side
            # mask[:, :20] = 1.0
            mask[:, :100] = 1.0

            # Mask outages on top right and bottom right
            mask[:180, 1725:] = 1.0
            mask[1030:, 1800:] = 1.0

        if board_id == 4:

            # # Mask the region to the top left
            mask[610:, :250] = 1.0
            # # There seems to be a dead spot in the middle of the image
            mask[503:518, 384:405] = 1.0

            # Mask the edges with low sensitivity due to masking
            # mask[:, 1948:] = 1.0
            # mask[:, :61] = 1.0
            # mask[:20, :] = 1.0
            # mask[1060:, :] = 1.0

            # Mask the right side
            mask[:, -50:] = 1.0

            # mask the bottom edge
            mask[:50, :] = 1.0

            # mask the top edge
            mask[-50:, :] = 1.0

            # Mask a vertical strip
            mask[:, 998:1002] = 1.0

            # Mask another vertical strip
            mask[:, 1266:1273] = 1.0

            # Mask the outage to the right
            mask[145:, 1735:] = 1.0
            # mask[data > 40000] = 1.0

            # Mask random vertical strip
            mask[:, 1080:1085] = 1.0

            """# More aggressive masking
            # mask out the outer 100 pixels on all edges
            n = 100
            # mask the top edge
            mask[:n, :] = 1.0
            # mask the bottom edge
            mask[-n:, :] = 1.0
            # mask the right edge
            mask[:, :n] = 1.0
            # mask the left edge
            mask[:, -n:] = 1.0"""

        if board_id == 5:
            # Mask the outage in the top-right.
            mask[700:, 1200:1900] = 1.0
            mask[1072:, :] = 1.0
            mask[:, 1940:] = 1.0
            mask[:15, :] = 1.0

        if board_id == 6:
            # Mask channel 0
            mask[0::2, 0::4] = 1.0

            # mask the top edge
            mask[:50, :] = 1.0
            # mask the right edge
            mask[:, :25]
            # mask the left edge
            mask[:, -50:] = 1.0
            # mask the bottom edge
            mask[-50:, :] = 1.0

        return mask.astype(bool)

    # --------------------------------------------------------------------------
    #  Dictionary-like access and arithmetic sugar
    # --------------------------------------------------------------------------

    # --- mapping helpers --------------------------------------------------
    def __getitem__(self, key: str) -> np.ndarray:
        """Return the numpy array for a given sensor address (case-insensitive)."""
        return self.data[key.lower()]

    def keys(self):
        """Expose the available sensor addresses (so `dict(im)` works)."""
        return self.data.keys()

    # --- internal helper --------------------------------------------------
    def _binary_op(self, other, op):
        """
        Core utility: apply `op` element-wise and return a **dict[str, ndarray]**.

        * If `other` is another WinterImage, operate on the intersection of
          sensor keys present in *both* images.
        * If `other` is a scalar (int | float | numpy scalar), broadcast it.
        """
        if isinstance(other, WinterImage):
            common = self.keys() & other.keys()
            if not common:
                raise ValueError("No overlapping sensor keys between images.")
            return {k: op(self.data[k], other.data[k]) for k in common}

        # assume scalar-like
        return {k: op(v, other) for k, v in self.data.items()}

    # --- arithmetic dunders ----------------------------------------------
    def __add__(self, other):
        return self._binary_op(other, np.add)

    def __radd__(self, other):
        return self._binary_op(other, np.add)

    def __sub__(self, other):
        return self._binary_op(other, np.subtract)

    def __rsub__(self, other):
        return self._binary_op(other, lambda a, b: np.subtract(b, a))

    def __mul__(self, other):
        return self._binary_op(other, np.multiply)

    def __rmul__(self, other):
        return self._binary_op(other, np.multiply)

    def __truediv__(self, other):
        return self._binary_op(other, np.divide)

    def __rtruediv__(self, other):
        return self._binary_op(other, lambda a, b: np.divide(b, a))

    def _resolve_addr(self, chan: Union[str, int], index_by: str = "addr") -> str:
        """
        Translate *chan* + *index_by* exactly the same way ``get_img`` did
        but *only* return the canonical sensor address string (e.g. "pc").
        """
        index_by = index_by.lower()
        if index_by in ("addr", "name"):
            if not isinstance(chan, str):
                raise TypeError("chan must be str when index_by='addr'")
            return chan.lower()

        if index_by in ("board_id", "id"):
            if not isinstance(chan, int):
                raise TypeError("chan must be int when index_by='board_id'")
            addr = self._addr_by_board_id.get(chan)
            if addr is None:
                raise ValueError(f"Board‑ID {chan} not found")
            return addr

        if index_by == "layer":
            if not isinstance(chan, int):
                raise TypeError("chan must be int when index_by='layer'")
            try:
                return self._mef_addr_order[chan]
            except IndexError:
                raise ValueError(f"Layer {chan} out of range")

        raise ValueError("index_by must be 'addr', 'board_id', or 'layer'")

    def _combine_headers(
        self,
        sensor_hdr: fits.Header | None,
    ) -> fits.Header:
        """
        Merge *top_level_header* + *sensor_hdr*.

        • Start with an empty header
        • Copy every card from the top-level HDU
        • For every key in *sensor_hdr*
            - if the key isn't present → append the card
            - if the key *is* present and the values match → leave as is
            - if the values differ → drop the key entirely
            strip out any structural keys that may break the FITS standard
        """
        combined = fits.Header()

        def _copy_cards(src_hdr: fits.Header):
            for card in src_hdr.cards:
                key = card.keyword

                # skip structural / primary‑only keys
                if key in _STRUCTURAL_KEYS or key.startswith("NAXIS"):
                    continue

                # copy if not present, else keep existing value
                if key not in combined:
                    combined.append(card, end=True)

        # Primary header first
        if self.top_level_header:
            _copy_cards(self.top_level_header)

        # Then sensor header
        if sensor_hdr:
            _copy_cards(sensor_hdr)

        return combined

    # --- nice representation ---------------------------------------------
    def __repr__(self):
        sensors = ", ".join(sorted(self.keys()))
        src = self.filename or "<in-memory>"
        return f"WinterImage({src}, sensors=[{sensors}])"
