import logging
import os
import sys

import numpy as np

# Set up the logger before importing any other modules
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logging.info("Starting build_calibration_masters.py")

from imagedaemon.calibration.master_frame_builder import (
    build_master_frames,
    intersect_headers,
    scan_directory,
)
from imagedaemon.image.image_operations import median_combine_images
from imagedaemon.image.winter_image import WinterImage
from imagedaemon.paths import CAL_DATA_DIR, RAW_DATA_DIR

# --------------------------------------------------------------------------
# Build master frames for WINTER camera
# --------------------------------------------------------------------------
camname = "winter"

# DARKS
dark_raw_dir = os.path.join(RAW_DATA_DIR, camname, "darks")
dark_cal_dir = os.path.join(CAL_DATA_DIR, camname, "masterdarks")

print(f"dark_raw_dir = {dark_raw_dir}")
print(f"dark_cal_dir = {dark_cal_dir}")

# build the master darks for all exposure times in the raw directory
build_master_frames(
    src_dir=dark_raw_dir,
    dst_dir=dark_cal_dir,
    obstype="DARK",
    camera_name=camname,
)

# BIAS
bias_raw_dir = os.path.join(RAW_DATA_DIR, camname, "bias")
bias_cal_dir = os.path.join(CAL_DATA_DIR, camname, "masterbias")
build_master_frames(
    src_dir=bias_raw_dir,
    dst_dir=bias_cal_dir,
    obstype="BIAS",
    camera_name=camname,
)
# FLATS
flat_raw_dir = os.path.join(RAW_DATA_DIR, camname, "flats", "labflats")
flat_cal_dir = os.path.join(CAL_DATA_DIR, camname, "masterflats")

# this one is a little different, we need to make master flats for each detector,
# and each image is *not* an MEF we can import into the WinterImage class
# so we need to use the scan_directory function to get the list of files
flat_data = {}
flat_header = {}
flat_top_level_headers = {}
for addr in ["sa", "sb", "sc", "pa", "pb", "pc"]:
    flat_raw_light_dir = os.path.join(flat_raw_dir, addr, "light")
    flat_raw_dark_dir = os.path.join(flat_raw_dir, addr, "dark")

    light_data_df = scan_directory(flat_raw_light_dir)
    light_filepaths = light_data_df.filepath.tolist()
    sensor_median_light = median_combine_images(light_filepaths)

    dark_data_df = scan_directory(flat_raw_dark_dir)
    dark_filepaths = dark_data_df.filepath.tolist()
    sensor_median_dark = median_combine_images(dark_filepaths)

    sensor_median_darkcorrected = sensor_median_light - sensor_median_dark
    sensor_median_flat = sensor_median_darkcorrected / np.nanmedian(
        sensor_median_darkcorrected
    )
    flat_data[addr] = sensor_median_flat

    # now we need to get the headers for the light and dark frames
    # for these lab frames, there's only a top level header
    light_headers = light_data_df.top_level_header.tolist()
    dark_headers = dark_data_df.top_level_header.tolist()
    # print(f"light_headers: {light_headers}")
    flat_header[addr] = intersect_headers(light_headers + dark_headers)
    # print(f"flat_header[addr]: {flat_header[addr]}")


flat_top_level_header = intersect_headers(
    [flat_header[addr] for addr in flat_header.keys()]
)

# debug: what is the flat_top_level_header?
print(f"type(flat_top_level_header) = {type(flat_top_level_header)}")
# print(f"flat_top_level_header: {flat_top_level_header}")

print(f"type(flat_header['pa']) = {type(flat_header['pa'])}")
# print(f"flat_header[pa]: {flat_header['pa']}")
# make a WinterImage object for all sensors
master_flat = WinterImage(
    data=flat_data, headers=flat_header, top_level_header=flat_top_level_header
)
# master_flat.plot_mosaic(title="Master Lab Flat")
master_flat.save_mef(os.path.join(flat_cal_dir, "masterflat.fits"))
