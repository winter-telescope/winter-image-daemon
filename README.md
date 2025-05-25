# WINTER Image Daemon Repository

This is a package which manages a daemon which handles image analysis operations for all cameras on the WINTER observatory. It maintains basic capabilities that are needed across cameras, such as focus, basic calibration, and the ability to run astrometry for building sky models for the telescope pointing.

## Getting the python environment set up

Try this:

1. conda update -n base -c defaults conda
2. go to top level `winter-image-daemon` directory
3. Make a new conda environment in this directory: `conda create --prefix .conda python=3.11`
4. Activate the new environment in this directory: `conda activate ./.conda`
5. Update pip: `pip install --upgrade pip`
6. Install dependencies: `pip install -e .` Alternately if you want to install the dev dependencies: `pip install -e ".[dev]"`

## You will need the following packages - Required :
Will need:
- astrometry.net and index files from: from https://portal.nersc.gov/project/cosmo/
- Astromatic swarp, scamp, source extractor 

There are more detailed instructions here: is to follow the instructions for these here: [mirar installation instructions](https://mirar.readthedocs.io/en/latest/installation.html)


## Required Data
The data reductions require sets of standard calibration data.

## Running the daemon
The daemon can be run on any number of machines, to handle any number of cameras.

A daemon instance is launched like: 

"""python:
imagedaemon-daemon --cameras winter,qcmos --logfile /var/log/daemon.log`
"""

