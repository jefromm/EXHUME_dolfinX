#!/bin/sh 

# this script is to prepare a dolfinx docker container to run EXHUME_X

# EXHUME_X uses h5py to read in extraction operators. to ensure compatibility, run
pip install --no-binary=h5py h5py meshio netCDF4



