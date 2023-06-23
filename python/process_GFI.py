#!/home/bgreene/anaconda3/bin/python
# --------------------------------
# Name: process_GFI.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 23 June 2023
# Purpose: load raw 20Hz txt file produced by MATLAB script from dat file,
# convert into Datasets, and save netcdf files grouped by day for later
# --------------------------------
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

# load huge text file
fdir = "/Users/briangreene/Nextcloud/Projects/ISOBAR/Hailuoto_2018/Flux2_GFI/20Hz/"
fload = f"{fdir}GFI2EC_I2_20Hz.txt"
print(f"Loading file: {fload}")
data = np.loadtxt(fload, dtype=np.float64, delimiter=",")

# convert timestamps into numpy datetime64 format
timestamp = np.array(data[:,0]*1000, dtype="datetime64[ms]")

# initialize Dataset
dfull = xr.Dataset()

# variable names for saving, in order of text file columns
var = ["u1", "u2", "u3", "v1", "v2", "v3", "w1", "w2", "w3", "T1", "T2", "T3"]

# loop over each variable and store data in dfull
for i, v in enumerate(var):
    # note: use i+1 since i=0 is timestamp column
    print(f"Grabbing data: {v}")
    dfull[v] = xr.DataArray(data=data[:,i+1],
                            dims="time",
                            coords=dict(time=timestamp))
    # add units
    # all velocities are m/s, temps are in degC
    if i <= 8:
        dfull[v].attrs["units"] = "m/s"
    else:
        dfull[v].attrs["units"] = "degC"
# add some global attrs
dfull.attrs["z1"] = 1.97
dfull.attrs["z2"] = 4.55
dfull.attrs["z3"] = 10.31

# data are loaded and arranged, now loop over days and save out netcdf files
# use dayofyear from 36 to 56+1 based on data
dayofyear = np.arange(36, 57, 1)
for dy in dayofyear:
    # grab day
    dtoday = dfull.isel(time=(dfull.time.dt.dayofyear==dy))
    # grab timestamp to create filename string
    tt0 = dtoday.time[0].values
    stt0 = np.datetime_as_string(tt0, "D")
    fsave = f"{fdir}GFI2EC_I2_20Hz_{stt0}.nc"
    # save out
    print(f"Saving file: {fsave}")
    with ProgressBar():
        dtoday.to_netcdf(fsave, mode="w")