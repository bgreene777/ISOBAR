#!/home/bgreene/anaconda3/bin/python
# --------------------------------
# Name: preprocess.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 27 June 2023
# Purpose: perform corrections to 20 Hz netcdf daily files such as:
# sonic double rotation, spike removal, interpolate missing values
# --------------------------------
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from glob import glob

def spike_removal(ds, dt=np.timedelta64(10, "m")):
    """Use median absolute deviation filter to remove outliers
    :param Dataset ds: xarray timeseries dataset for analyzing; by default\
        apply to all variables in ds
    :param timedelta64 dt: window for calculating medians over; default=10 min
    returns Dataset
    """
    # create array of time bins for looping
    time = ds.time.values
    tbin = np.arange(time[0], time[-1]+dt, dt)
    # constants for median filtering
    q = 7.
    const = 0.6745
    # loop over timesteps
    for jt in range(1, len(tbin)):
        jt_use = np.where((time >= tbin[jt-1]) & (time < tbin[jt]))[0]
        # loop over variables
        for v in ds.keys():
            # calculate median over period
            med = ds[v][jt_use].median(skipna=True)
            # calculate mean absolute deviation
            MAD = np.abs(ds[v][jt_use] - med)
            # calculate hi and lo bounds
            lo = med - (q*MAD/const)
            hi = med + (q*MAD/const)
            # search for values outside the range lo-hi
            jout = np.where((ds[v][jt_use] < lo) | (ds[v][jt_use] > hi))[0]
            # mask values if jout not empty
            if len(jout) > 0:
                ds[v][jt_use][jout] = np.nan
    # add attrs to ds
    ds.attrs["despike"] = "True"
    ds.attrs["despike_dt"] = dt
    return ds
# --------------------------------
def double_rotate(ds, dt=np.timedelta64(10, "m"), nlevel=3):
    """Double coordinate rotation on 3D wind components measured in sonic
    anemometer reference frame. Based on equations 22--29 in 
    Wilczak et al. (2001), BLM
    :param Dataset ds: xarray timeseries dataset for analyzing
    :param timedelta64 dt: window for computing individual runs; default=10min
    :param int nlevel: how many levels of sonic anemometers to loop over;\
        default=3
    return Dataset with rotated coords
    """
    # begin by defining array of time bins
    time = ds.time.values
    tbin = np.arange(time[0], time[-1]+dt, dt)
    # begin loop over levels
    for jl in range(nlevel):
        # grab measured values: um, vm, wm
        um, vm, wm = ds[f"u{jl+1}"], ds[f"v{jl+1}"], ds[f"w{jl+1}"]
        # create long arrays to fill with final rotated runs
        urf, vrf, wrf = [np.zeros(ds.time.size, dtype=np.float64) for _ in range(3)]
        # loop over time bins and perfrom double rotation
        for jt in range(1, len(tbin)):
            jt_use = np.where((time >= tbin[jt-1]) & (time < tbin[jt]))[0]
            # ROTATION 1: <v> = 0
            # bin average um and vm; don't need w yet
            um_bar = um[jt_use].mean(skipna=True)
            vm_bar = vm[jt_use].mean(skipna=True)
            angle1 = np.arctan2(vm_bar, um_bar)
            # compute rotated ur1, vr1, wr1
            ur1 = um[jt_use]*np.cos(angle1) + vm[jt_use]*np.sin(angle1)
            vr1 =-um[jt_use]*np.sin(angle1) + vm[jt_use]*np.cos(angle1)
            wr1 = wm[jt_use]
            # ROTATION 2: <w> = 0
            # bin average ur1 and wr1; don't need vr1 this time
            ur1_bar = ur1.mean(skipna=True)
            wr1_bar = wr1.mean(skipna=True)
            angle2 = np.arctan2(wr1_bar, ur1_bar)
            # compute final rotated values and store in same line
            urf[jt_use] = ur1*np.cos(angle2) + wr1*np.sin(angle2)
            vrf[jt_use] = vr1
            wrf[jt_use] =-ur1*np.sin(angle2) + wr1*np.cos(angle2)
        # outside of time loop; covert to DataArrays and store back in ds
        ds[f"u{jl+1}r"] = xr.DataArray(data=urf, coords=dict(time=ds.time), 
                                       attrs=ds[f"u{jl+1}"].attrs)
        ds[f"v{jl+1}r"] = xr.DataArray(data=vrf, coords=dict(time=ds.time), 
                                       attrs=ds[f"v{jl+1}"].attrs)
        ds[f"w{jl+1}r"] = xr.DataArray(data=wrf, coords=dict(time=ds.time), 
                                       attrs=ds[f"w{jl+1}"].attrs)
    # finished looping
    # add some global attrs
    ds.attrs["rotate"] = "True"
    ds.attrs["rotate_dt"] = dt
    return ds