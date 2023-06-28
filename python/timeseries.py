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
# --------------------------------
def Richardson(ts_ec, ts_T, resolution="30min"):
    """Purpose: calculate gradient Richardson number based on profiles of
    vector wind and temperature from sonic anemometers and thermocouples, 
    respectively, and return array of values at resolution of ts_T 
    (e.g., 10/30/60 min) for each sonic level.
    :param Dataset ts_ec: high-frequency sonic anemometer data at\
        multiple levels. Variables should be named u1, u2, ..., v1,\
        v2, ... and include attrs for corresponding z1, z2, etc.
    :param Dataset ts_T: 'slow'-response temperature data at lower\
        temporal resolution than ts_ec. Variables should be named\
        T1, T2, ... and include attrs for corresponding z1, z2, etc.
    :param str resolution: averaging interval to consider with syntax\
        for xarray.Dataset.resample()
    return Dataset of Ri_g at levels ts_ec.z1, etc.
    """
    # begin by resampling ec and T at desired resolution
    # this will also align timeseries to similar time vector
    ts_ec_m = ts_ec.resample(time=resolution).mean(skipna=True)
    ts_T_m = ts_T.resample(time=resolution).mean(skipna=True)
    # line up ec and T arrays by only grabbing T in bounds of ec
    ts_T_m = ts_T_m.where((ts_T_m.time >= ts_ec_m.time[0]) &\
                          (ts_T_m.time <= ts_ec_m.time[-1]), 
                          drop=True)
    # grab these new time vectors
    time = ts_ec_m.time
    # grab dimensions
    nz_ec = sum(["z" in key for key in ts_ec.attrs.keys()])
    nz_T = sum(["z" in key for key in ts_T.attrs.keys()])
    nt = ts_ec_m.time.size # this should be the same!
    # constants
    g = 9.81 # m s^-2
    # make arrays of heights
    z_ec = np.zeros(nz_ec+1, dtype=np.float64)
    z_T = np.zeros(nz_T, dtype=np.float64)
    # set roughness length z_ec[0] = z_0 st u(z=z_0)=0, v(z=z_0)=0
    z_ec[0] = 0.001
    # reshape variables into single 2D arrays shape [nz, nt]
    u_all, v_all = [np.zeros((nz_ec+1, nt), dtype=np.float64) for _ in range(2)]
    T_all = np.zeros((nz_T, nt), dtype=np.float64)
    # fill ec data arrays
    for jz in range(nz_ec):
        # u_all and v_all first level all zeros
        u_all[jz+1,:] = ts_ec_m[f"u{jz+1}"].to_numpy()
        v_all[jz+1,:] = ts_ec_m[f"v{jz+1}"].to_numpy()
        z_ec[jz+1] = ts_ec_m.attrs[f"z{jz+1}"]
    # fill T data array
    for jz in range(nz_T):
        T_all[jz,:] = ts_T_m[f"T{jz+1}"].to_numpy()
        z_T[jz] = ts_T_m.attrs[f"z{jz+1}"]

    # define helper function to calculate gradients
    def calc_ddz(z, b, c):
        return (b + 2*c*np.log(z)) / z
    
    # define empty arrays to fill
    Rig_all, dudz_all, dvdz_all, dTdz_all =\
        [np.zeros((nz_ec, nt), dtype=np.float64) for _ in range(4)]
    
    # loop over time ranges - match those from ec
    for jt, tt in enumerate(time):
        # grab u, v, T using jt_ec
        u = u_all[:,jt]
        v = v_all[:,jt]
        T = T_all[:,jt]
        # case 1: missing value at any height for any param: cannot continue :(
        if ((sum(np.isnan(u))>0) | (sum(np.isnan(v))>0) | (sum(np.isnan(T))>0)):
            Rig_all[:,jt] = np.nan
            dudz_all[:,jt] = np.nan
            dvdz_all[:,jt] = np.nan
            dTdz_all[:,jt] = np.nan
        # case 2: use np.polyfit to fit each profile
        else:
            u_fit = np.polyfit(np.log(z_ec), u, deg=2)
            v_fit = np.polyfit(np.log(z_ec), v, deg=2)
            T_fit = np.polyfit(np.log(z_T), T, deg=2)
            # use helper functions to calculate gradients at height of sonics
            du_dz = calc_ddz(z_ec[1:], u_fit[1], u_fit[0])
            dv_dz = calc_ddz(z_ec[1:], v_fit[1], v_fit[0])
            dT_dz = calc_ddz(z_ec[1:], T_fit[1], T_fit[0])
            # calculate num and denom of Ri_g: N2, S2
            beta = g / (T[0] + 273.15) # T0 in Kelvin
            dtheta_dz = dT_dz + 0.01 # correction for potential temp gradient
            # store gradients
            dudz_all[:,jt] = du_dz
            dvdz_all[:,jt] = dv_dz
            dTdz_all[:,jt] = dtheta_dz
            N2 = beta * dtheta_dz
            S2 = (du_dz ** 2.) + (dv_dz ** 2.)
            Rig_all[:,jt] = N2 / S2

    # OUTSIDE TIME LOOP
    # convert Rig_all and everything else to xarray Dataset to return
    Ri = xr.Dataset(data_vars=None, coords=dict(z=z_ec[1:], time=time),
                    attrs=(ts_ec.attrs | ts_T.attrs))
    Ri["Rig"] = xr.DataArray(data=Rig_all, coords=dict(z=z_ec[1:], time=time))
    Ri["du_dz"] = xr.DataArray(data=dudz_all, coords=dict(z=z_ec[1:], time=time))
    Ri["dv_dz"] = xr.DataArray(data=dvdz_all, coords=dict(z=z_ec[1:], time=time))
    Ri["dT_dz"] = xr.DataArray(data=dTdz_all, coords=dict(z=z_ec[1:], time=time))

    return Ri