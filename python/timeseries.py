#!/home/bgreene/anaconda3/bin/python
# --------------------------------
# Name: preprocess.py
# Author: Brian R. Greene
# University of Oklahoma
# Created: 27 June 2023
# Purpose: perform corrections to 20 Hz netcdf daily files such as:
# sonic double rotation, spike removal, interpolate missing values
# --------------------------------
import xrft
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from glob import glob
from scipy import linalg

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
def double_rotate(ds, jl):
    """Double coordinate rotation on 3D wind components measured in sonic
    anemometer reference frame. Based on equations 22--29 in 
    Wilczak et al. (2001), BLM.
    Designed to be used with xarray groupby/resample syntax
    :param Dataset ds: xarray timeseries dataset for analyzing
    :param int jl: level number for use in indexing
    return Dataset with rotated coords
    """
    # grab measured values: um, vm, wm
    um, vm, wm = ds[f"u{jl}"], ds[f"v{jl}"], ds[f"w{jl}"]
    # ROTATION 1: <v> = 0
    # bin average um and vm; don't need w yet
    um_bar = um.mean(dim="time", skipna=True)
    vm_bar = vm.mean(dim="time", skipna=True)
    angle1 = np.arctan2(vm_bar, um_bar)
    # compute rotated ur1, vr1, wr1
    ur1 = um*np.cos(angle1) + vm*np.sin(angle1)
    vr1 =-um*np.sin(angle1) + vm*np.cos(angle1)
    wr1 = wm
    # ROTATION 2: <w> = 0
    # bin average ur1 and wr1; don't need vr1 this time
    ur1_bar = ur1.mean(dim="time", skipna=True)
    wr1_bar = wr1.mean(dim="time", skipna=True)
    angle2 = np.arctan2(wr1_bar, ur1_bar)
    # compute final rotated values and store in same line
    urf = ur1*np.cos(angle2) + wr1*np.sin(angle2)
    vrf = vr1
    wrf =-ur1*np.sin(angle2) + wr1*np.cos(angle2)
    # store back in ds
    ds[f"u{jl}r"] = urf
    ds[f"v{jl}r"] = vrf
    ds[f"w{jl}r"] = wrf    
    # finished
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
    ts_ec_m = ts_ec.resample(time=resolution, closed="left").mean(skipna=True)
    ts_T_m = ts_T.resample(time=resolution, closed="left").mean(skipna=True)
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
                    attrs=(ts_ec.attrs | dict(nz=np.shape(Rig_all)[0])))
    Ri["Rig"] = xr.DataArray(data=Rig_all, coords=dict(z=z_ec[1:], time=time))
    Ri["du_dz"] = xr.DataArray(data=dudz_all, coords=dict(z=z_ec[1:], time=time))
    Ri["dv_dz"] = xr.DataArray(data=dvdz_all, coords=dict(z=z_ec[1:], time=time))
    Ri["dT_dz"] = xr.DataArray(data=dTdz_all, coords=dict(z=z_ec[1:], time=time))

    return Ri
# --------------------------------
def anisotropy(ts, jl):
    # check for nans, return nan if so
    if sum(np.isnan(ts[f"u{jl}r"])) > 0:
        return xr.DataArray(data=np.nan*np.zeros(3, dtype=np.float64),
                            coords=dict(eigenvalue=np.arange(3)))
    # detrend u, v, w
    ud = xrft.detrend(ts[f"u{jl}r"], dim="time", detrend_type="linear").to_numpy()
    vd = xrft.detrend(ts[f"v{jl}r"], dim="time", detrend_type="linear").to_numpy()
    wd = xrft.detrend(ts[f"w{jl}r"], dim="time", detrend_type="linear").to_numpy()
    # assemble 3-component velocity list
    V = [ud, vd, wd]
    # calculate 2*TKE = <u_i' u_i'>
    ee = np.nanmean(ud*ud) + np.nanmean(vd*vd) + np.nanmean(wd*wd)
    # initialize empty bij array
    bij = np.zeros((3,3), dtype=np.float64)
    # loop over i, j and construct bij tensor
    for i in range(3):
        for j in range(3):
            bij[i,j] = np.nanmean(V[i]*V[j]) / ee
    # subtract off 1/3 from trace
    bij -= np.identity(3)*(1./3)
    # calculate eigenvalues of bij, sort in descending order
    ll = np.sort(linalg.eig(bij)[0].real)[::-1]
    # convert eigenvalues into barycentric invariants
    # xB = ll[0] - ll[1] + 0.5*(3*ll[2] + 1)
    # yB = (np.sqrt(3)/2) * (3*ll[2] + 1)

    return xr.DataArray(data=ll, coords=dict(eigenvalue=np.arange(3)))

# --------------------------------
# Main
# --------------------------------
if __name__ == "__main__":
    # load some files
    fdata = "/home/bgreene/ISOBAR/data/"
    # 1s GFI2
    fGFI2_1s = f"{fdata}GFI2_AWS/GFI2AWS_l3_1sec_20180205-25.nc"
    GFI2_1s = xr.load_dataset(fGFI2_1s)
    # make sure starts and ends on multiple of 30 min
    # will hardcode this since start/end times known
    GFI2_t0 = np.datetime64("2018-02-05T11:29:59.999")
    GFI2_tf = np.datetime64("2018-02-25T05:30:00.000000000")
    GFI2_1s = GFI2_1s.where((GFI2_1s.time >= GFI2_t0) &\
                            (GFI2_1s.time <= GFI2_tf), drop=True)
    # shape into Dataset for use with Richardson function
    ts_T = xr.Dataset()
    ts_T["T1"] = GFI2_1s["ta_1m_2"]
    ts_T["T2"] = GFI2_1s["ta_2m_2"]
    ts_T["T3"] = GFI2_1s["ta_7m_2"]
    ts_T.attrs["z1"] = GFI2_1s.ta_1m_2.height
    ts_T.attrs["z2"] = GFI2_1s.ta_2m_2.height
    ts_T.attrs["z3"] = GFI2_1s.ta_7m_2.height
    # 20 Hz sonic data
    fec = f"{fdata}GFI2EC_20Hz/"
    fec_all = np.sort(glob(fec+"*.nc"))
    # grab sampling rate as timedelta
    # 20 Hz = 1/20 sec = 0.05 sec = 50 ms
    t_sample = np.timedelta64(50, "ms")
    # loop over files/days to compute statistics
    for jf, ff in enumerate(fec_all[:-1]):
        # load
        print(f"Loading file: {ff}")
        d = xr.load_dataset(ff)
        # line up ec data with processed temperature data 
        # (only applies to first and last days)
        if jf == 0:
            ts_T_t0 = ts_T.time[0]
            d = d.where(d.time >= ts_T_t0, drop=True)
        elif jf == len(fec_all) - 1:
            ts_T_tf = ts_T.time[-1]
            d = d.where(d.time <= ts_T_tf, drop=True)
        # despike
        print("Spike removal and gap filling")
        dspike = spike_removal(d)
        # fill gaps
        dfill = dspike.interpolate_na(dim="time", method="linear",
                                      use_coordinate=True, limit=10)
        # compute Richardson number profiles
        print("Compute Ri profiles - 30 and 10 min")
        Ri = Richardson(dfill, ts_T, resolution="30min")
        # also compute 10-min Ri profiles for comparisons later
        Ri_10min = Richardson(dfill, ts_T, resolution="10min")
        # initialize dictionary of counters for each level
        count = {}
        # initialize dictionary of labels for each level for grouping
        labels = {}
        for jz in range(Ri.nz):
            count[f"z{jz+1}"] = 0
            labels[f"z{jz+1}"] = []
        # loop over individual blocks of Ri
        # if stable (>0) then compute runs of 10 min increments
        # if unstable (<0) then compute runs of 30 min
        # recall: time array in Ri is made using "resample", so blocks
        # start at left index instead of centered
        # make array of times
        print("Analyze segments for stability")
        dtRi = np.timedelta64(30, "m")
        times = np.arange(Ri.time[0].values, Ri.time[-1].values+2*dtRi, dtRi)
        for jt in range(Ri.time.size):
            t0 = times[jt]
            t1 = times[jt+1]
            # grab dfill between t0 and t1
            dfill_use = dfill.where((dfill.time >= t0) & (dfill.time < t1),
                                     drop=True)
            # loop over Ri heights
            for jz in range(Ri.nz):
                # check stability
                if Ri.Rig.isel(z=jz, time=jt).values > 0:
                    # resample dfill_use to 10-min and rotate coords
                    dt = np.timedelta64(10, "m")
                    ngroup = 3
                else:
                    # resample dfill_use to 30-min and rotate coords
                    dt = np.timedelta64(30, "m")
                    ngroup = 1
                # calc number of points in each group based on dt and t_sample
                # append that many labels to running labels/count lists
                # increment count for each group
                nlab = int(dt / t_sample)
                for n in range(ngroup):
                    labels[f"z{jz+1}"] += ([count[f"z{jz+1}"]] * nlab)
                    count[f"z{jz+1}"] += 1

        # apply labels to dfill
        # loop over jz
        for jz in range(Ri.nz):
            newlab = ("time", np.array(labels[f"z{jz+1}"]))
            dfill = dfill.assign_coords({f"lab{jz+1}": newlab})
        print("Rotate sonic coordinates")
        # rotate coordinates within each group at each height
        drot = xr.Dataset() # clear drot from last iteration
        for jz in range(Ri.nz):
            # create new drot on first jz
            if jz == 0:
                drot = dfill.groupby(f"lab{jz+1}").map(double_rotate, jl=jz+1)
            else:
                drot = drot.groupby(f"lab{jz+1}").map(double_rotate, jl=jz+1)
        print("Compute anisotropy tensor eigenvalues")
        # calculate anisotropy tensor eigenvalues at each height
        # create Dataset of these values
        ll_all = xr.Dataset()
        for jz in range(Ri.nz):
            # store as temp variable because will modify later with long syntax
            temp = drot.groupby(f"lab{jz+1}").map(anisotropy, jl=jz+1)
            # add time back as coordinates for each level
            # grab time from drot as first from each group
            jtime = drot.time.groupby(f"lab{jz+1}").first()
            # convert into a coordinate in ll_all
            # create new coord f"time{jz+1}" from f"lab{jz+1}" that is based on jtime 
            temp2 = temp.assign_coords({f"time{jz+1}": (f"lab{jz+1}", jtime)})
            # swap dims then store in ll_all
            ll_all[f"z{jz+1}"] = temp2.swap_dims({f"lab{jz+1}": f"time{jz+1}"})

        # save out files
        st0 = np.datetime_as_string(times[0], "D")
        # save 30 min Ri profiles
        fsave_Ri = f"{fdata}Ri/Ri_30min_{st0}.nc"
        print(f"Saving file: {fsave_Ri}")
        Ri.to_netcdf(fsave_Ri, "w")
        # save 10 min Ri profiles
        fsave_Ri_10 = f"{fdata}Ri/Ri_10min_{st0}.nc"
        print(f"Saving file: {fsave_Ri_10}")
        Ri_10min.to_netcdf(fsave_Ri_10, "w")
        # save anisotropy tensor eigenvalues
        fsave = f"{fdata}anisotropy/bij_eigenvalues_{st0}.nc"
        print(f"Saving file: {fsave}")
        ll_all.to_netcdf(fsave, "w")