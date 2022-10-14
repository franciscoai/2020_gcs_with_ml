# pylint: disable=E1101
# pylint: disable=C0103
import calendar
import os
import sys
from datetime import datetime

import numpy as np
from ai import cdas, cs
from ai.fri3d.optimize import PolyProfile, SignProfile, fit2insitu
from astropy import units as u
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

# define nT unit for unit transformations
u.nT = u.def_unit("nT", 1e-9 * u.T)

# enable CDAS data cache
cdas.set_cache(True, os.path.join(os.path.dirname(__file__), "data"))

# get magnetic field data from CDAS
data_mag = cdas.get_data(
    "sp_phys", "WI_H0_MFI", datetime(2008, 12, 17), datetime(2008, 12, 18), ["BF1", "BGSE"], cdf=False
)

# get plasma data from CDAS
data_pla = cdas.get_data("sp_phys", "WI_K0_SWE", datetime(2008, 12, 17), datetime(2008, 12, 18), ["V_GSE"], cdf=False)
breakpoint()
# filter the magnetic field data
m = np.logical_and(
    data_mag["Epoch"] >= datetime(2008, 12, 17, 3, 30), data_mag["Epoch"] <= datetime(2008, 12, 17, 14, 30)
)
d = data_mag["Epoch"][m]
t = np.asarray([calendar.timegm(x.timetuple()) for x in d])
b = data_mag["BGSE"][m]
b[b == -1e31] = np.nan

# convert magnetic field data from GSE to HEEQ coordinates
bx0, by0, bz0 = cs.cxform("GSE", "HEEQ", d, np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d)))
bx1, by1, bz1 = cs.cxform("GSE", "HEEQ", d, np.ravel(b[:, 0]), np.ravel(b[:, 1]), np.ravel(b[:, 2]))
b = np.array([bx1 - bx0, by1 - by0, bz1 - bz0]).T

# filter plasma speed data
data_pla["V_GSE"][np.any(data_pla["V_GSE"] == -1e31, axis=1), :] = np.nan
# construct an interpolator for the plasma data
f = interp1d(
    np.asarray([calendar.timegm(x.timetuple()) for x in data_pla["Epoch"]]),
    np.sqrt(data_pla["V_GSE"][:, 0] ** 2 + data_pla["V_GSE"][:, 1] ** 2 + data_pla["V_GSE"][:, 2] ** 2),
    kind="linear",
    bounds_error=False,
    fill_value=np.nan,
)
# approximate plasma data for the same timestamps as magnetic field
vt = f(t)

# convert magnetic field from nT to T
b = u.nT.to(u.T, b)
# convert plasma speed from km/s to m/s
vt = u.Unit("km/s").to(u.Unit("m/s"), vt)

# create a mask to choose randomly 500 points from the fitted data
m = np.sort(np.random.choice(t.size, 500, replace=False))

# fit the FRi3D model to in-situ data
dfr, profiles = fit2insitu(
    u.au.to(u.m, 1),
    0,
    0,
    t[m],
    b[m, :],
    vt[m],
    latitude=PolyProfile(bounds=[u.deg.to(u.rad, [-30, 30]).tolist()]),
    longitude=PolyProfile(bounds=[u.deg.to(u.rad, [-30, 30]).tolist()]),
    toroidal_height=PolyProfile(
        bounds=[u.Unit("km/s").to(u.Unit("m/s"), [300, 500]).tolist(), u.au.to(u.m, [0.8, 1.2]).tolist()]
    ),
    poloidal_height=PolyProfile(bounds=[u.au.to(u.m, [0.01, 0.2]).tolist()]),
    half_width=PolyProfile(bounds=[u.deg.to(u.rad, [30, 60]).tolist()]),
    tilt=PolyProfile(bounds=[u.deg.to(u.rad, [-30, 30]).tolist()]),
    flattening=PolyProfile(bounds=[[0.2, 0.8]]),
    pancaking=PolyProfile(bounds=[u.deg.to(u.rad, [20, 40]).tolist()]),
    skew=PolyProfile(params=[0]),
    twist=PolyProfile(bounds=[[0, 3]]),
    flux=PolyProfile(bounds=[[1e13, 1e15]]),
    sigma=PolyProfile(params=[2]),
    polarity=SignProfile(bounds=[[-1, 1]]),
    chirality=SignProfile(bounds=[[-1, 1]]),
    verbose=True,
)
