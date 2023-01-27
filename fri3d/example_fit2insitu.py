# pylint: disable=E1101
# pylint: disable=C0103
from bdb import Breakpoint
import calendar
from multiprocessing.resource_sharer import stop
import os
import sys
from datetime import datetime

import numpy as np
from ai import cdas, cs
from ai.fri3d.optimize import PolyProfile, SignProfile, fit2insitu
from astropy import units as u
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

path1 = os.getcwd()

# define nT unit for unit transformations
u.nT = u.def_unit("nT", 1e-9 * u.T)

# enable CDAS data cache

#asd=os.path.join(os.path.dirname(__file__), "data")
cdas.set_cache(True, os.path.join(os.path.dirname(__file__), "data"))          

# get magnetic field data from CDAS (https://cdaweb.gsfc.nasa.gov/misc/NotesW.html)
data_mag = cdas.get_data(
    "sp_phys", "WI_H0_MFI", datetime(2008, 12, 17), datetime(2008, 12, 18), ["BF1", "BGSE"], cdf=False)  
# "WI_H0_MFI" = Wind Magnetic Fields Investigation: 3 sec, 1 min, and hourly Definitive Data
# Magnetic field vector in GSE cartesian coordinates (1 min) [BGSE]
# Magnetic field magnitude (1 min) [BF1]

# get plasma data from CDAS
data_pla = cdas.get_data(
    "sp_phys", "WI_K0_SWE", datetime(2008, 12, 17), datetime(2008, 12, 18), ["V_GSE"], cdf=False) 
# "WI_K0_SWE" = Wind Solar Wind Experiment, Key Parameters
# Solar Wind Velocity in GSE coord., 3 comp. [V_GSE]
#breakpoint()
# filter the magnetic field data
m = np.logical_and(data_mag["EPOCH"] >= datetime(2008, 12, 17, 3, 30), data_mag["EPOCH"] <= datetime(2008, 12, 17, 14, 30))
d = data_mag["EPOCH"][m]
t = np.asarray([calendar.timegm(x.timetuple()) for x in d])
#breakpoint()
#b = data_mag["BGSE"][m]
b = data_mag["B"][m]
b[b == -1e31] = np.nan

bx0=data_mag['BX_(GSE)'][m]
by0=data_mag['BY_(GSE)'][m]
bz0=data_mag['BZ_(GSE)'][m]

# convert magnetic field data from GSE to HEEQ coordinates
#bx0, by0, bz0 = cs.cxform("GSE", "HEEQ", d, np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d)))
#bx1, by1, bz1 = cs.cxform("GSE", "HEEQ", d, np.ravel(b[:, 0]), np.ravel(b[:, 1]), np.ravel(b[:, 2]))
#b = np.array([bx1 - bx0, by1 - by0, bz1 - bz0]).T
bx, by, bz = cs.cxform("GSE", "HEEQ", d, bx0,by0,bz0)
b = np.array([bx,by,bz]).T
#print(b)
#print(data_pla["VX_(GSE)"])
print(data_pla["VY_(GSE)"].size)

# filter plasma speed data
#data_pla["VX_(GSE)"][np.any(data_pla["VX_(GSE)"] == -1e31, axis=0)] = np.nan
#data_pla["VY_(GSE)"][np.any(data_pla["VY_(GSE)"] == -1e31, axis=0), :] = np.nan
#data_pla["VZ_(GSE)"][np.any(data_pla["VZ_(GSE)"] == -1e31)] = np.nan
for x in data_pla["VY_(GSE)"]:
  if data_pla["VY_(GSE)",x] == -1e31:
    x = np.nan

print(data_pla["VY_(GSE)"].size)
# construct an interpolator for the plasma data
f = interp1d(
    np.asarray([calendar.timegm(x.timetuple()) for x in data_pla["EPOCH"]]),
    np.sqrt(data_pla["VX_(GSE)"] ** 2 + data_pla["VY_(GSE)"] ** 2 + data_pla["VZ_(GSE)"] ** 2),
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
print('hecho')
plt.savefig(path1 + '/output/fri3d_example_insitu.png')