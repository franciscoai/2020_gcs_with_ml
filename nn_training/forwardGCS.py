#!/usr/bin/env python
# coding: utf-8

# ## Librerias
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pyGCS_raytrace import pyGCS
from pyGCS_raytrace.rtraytracewcs import rtraytracewcs
import numpy as np
import datetime
import struct
import matplotlib.pyplot as plt
from scipy.stats import kde
from astropy.io import fits
import sunpy
from sunpy.coordinates.ephemeris import get_horizons_coord
import sunpy.map
import pandas as pd
from sunpy.sun.constants import radius as _RSUN

## Función ajuste del centro del sol
def center_rSun_pixel(headers, plotranges, sat):
    x_cS = (headers[sat]['CRPIX1']*plotranges[sat][sat]*2) / \
        headers[sat]['NAXIS1'] - plotranges[sat][sat]
    y_cS = (headers[sat]['CRPIX2']*plotranges[sat][sat]*2) / \
        headers[sat]['NAXIS2'] - plotranges[sat][sat]
    return x_cS, y_cS

ISSIflag = True # flag if using LASCO data from ISSI which has STEREOlike headers already
exec_path = os.getcwd()
DATA_PATH = '/gehme/data'
OPATH = exec_path + '/../output/' #'/gehme/projects/2020_gcs_with_ml/data/forwardGCS_test'
# Files to get headers
secchipath = DATA_PATH + '/stereo/secchi/L0'
lascopath = DATA_PATH + '/soho/lasco/level_1/c2'
CorA = secchipath + '/a/img/cor2/20101214/level1/20101214_162400_04c2A.fts'
CorB = secchipath + '/b/img/cor2/20101214/level1/20101214_162400_04c2B.fts'
LascoC2 = None # lascopath + '/20101214/25354684.fts'
# synthetic coronograph image additions
cor2 = 2 # Tamaño de los occulters referenciados al RSUN
c3 = 3.7 # Tamaño de los occulters referenciados al RSUN


## main
os.makedirs(OPATH, exist_ok=True)
# STEREO A
ima2, hdra2 = sunpy.io._fits.read(CorA)[0]
smap_SA2 = sunpy.map.Map(ima2, hdra2)
# STEREO B
imb2, hdrb2 = sunpy.io._fits.read(CorB)[0]
smap_SB2 = sunpy.map.Map(imb2, hdrb2)
# LASCO
if LascoC2 is not None:
    if ISSIflag:
        imL2, hdrL2 = sunpy.io._fits.read(LascoC2)[0]
    else:
        with fits.open(LascoC2) as myfitsL2:
            imL2 = myfitsL2[0].data
            myfitsL2[0].header['OBSRVTRY'] = 'SOHO'
            coordL2 = get_horizons_coord(-21, datetime.datetime.strptime(
                myfitsL2[0].header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"), 'id')
            coordL2carr = coordL2.transform_to(
                sunpy.coordinates.frames.HeliographicCarrington)
            coordL2ston = coordL2.transform_to(
                sunpy.coordinates.frames.HeliographicStonyhurst)
            myfitsL2[0].header['CRLT_OBS'] = coordL2carr.lat.deg
            myfitsL2[0].header['CRLN_OBS'] = coordL2carr.lon.deg
            myfitsL2[0].header['HGLT_OBS'] = coordL2ston.lat.deg
            myfitsL2[0].header['HGLN_OBS'] = coordL2ston.lon.deg
            hdrL2 = myfitsL2[0].header
    headers = [hdra2, hdrL2, hdrb2]
    print(hdrL2['INSTRUME'])
else:
    headers = [hdra2, hdrb2]

# # DATOS DE ENTRADA
# arrays aleatorias de parámetros 1D:
n = 3  # cant de valores de cada parámetro
CMElons = np.random.randint(60, 63, n)
CMElats = np.random.randint(20, 23, n)
CMEtilts = np.random.randint(70, 73, n)
heights = np.random.randint(6, 9, n)
ks = np.random.random_sample(size=n)
angs = np.random.randint(30, 33, n)

# cada array de parámetro pasa a ser una columna del set de parámetros:
set_parameters = np.column_stack(
    (CMElons, CMElats, CMEtilts, heights, ks, angs))

# ## Save configuración en .CSV
date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_')
configfile_name = OPATH + '/' + date_str+'Set_Parameters.csv'
header_name = ['CMElon', 'CMElat', 'CMEtilt', 'height', 'k','ang']
set = pd.DataFrame(set_parameters, columns=header_name)
set.to_csv(configfile_name)

# ## Función forwardGCS, simula CMEs en 3D de distintos parámetros morfológicos pero desde la misma posición de los satélites, esta dada por los headers
def forwardGCS(configfile_name, headers, size_occ=[2, 3.7, 2]):
    # Get the location of sats and the range of each image:
    satpos, plotranges = pyGCS.processHeaders(headers)
    df = pd.DataFrame(pd.read_csv(configfile_name))
    for row in range(len(df)):
        clouds = pyGCS.getGCS(df['CMElon'][row], df['CMElat'][row], df['CMEtilt']
                        [row], df['height'][row], df['k'][row], df['ang'][row], satpos)
        for sat in range(len(satpos)):
            #Btot figure raytrace:     
            # szx, szy = 512,512 # len(clouds[sat, :, 1]), len(clouds[sat, :, 2])
            # imsize=np.array([szx, szy], dtype='int32')
            # df['ang'][row], df['height'][row], df['k'][row] = 5.2359879e-01, 6.8295116e+00, 1.2376090e-01
            btot = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],
                                 df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row]) #, imsize=imsize)
            m = np.nanmean(btot)
            sd = np.nanstd(btot)
            fig = plt.figure(figsize=(4,4), facecolor='black')            
            plt.imshow(btot, vmax=m+3*sd, vmin=m-3*sd, origin='lower')
            fig.savefig(OPATH + '/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_btot.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig.get_facecolor())
            plt.close(fig)
            
            # mesh figure (working OK)
            fig = plt.figure(figsize=(4,4), facecolor='black') 
            ax = fig.add_subplot()
            x = clouds[sat, :, 1]
            y = clouds[0, :, 2]
            # calculo el centro y el radio del sol en coordenada de plotrange
            x_cS, y_cS = center_rSun_pixel(headers, plotranges, sat)
            # plots GCS mesh
            plt.scatter(clouds[sat, :, 1], clouds[0, :, 2],
                        s=5, c='purple', linewidths=0)
            # circulos occulter y limbo:
            occulter = plt.Circle((x_cS, y_cS), size_occ[sat], fc='white')
            limbo = plt.Circle((x_cS, y_cS), 1, ec='black', fc='white')
            ax.add_artist(occulter)
            ax.add_artist(limbo)
            # tamaño de la imagen referenciado al tamaño de la imagen del sol:
            plt.xlim(plotranges[sat][0], plotranges[sat][1])
            plt.ylim(plotranges[sat][2], plotranges[sat][3])
            plt.axis('off')
            fig.savefig(OPATH + '/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_mesh.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig.get_facecolor())
            plt.close(fig)

if __name__ == "__main__":
    forwardGCS(configfile_name, headers)
