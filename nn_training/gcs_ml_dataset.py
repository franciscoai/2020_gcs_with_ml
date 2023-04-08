#!/usr/bin/env python
# coding: utf-8

# ## Librerias
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pyGCS_raytrace import pyGCS
#from pyGCS_raytrace.GCSgui import runGCSgui
from pyGCS_raytrace.rtraytracewcs import rtraytracewcs
import numpy as np
import datetime
import matplotlib.pyplot as plt
from astropy.io import fits
import sunpy
from sunpy.coordinates.ephemeris import get_horizons_coord
import sunpy.map
import pandas as pd
from sunpy.sun.constants import radius as _RSUN
from ext_libs.rebin import rebin

## Función ajuste del centro del sol
def center_rSun_pixel(headers, plotranges, sat):
    x_cS = (headers[sat]['CRPIX1']*plotranges[sat][sat]*2) / \
        headers[sat]['NAXIS1'] - plotranges[sat][sat]
    y_cS = (headers[sat]['CRPIX2']*plotranges[sat][sat]*2) / \
        headers[sat]['NAXIS2'] - plotranges[sat][sat]
  
    return x_cS, y_cS

ISSIflag = False # flag if using LASCO data from ISSI which has STEREO like headers already
exec_path = os.getcwd()
DATA_PATH = '/gehme/data'
OPATH = exec_path + '/../../output/cme_seg_dataset' #'/gehme/projects/2020_gcs_with_ml/data/forwardGCS_test'

# Files to get headers
secchipath = DATA_PATH + '/stereo/secchi/L1'
lascopath = DATA_PATH + '/soho/lasco/level_1/c2'
#event 1
preCorA = secchipath + '/a/img/cor2/20110317/20110317_115400_14c2A.fts' #'/a/img/cor2/20110317/20110317_132400_14c2A.fts'
CorA    = secchipath + '/a/img/cor2/20110317/20110317_133900_14c2A.fts'
preCorB = secchipath + '/b/img/cor2/20110317/20110317_123900_14c2B.fts' #'/b/img/cor2/20110317/20110317_132400_14c2B.fts'
CorB    = secchipath + '/b/img/cor2/20110317/20110317_133900_14c2B.fts'

# #event 2 DECIRLE A FER QUE LOS PASE A L1!!
# CorA    =secchipath + '/a/img/cor2/20110303/20110303_080800_14c2A.fts'
# preCorA =secchipath + '/a/img/cor2/20110303/20110303_020800_14c2A.fts' 
# CorB    =secchipath + '/b/img/cor2/20110303/20110303_080915_14c2B.fts'
# preCorB =secchipath + '/b/img/cor2/20110303/20110303_040915_14c2B.fts'


LascoC2 = None
#LascoC2 = lascopath + '/20110317/25365451.fts'
# synthetic coronograph image additions
cor2 = 2 # Tamaño de los occulters referenciados al RSUN
c3 = 3.7 # Tamaño de los occulters referenciados al RSUN


## main
os.makedirs(OPATH, exist_ok=True)
# STEREO A
ima2, hdra2 = sunpy.io._fits.read(CorA)[0]
preima2, prehdra2 = sunpy.io._fits.read(preCorA)[0]
#smap_SA2 = sunpy.map.Map(ima2, hdra2)
# STEREO B
imb2, hdrb2 = sunpy.io._fits.read(CorB)[0]
preimb2, prehdrb2 = sunpy.io._fits.read(preCorB)[0]

#smap_SB2 = sunpy.map.Map(imb2, hdrb2)
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
    
else:
    headers = [hdra2, hdrb2]
    ims = [ima2, imb2]
    preims=[preima2, preimb2]

    #print(hdra2)
     
    #######################

# # DATOS DE ENTRADA
# arrays aleatorias de parámetros 1D:
""" n = 3  # cant de valores de cada parámetro
CMElons = np.random.randint(60, 63, n)
CMElats = np.random.randint(20, 23, n)
CMEtilts = np.random.randint(70, 73, n)
heights = np.random.randint(6, 9, n)
ks = np.random.random_sample(size=n)
angs = np.random.randint(30, 33, n) """
################################################
""" n = 1  # cant de valores de cada parámetro
CMElons = math.degrees(1.99032)  #used in rtsccguicloud_calcneang
CMElats = math.degrees(0.936634) #used in rtsccguicloud_calcneang
CMEtilts = math.degrees(1.52201) #used in rtsccguicloud_calcneang
heights = 4.3667865e+00 #modparam[2] [Rsun]
ks = 1.3911580e-01 #modparam[3]
angs = math.degrees(5.2359879e-01) #modparam[1] """
################################################
#parameters event 20110317:
CMElons = 166.583  #  Longitude as in IDL GUI but
CMElats = -21.2418 #  Latitude as in IDL GUI
CMEtilts = -65.9628 # Rotation as in IDL GUI
heights =  6.78565 # Heigth as in IDL GUI
ks = 0.231364 # Ratio as in IDL GUI
angs = 20.6829 # Half angle as in IDL GUI

# Raytracing options
imsize=np.array([512, 512], dtype='int32') # output image size

###################################################
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
def forwardGCS(configfile_name, headers, size_occ=[2, 3.7, 2], mesh=False):
    # Get the location of sats and the range of each image:
    satpos, plotranges = pyGCS.processHeaders(headers)
    
    df = pd.DataFrame(pd.read_csv(configfile_name))
    for row in range(len(df)):
        if mesh:
            clouds = pyGCS.getGCS(df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], satpos)
        for sat in range(len(satpos)): 
            
            #background event
            preev = rebin(preims[sat],imsize,operation='mean') 
            back = rebin(ims[sat],imsize,operation='mean') - preev
            m = np.nanmean(back)
            sd = np.nanstd(back)
            fig = plt.figure(figsize=(4,4), facecolor='black')
            ax = fig.add_subplot()    
            x_cS, y_cS = center_rSun_pixel(headers, plotranges, sat)      
            ax.imshow(back, origin='lower', cmap='gray', vmax=m+3*sd, vmin=m-3*sd, extent=plotranges[sat])
            occulter = plt.Circle((x_cS, y_cS), size_occ[sat], fc='white')
            limbo = plt.Circle((x_cS, y_cS), 1, ec='black', fc='white')
            ax.add_artist(occulter)
            ax.add_artist(limbo)
            fig.savefig(OPATH + '/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_back.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig.get_facecolor())


            #Total intensity (Btot) figure from raytrace:               
            btot = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], imsize=imsize)
            btot =  btot/np.mean(btot)*1e-11 + back
            m = np.nanmean(btot)
            sd = np.nanstd(btot)
            fig = plt.figure(figsize=(4,4), facecolor='black')
            ax = fig.add_subplot()    
            x_cS, y_cS = center_rSun_pixel(headers, plotranges, sat)      
            ax.imshow(btot, origin='lower', cmap='gray', vmax=m+3*sd, vmin=m-3*sd, extent=plotranges[sat])
            occulter = plt.Circle((x_cS, y_cS), size_occ[sat], fc='white')
            limbo = plt.Circle((x_cS, y_cS), 1, ec='black', fc='white')
            ax.add_artist(occulter)
            ax.add_artist(limbo)
            fig.savefig(OPATH + '/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_btot.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig.get_facecolor())

            if mesh:
                # overlaps mesh figure
                x = clouds[sat, :, 1]
                y = clouds[0, :, 2]
                plt.scatter(x, y, s=0.5, c='green', linewidths=0)
                plt.axis('off')
                fig.savefig(OPATH + '/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_mesh.png'.format(
                    df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig.get_facecolor())
                plt.close(fig)

if __name__ == "__main__":
    forwardGCS(configfile_name, headers, mesh=True)

