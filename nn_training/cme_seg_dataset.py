#!/usr/bin/env python
# coding: utf-8

# ## Librerias
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pyGCS_raytrace import pyGCS
#from pyGCS_raytrace.GCSgui import runGCSgui
from pyGCS_raytrace.rtraytracewcs import rtraytracewcs
from nn_training.segmentation import segmentation
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

def rnd_samples(rng, n):
# gernerate n random (uniform dist) float samples from range rnd[0] to rnd[1]    
    return (rng[1] - rng[0]) * np.random.random(n) + rng[0]

######Main
# CONSTANTS
#files
exec_path = os.getcwd()
DATA_PATH = '/gehme/data'
OPATH = '/gehme/projects/2020_gcs_with_ml/data/cme_seg_dataset' #'/gehme/projects/2020_gcs_with_ml/data/forwardGCS_test'

secchipath = DATA_PATH + '/stereo/secchi/L1'
preCorA = secchipath + '/a/img/cor2/20110317/20110317_115400_14c2A.fts' #'/a/img/cor2/20110317/20110317_132400_14c2A.fts'
CorA    = secchipath + '/a/img/cor2/20110317/20110317_133900_14c2A.fts'
preCorB = secchipath + '/b/img/cor2/20110317/20110317_123900_14c2B.fts' #'/b/img/cor2/20110317/20110317_132400_14c2B.fts'
CorB    = secchipath + '/b/img/cor2/20110317/20110317_133900_14c2B.fts'

lascopath = DATA_PATH + '/soho/lasco/level_1/c2'
LascoC2 = None # lascopath + '/20110317/25365451.fts'
ISSIflag = False # flag if using LASCO data from ISSI which has STEREO like headers already

# GCS parameters
par_names = ['CMElon', 'CMElat', 'CMEtilt', 'height', 'k','ang'] # par names
par_units = ['deg', 'deg', 'deg', 'Rsun','','deg'] # param units
par_num = [3, 3, 3, 3, 3, 3]  # total number of samples that will be generated for each param
par_rng = [[165,167],[-22,-20],[-66,-64],[5,7],[0.21,0.23], [19,21]] # min-max ranges of each parameter in par_names

# Sintethyc image options
size_occ = [2, 3.7, 2] # Occulters size for [sat1, sat2,sat3] in [Rsun] 3.7
imsize=np.array([512, 512], dtype='int32') # output image size

## main
#read headers
# STEREO A
ima2, hdra2 = sunpy.io._fits.read(CorA)[0]
preima2, prehdra2 = sunpy.io._fits.read(preCorA)[0]
# STEREO B
imb2, hdrb2 = sunpy.io._fits.read(CorB)[0]
preimb2, prehdrb2 = sunpy.io._fits.read(preCorB)[0]
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
 
# generate param arrays
all_par = []
for (rng, num) in zip(par_rng, par_num):
    all_par.append(np.linspace(rng[0],rng[1], num))

# Save configuración en .CSV
os.makedirs(OPATH, exist_ok=True)
date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_')
configfile_name = OPATH + '/' + date_str+'Set_Parameters.csv'
set = pd.DataFrame(np.column_stack(all_par), columns=par_names)
set.to_csv(configfile_name)

# generate views
def forwardGCS(configfile_name, headers, size_occ=size_occ, mesh=False):
    # Get the location of sats and the range of each image:
    satpos, plotranges = pyGCS.processHeaders(headers)
    df = pd.DataFrame(pd.read_csv(configfile_name))
    for row in range(len(df)):
        if mesh:
            clouds = pyGCS.getGCS(df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], satpos)
        
        for sat in range(len(satpos)): 
            #creating folders for each case
            folder = os.path.join(OPATH, str(row*len(satpos)+sat))
            if os.path.exists(folder):
                os.system("rm -r " + folder) 
            os.makedirs(folder)
            mask_folder = os.path.join(folder, "mask")
            os.makedirs(mask_folder)         

            #background event
            #preev = rebin(preims[sat],imsize,operation='mean') 
            #back = rebin(ims[sat],imsize,operation='mean') - preev

            #Total intensity (Btot) figure from raytrace:               
            btot_orig = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], imsize=imsize)
            print("btot shape: ", np.shape(btot_orig))
            # plt.hist(btot_orig.flatten(),range=[0,5e-11],bins=200, log=True)
            btot =  btot_orig#/np.mean(btot_orig)*1e-11 + back
            m = np.nanmean(btot)
            sd = np.nanstd(btot)
            fig = plt.figure(figsize=(4,4), facecolor='black')
            ax = fig.add_subplot()    
            x_cS, y_cS = center_rSun_pixel(headers, plotranges, sat)      
            ax.imshow(btot, origin='lower', cmap='gray', vmax=m+3*sd, vmin=m-3*sd, extent=plotranges[sat])
            occulter = plt.Circle((x_cS, y_cS), size_occ[sat], fc='white')
            #limbo = plt.Circle((x_cS, y_cS), 1, ec='black', fc='white')
            ax.add_artist(occulter)
            #ax.add_artist(limbo)

            fig.savefig(folder +'/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_btot.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig.get_facecolor())
            

            occulter = plt.Circle((x_cS, y_cS), size_occ[sat], fc='white')
            fig2 = plt.figure(figsize=(10,10), facecolor='black')
            ax2 = fig2.add_subplot()
            ax2.imshow(np.zeros(imsize), origin='lower', cmap='gray', vmax=1, vmin=0, extent=plotranges[sat])             
            #limbo = plt.Circle((x_cS, y_cS), 1, ec='black', fc='white')
            ax2.add_artist(occulter)
            #ax2.add_artist(limbo)
            fig2.savefig(mask_folder +'/1.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig.get_facecolor())

            occulter = plt.Circle((x_cS, y_cS), size_occ[sat], fc='black')
            mask = segmentation(btot_orig)
            print("mask shape: ", np.shape(mask))
            fig3 = plt.figure(figsize=(10,10), facecolor='black')
            ax3 = fig3.add_subplot()        
            ax3.imshow(mask, origin='lower', cmap='gray', vmax=1, vmin=0, extent=plotranges[sat])
            ax3.add_artist(occulter)
            fig3.savefig(mask_folder +'/2.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig.get_facecolor())

            #plt.show()

            

            if mesh:
                # overlaps mesh figure
                x = clouds[sat, :, 1]
                y = clouds[0, :, 2]
                plt.scatter(x, y, s=0.5, c='green', linewidths=0)
                plt.axis('off')
                fig.savefig(OPATH+ '/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_mesh.png'.format(
                    df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig.get_facecolor())
                plt.close(fig)

    os.system("chgrp -R gehme " + OPATH)

if __name__ == "__main__":
    forwardGCS(configfile_name, headers, mesh=False)

