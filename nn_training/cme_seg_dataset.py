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
from nn_training.corona_background.get_corona import get_corona
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

#sattelite positions
secchipath = DATA_PATH + '/stereo/secchi/L1'
CorA    = secchipath + '/a/img/cor2/20110317/20110317_133900_14c2A.fts' # sat1
CorB    = secchipath + '/b/img/cor2/20110317/20110317_133900_14c2B.fts' # sat2
lascopath = DATA_PATH + '/soho/lasco/level_1/c2' # sat3
LascoC2 = None # lascopath + '/20110317/25365451.fts'
ISSIflag = False # flag if using LASCO data from ISSI which has STEREO like headers already

# GCS parameters
par_names = ['CMElon', 'CMElat', 'CMEtilt', 'height', 'k','ang', 'level_cme'] # par names
par_units = ['deg', 'deg', 'deg', 'Rsun','','deg',''] # param units
par_num = [1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4]  # total number of samples that will be generated for each param
par_rng = [[-180,180],[-70,-70],[-180,180],[4,15],[0.1,0.49], [-180,180],[5e2,9e2]] # min-max ranges of each parameter in par_names
par_num = [3, 3, 3, 3, 3, 3,3]  # total number of samples that will be generated for each param
#par_rng = [[165,167],[-22,-20],[-66,-64],[10,15],[0.21,0.23], [19,21],[5e2,1e3]]


# Syntethic image options
imsize=np.array([512, 512], dtype='int32') # output image size
size_occ = [2.6, 3.7, 2] # Occulters size for [sat1, sat2 ,sat3] in [Rsun] 3.7
level_occ=1000 #mean level of the occulter relative to the background level
level_noise=0 #photon noise level of cme image relative to photon noise. Set to 0 to avoid
mesh=False # set to also save a png with the GCSmesh
otype="png" # set the ouput file type: 'png' or 'fits'

## main
#read headers
# STEREO A
ima2, hdra2 = sunpy.io._fits.read(CorA)[0]
# STEREO B
imb2, hdrb2 = sunpy.io._fits.read(CorB)[0]
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

# Get the location of sats and gcs:
satpos, plotranges = pyGCS.processHeaders(headers)
df = pd.DataFrame(pd.read_csv(configfile_name))

#back corona, temporal must be change to have a different corona per image
same_corona = [get_corona(0,imsize=imsize), get_corona(1,imsize=imsize)]

# generate views
for row in range(len(df)):
    print(f'Saving image pair {row} of {len(df)-1}')
    for sat in range(len(satpos)): 
        #creating folders for each case
        folder = os.path.join(OPATH, str(row*len(satpos)+sat))
        if os.path.exists(folder):
            os.system("rm -r " + folder) 
        os.makedirs(folder)
        mask_folder = os.path.join(folder, "mask")
        os.makedirs(mask_folder)     

        #defining ranges an radio of the occulter
        x = np.linspace(plotranges[sat][0], plotranges[sat][1], num=512)
        y = np.linspace(plotranges[sat][2], plotranges[sat][3], num=512)
        xx, yy = np.meshgrid(x, y)
        x_cS, y_cS = center_rSun_pixel(headers, plotranges, sat)  
        r = np.sqrt((xx - x_cS)**2 + (yy - y_cS)**2)


        #Total intensity (Btot) figure from raytrace:               
        btot = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], imsize=imsize)
        #btot1 = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row]*1.3, df['k'][row]*1.05, df['ang'][row], imsize=imsize)
        #btot = btot1 - btot

        #mask for cme
        mask = btot #segmentation(btot)
        mask[r <= size_occ[sat]] = 0   

        #mask for occulter
        arr = np.zeros(xx.shape)
        arr[r <= size_occ[sat]] = 1    

        #background corona
        back = same_corona[sat]
        level_back = np.mean(back)               

        #cme
        btot = (btot/np.max(btot))*df['level_cme'][row]*level_back
        #adds noise
        if level_noise > 0:
            noise=np.random.poisson(lam=btot)*level_noise
            btot+=noise        
        #adds background
        btot+=back
        m = np.nanmean(btot)
        sd = np.nanstd(btot)
        #adds occulter
        btot[r <= size_occ[sat]] = level_occ*level_back

        if otype=="fits":
            #mask for cme
            cme_mask = fits.PrimaryHDU(mask)
            cme_mask.writeto(mask_folder +'/2.fits'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), overwrite=True)
            #cme
            cme = fits.PrimaryHDU(btot)
            cme.writeto(folder +'/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_btot.fits'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), overwrite=True)
            #mask for occulter
            occ = fits.PrimaryHDU(arr)
            occ.writeto(mask_folder +'/1.fits'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), overwrite=True)
            
        elif otype =="png":
            fig = plt.figure(figsize=(4,4), facecolor='black')
            ax = fig.add_subplot()            
            #mask for cme
            fig3 = plt.figure(figsize=(4,4), facecolor='black')
            ax3 = fig3.add_subplot()        
            ax3.imshow(mask, origin='lower', cmap='gray',vmax=1, vmin=0, extent=plotranges[sat])
            fig3.savefig(mask_folder +'/2.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig3.get_facecolor())
            #mask for occulter
            fig_occ = plt.figure(figsize=(4,4), facecolor='black')
            ax_occ = fig_occ.add_subplot()  
            ax_occ.imshow(arr, cmap='gray',extent=plotranges[sat],origin='lower', vmax=1, vmin=0)
            fig_occ.savefig(mask_folder +'/1.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig.get_facecolor())            
            #cme
            m = np.mean(btot)
            sd =np.std(btot)
            ax.imshow(btot, origin='lower', cmap='gray', vmin=m-3*sd, vmax=m+3*sd,extent=plotranges[sat])
            fig.savefig(folder +'/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_btot.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig.get_facecolor())
        else:
            print("otype value not recognized")    

        if mesh:
            # overplot  GCS mesh to cme figure
            clouds = pyGCS.getGCS(df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], satpos)                
            x = clouds[sat, :, 1]
            y = clouds[0, :, 2]
            plt.scatter(x, y, s=0.5, c='green', linewidths=0)
            plt.axis('off')
            fig.savefig(OPATH+ '/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_mesh.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig.get_facecolor())
            plt.close(fig)

os.system("chgrp -R gehme " + OPATH)
