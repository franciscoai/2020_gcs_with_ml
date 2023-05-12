#!/usr/bin/env python
# coding: utf-8

# ## Librerias
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pyGCS_raytrace import pyGCS
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
from nn_training.low_freq_map import low_freq_map
import scipy

def center_rSun_pixel(headers, plotranges, sat):
    '''
    Gets the location of Suncenter 
    '''    
    x_cS = (headers[sat]['CRPIX1']*plotranges[sat][sat]*2) / \
        headers[sat]['NAXIS1'] - plotranges[sat][sat]
    y_cS = (headers[sat]['CRPIX2']*plotranges[sat][sat]*2) / \
        headers[sat]['NAXIS2'] - plotranges[sat][sat]
    return x_cS, y_cS

def deg2px(x,y,plotranges,imsize):
    '''
    Computes spatial plate scale in both dimensions
    '''    
    scale_x = (plotranges[sat][1]-plotranges[sat][0])/imsize[0]
    scale_y =(plotranges[sat][3]-plotranges[sat][2])/imsize[1]
    x_px=[]
    y_px=[]    
    for i in range(len(x)):
        v_x= (np.round((x[i]-plotranges[sat][0])/scale_x)).astype("int") 
        v_y= (np.round((y[i]-plotranges[sat][2])/scale_y)).astype("int") 
        x_px.append(v_x)
        y_px.append(v_y)
    return(y_px,x_px)

######Main
# CONSTANTS
#files
exec_path = os.getcwd()
DATA_PATH = '/gehme/data'
OPATH = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_dataset_fran' #'/gehme/projects/2020_gcs_with_ml/data/forwardGCS_test'

#sattelite positions
secchipath = DATA_PATH + '/stereo/secchi/L1'
CorA    = secchipath + '/a/img/cor2/20110317/20110317_133900_14c2A.fts' # sat1
CorB    = secchipath + '/b/img/cor2/20110317/20110317_133900_14c2B.fts' # sat2
lascopath = DATA_PATH + '/soho/lasco/level_1/c2' # sat3
LascoC2 = None # lascopath + '/20110317/25365451.fts'
ISSIflag = False # flag if using LASCO data from ISSI which has STEREO like headers already

# GCS parameters [first 6]
# The other parameters are:
# level_cme: CME intensity level relative to the mean background corona
par_names = ['CMElon', 'CMElat', 'CMEtilt', 'height', 'k','ang', 'level_cme'] # par names
par_units = ['deg', 'deg', 'deg', 'Rsun','','deg',''] # par units
par_rng = [[-180,180],[-70,70],[-90,90],[8,14],[0.2,0.6], [10,60],[1e2,5e2]] # min-max ranges of each parameter in par_names
par_num = 4000  # total number of samples that will be generated for each param (ther are 2 or 3 images (satellites) per param combination)
#par_rng = [[165,167],[-22,-20],[-66,-64],[10,15],[0.21,0.23], [19,21],[9e4,10e4]] # example used for script development
rnd_par=True # set to randomnly shuffle the generated parameters linspace 

# Syntethic image options
imsize=np.array([512, 512], dtype='int32') # output image size
size_occ = [2.6, 3.7, 2] # Occulters size for [sat1, sat2 ,sat3] in [Rsun] 3.7
level_occ=0. #mean level of the occulter relative to the background level
level_noise=0 #photon noise level of cme image relative to photon noise. Set to 0 to avoid
mesh=False # set to also save a png with the GCSmesh
otype="png" # set the ouput file type: 'png' or 'fits'
im_range=1. # range of the color scale of the output final syntethyc image in std dev around the mean
back_rnd_rot=True # set to randomly rotate the background image around its center

## main
par_num = [par_num] * len(par_rng)
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
    cpar = np.linspace(rng[0],rng[1], num)
    if rnd_par:
        np.random.shuffle(cpar)
    all_par.append(cpar)

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
        #defining ranges and radius of the occulter
        x = np.linspace(plotranges[sat][0], plotranges[sat][1], num=imsize[0])
        y = np.linspace(plotranges[sat][2], plotranges[sat][3], num=imsize[1])
        xx, yy = np.meshgrid(x, y)
        x_cS, y_cS = center_rSun_pixel(headers, plotranges, sat)  
        r = np.sqrt((xx - x_cS)**2 + (yy - y_cS)**2)

     
        #mask for cme outer envelope
        clouds = pyGCS.getGCS(df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], satpos)                
        x = clouds[sat, :, 1]
        y = clouds[0, :, 2]
        cloud_arr= np.zeros(imsize)
        p_x,p_y=deg2px(x,y,plotranges,imsize)
        for i in range(len(p_x)):
            cloud_arr[p_x[i], p_y[i]] = 1
        # fig8 = plt.figure(figsize=(4,4), facecolor='black')
        # plt.imshow(cloud_arr, origin='lower')
        # fig8.savefig(OPATH+ '/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_mesh_test.png'.format(
        #         df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig8.get_facecolor())
        # plt.close(fig8)           

        btot_mask = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], imsize=imsize, occrad=size_occ[sat], in_sig=0.1, out_sig=0.1, nel=1e5)     
        mask = segmentation(btot_mask)
        mask[r <= size_occ[sat]] = 0  
        if len(np.array(np.where(mask==1)).flatten())/len(mask.flatten())<0.005: # only if there is a cme that covers more than 0.5% of the image
            print(f'WARNINGN: CME number {row} mask is null because it is probably behind the occulter, skipping all views...')
            break

        #Total intensity (Btot) figure from raytrace:               
        btot0 = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], imsize=imsize, occrad=size_occ[sat], in_sig=0.5, out_sig=0.25, nel=1e5)
                      
        #adds a flux rope-like structure
        height_diff = np.random.uniform(low=0.55, high=0.65)
        aspect_ratio_frope = np.random.uniform(low=0.9, high=0.14)
        btot1 = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row]*height_diff, 0.13, df['ang'][row], imsize=imsize, occrad=size_occ[sat], in_sig=0.7, out_sig=0.3, nel=1e5)
        btot = btot0 + btot1

        #mask for occulter
        arr = np.zeros(xx.shape)
        arr[r <= size_occ[sat]] = 1    

        #background corona
        back = same_corona[sat]
        if back_rnd_rot:
            back =  scipy.ndimage.rotate(back, np.random.randint(low=0, high=360), reshape=False)
        level_back = np.mean(back)               

        #cme
        var_map = low_freq_map(dim=[imsize[0],imsize[1],1],off=[1],var=[1.5],func=[13])
        btot = var_map*(btot/np.max(btot))*df['level_cme'][row]*level_back
        #adds noise
        if level_noise > 0:
            noise=np.random.poisson(lam=btot)*level_noise
            btot+=noise    
        else:
            noise=0    
        #adds background
        btot+=back
        #adds occulter
        btot[r <= size_occ[sat]] = level_occ*level_back + noise


        #creating folders for each case
        folder = os.path.join(OPATH, str(row*len(satpos)+sat))
        if os.path.exists(folder):
            os.system("rm -r " + folder) 
        os.makedirs(folder)
        mask_folder = os.path.join(folder, "mask")
        os.makedirs(mask_folder) 

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
            #mask for cme
            fig3 = plt.figure(figsize=(4,4), facecolor='black')
            ax3 = fig3.add_subplot()        
            ax3.imshow(mask, origin='lower', cmap='gray',vmax=1, vmin=0, extent=plotranges[sat])
            fig3.savefig(mask_folder +'/2.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig3.get_facecolor())
            plt.close(fig3)
            #mask for occulter
            fig_occ = plt.figure(figsize=(4,4), facecolor='black')
            ax_occ = fig_occ.add_subplot()  
            ax_occ.imshow(arr, cmap='gray',extent=plotranges[sat],origin='lower', vmax=1, vmin=0)
            fig_occ.savefig(mask_folder +'/1.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig_occ.get_facecolor())            
            plt.close(fig_occ)
            #full image
            m = np.mean(btot)
            sd =np.std(btot)
            fig = plt.figure(figsize=(4,4), facecolor='black')
            ax = fig.add_subplot()                
            ax.imshow(btot, origin='lower', cmap='gray', vmin=m-im_range*sd, vmax=m+im_range*sd,extent=plotranges[sat])
            fig.savefig(folder +'/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_btot.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), facecolor=fig.get_facecolor())
            if not mesh:
                plt.close(fig)
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

#os.system("chgrp -R gehme " + OPATH)
