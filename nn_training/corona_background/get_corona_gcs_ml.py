#!/usr/bin/env python
# coding: utf-8

# ## Librerias
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import sunpy
import pandas as pd
from ext_libs.rebin import rebin
import scipy
from astropy.io import fits

import datetime


def get_corona(sat, imsize=None, diff=True, rnd_rot=False, obs_datetime=None):
    '''
    Returns a measured "quiet" (with no CME) solar corona observed by satelitte sat, the implemented instruments are

    sat=
        0: Stereo A Cor2
        1: Stereo B Cor2
        2: Lasco C2

    OPTIONS:
        diff: Set to True to return a time differential corona.
        imsize: Set to [x,y] to imsize the output image to that size
        rnd_rot: Set to rotate the ouput image by a random angle around the central pixel

    '''
    # CONSTANTS
    #files
    
    
    cor2_path="/gehme/projects/2020_gcs_with_ml/data/corona_background_affects/cor2"
    lasco_path="/gehme/projects/2020_gcs_with_ml/data/corona_back_database/lasco"
    h_cor2b="/gehme/data/stereo/secchi/L1/b/img/cor2/20130209/20130209_062400_14c2B.fts"
    h_cor2a="/gehme/data/stereo/secchi/L1/a/img/cor2/20130209/20130209_062400_14c2A.fts" 
    h_lasco="path/to/lasco/header"
    size_occ=[2.6, 3.7, 2]# Occulters size for [sat1, sat2 ,sat3] in [Rsun]
    # size_occ=[1.4, 1.4, 2]# Occulters size for [sat1, sat2 ,sat3] in [Rsun] 
    max_time_diff= datetime.timedelta(hours=2)

    # main
     # STEREO A
    if sat==0:
        path=cor2_path+"/cor2_b"
    # STEREO B
    elif sat==1:
        path=cor2_path+"/cor2_a"
    # LASCO    
    elif sat==2:
        path=lasco_path+"/c2"
    else:
        os.error('Input instrument not recognized, check value of sat')

    files=[f for f in os.listdir(path) if f.endswith('.fits')]
    
    if obs_datetime is not None:
        # get the closest file to the input datetime
        files_datetime = [pd.to_datetime(f.split('_')[0] + '_' + f.split('_')[1].split('.')[0], format='%Y%m%d_%H%M%S') for f in files]
        time_diff = [np.abs(fdt - obs_datetime) for fdt in files_datetime]
        if np.min(time_diff) > max_time_diff:
            print('WARNING: No file found within ', max_time_diff, ' of the input datetime')
        p0 = files[np.argmin(time_diff)]
    else:
        p0= np.random.choice(files)
    # get full datetime with hours, minutes, seconds
    day = p0.split('_')[0]
    time = p0.split('_')[1]
    obs_datetime = pd.to_datetime(day + '_' + time.split('.')[0], format='%Y%m%d_%H%M%S')

    p0=path+"/"+p0
    
    oimg= fits.open(p0)[0].data
    if sat == 0:
        h0= fits.getheader(h_cor2b)
    elif sat == 1:
        h0= fits.getheader(h_cor2a)
    else:
        h0= fits.getheader(h_lasco)

    if rnd_rot:
        oimg = scipy.ndimage.rotate(oimg, np.random.randint(low=0, high=360), reshape=False)
 
    if imsize is not None:
        oimg = rebin(oimg,imsize,operation='mean')
    return oimg, h0, size_occ[sat], obs_datetime

    # STEREO A C1
    # if sat==0:
    #     p0 = '/gehme/data/stereo/secchi/L1/a/seq/cor1/20130424/20130424_051500_1B4c1A.fts'
    #     p1 =  '/gehme/data/stereo/secchi/L1/a/seq/cor1/20130424/20130424_052000_1B4c1A.fts'
    #     p0img,h0 = sunpy.io._fits.read(p0)[0]
    #     p1img, _ = sunpy.io._fits.read(p1)[0]
    #     oimg = p1img - p0img
    # # STEREO B C1
    # elif sat==1:
    #     p0 = '/gehme/data/stereo/secchi/L1/b/seq/cor1/20130424/20130424_051500_1B4c1B.fts'
    #     p1 = '/gehme/data/stereo/secchi/L1/b/seq/cor1/20130424/20130424_052000_1B4c1B.fts'
    #     p0img,h0 = sunpy.io._fits.read(p0)[0]
    #     p1img, _ = sunpy.io._fits.read(p1)[0]
    #     oimg = p1img - p0img        
    # # lasco C1
    # elif sat==2:
    #     p0 = '/gehme/data/soho/lasco/level_1/c2/20130424/25456651.fts'
    #     p1 = '/gehme/data/soho/lasco/level_1/c2/20130424/25456652.fts'
    #     p0img,h0 = sunpy.io._fits.read(p0)[0]
    #     p1img, _ = sunpy.io._fits.read(p1)[0]
    #     oimg = p1img - p0img
    # else:
    #     os.error('Input instrument not recognized, check value of sat')

    # if rnd_rot:
    #     oimg = scipy.ndimage.rotate(oimg, np.random.randint(low=0, high=360), reshape=False)
 
    # if imsize is not None:
    #     sz_ratio = np.array(oimg.shape)/np.array(imsize)
    #     oimg = rebin(oimg,imsize,operation='mean') 
    #     h0['NAXIS1'] = imsize[0]
    #     h0['NAXIS2'] = imsize[1]
    #     h0['CDELT1'] = h0['CDELT1']*sz_ratio[0]
    #     h0['CDELT2'] = h0['CDELT2']*sz_ratio[1]
    #     h0['CRPIX2'] = int(h0['CRPIX2']/sz_ratio[1])
    #     h0['CRPIX1'] = int(h0['CRPIX1']/sz_ratio[1]) 
    #     size_occ[sat] = size_occ[sat]*np.mean(sz_ratio)

    # if sat == 2:
    #     breakpoint()
    # return oimg, h0, size_occ[sat]

