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
from sunpy.coordinates.ephemeris import get_horizons_coord
import datetime


def get_corona(sat, imsize=None, diff=True, rnd_rot=False, custom_headers=False):
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
        custom_headers: Set to True to use the headers of the H files instead of the corona files

    '''
    # CONSTANTS
    #files
    
    
    cor2_a_path= "/gehme/projects/2020_gcs_with_ml/data/corona_background_affects/cor2/cor2_a"
    cor2_b_path= "/gehme/projects/2020_gcs_with_ml/data/corona_background_affects/cor2/cor2_b"
    lasco_path="/gehme/projects/2020_gcs_with_ml/data/corona_background_affects/lasco/c2/3VP"
    H_COR2B = "/gehme/data/stereo/secchi/L1/b/img/cor2/20130209/20130209_062400_14c2B.fts"
    H_COR2A = "/gehme/data/stereo/secchi/L1/a/img/cor2/20130209/20130209_062400_14c2A.fts" 
    H_LASCO = "/gehme/data/soho/lasco/level_1/c2/20130209/25447666.fts"     
    size_occ  = [2.9, 3.9, 1.9]
    size_occ_ext=[16+1, 16+1, 6+1] # Occulters size for [sat1, sat2 ,sat3] in [Rsun]
    occ_center = [[0.,0.], [0.3,-0.3] , [0.,0.]] # [0.3,-0.6] Occulters center offsets [x,y] for [sat1, sat2 ,sat3] in arcsec?

    # main
    # STEREO A
    if sat==0:
        path=cor2_a_path
        h_path=H_COR2A
    # STEREO B
    elif sat==1:
        path=cor2_b_path
        h_path=H_COR2B
    # LASCO    
    elif sat==2:
        path=lasco_path
        h_path=H_LASCO
    else:
        os.error('Input instrument not recognized, check value of sat')

    files=[f for f in os.listdir(path) if (f.endswith('.fits') or f.endswith('.fts'))]
    
    p0= np.random.choice(files)
    p0=path+"/"+p0
    
    print('Using back file ', p0)
    oimg= fits.open(p0)[0].data

    # returns allways same header
    if custom_headers:
        h0 = fits.getheader(h_path)
    else:
        h0= fits.getheader(p0)

    if rnd_rot:
        oimg = scipy.ndimage.rotate(oimg, np.random.randint(low=0, high=360), reshape=False)
 
    if imsize is not None:
        oimg = rebin(oimg,imsize,operation='mean') 
   
    return oimg, h0, size_occ[sat], size_occ_ext[sat], occ_center[sat]