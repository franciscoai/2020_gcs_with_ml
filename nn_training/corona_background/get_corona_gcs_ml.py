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


def get_corona(sat, imsize=None, diff=True, rnd_rot=False):
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
    
    
    cor2_path="/gehme/projects/2020_gcs_with_ml/data/corona_back_database/cor2"
    lasco_path="/gehme/projects/2020_gcs_with_ml/data/corona_back_database/lasco"
    size_occ=[2.6, 3.7, 2]# Occulters size for [sat1, sat2 ,sat3] in [Rsun] 3.7

    # main
    # STEREO A
    if sat==0:
        path=cor2_path+"/cor2_a"
    # STEREO B
    elif sat==1:
        path=cor2_path+"/cor2_b"
    elif sat==2:
        path=lasco_path+"/c2"
    else:
        os.error('Input instrument not recognized, check value of sat')

    # files=os.listdir(path)
    # p0= np.random.choice(files)
    # p0=path+"/"+p0
    # i0, h0 = sunpy.io._fits.read(p0)[0]
    # oimg=i0

    # For Mariano
    data_path = '/gehme/data/stereo/secchi/L1'
    p0 = data_path + '/a/img/cor2/20110319/20110319_110800_14c2A.fts'
    p1 = data_path + '/a/img/cor2/20110319/20110319_120800_14c2A.fts'
    p0img,h0 = sunpy.io._fits.read(p0)[0]
    p1img, _ = sunpy.io._fits.read(p1)[0]

    oimg = p1img - p0img

    if rnd_rot:
        oimg = scipy.ndimage.rotate(oimg, np.random.randint(low=0, high=360), reshape=False)
 
    if imsize is not None:
        oimg = rebin(oimg,imsize,operation='mean') 
        
    return oimg, h0, size_occ[sat]



    #sattelite positions
#    secchipath = data_path + '/stereo/secchi/L1'
#    CorA    = secchipath + '/a/img/cor2/20110317/20110317_133900_14c2A.fts' # sat1
#    CorA    = secchipath + '/a/img/cor2/20110317/20110317_133900_14c2A.fts' # sat1
#    CorB    = secchipath + '/b/img/cor2/20110317/20110317_133900_14c2B.fts' # sat2
#    lascopath = data_path + '/soho/lasco/level_1/c2' # sat3
#    LascoC2 = None # lascopath + '/20110317/25365451.fts'
#    ISSIflag = False # flag if using LASCO data from ISSI which has STEREO like headers already
  
  #read headers
  # STEREO A
#    ima2, hdra2 = sunpy.io._fits.read(CorA)[0]
  # STEREO B
#    imb2, hdrb2 = sunpy.io._fits.read(CorB)[0]
# LASCO
#    if LascoC2 is not None:
#        if ISSIflag:
#            imL2, hdrL2 = sunpy.io._fits.read(LascoC2)[0]
#        else:
#            with fits.open(LascoC2) as myfitsL2:
#                imL2 = myfitsL2[0].data
#                myfitsL2[0].header['OBSRVTRY'] = 'SOHO'
#                coordL2 = get_horizons_coord(-21, datetime.datetime.strptime(
#                    myfitsL2[0].header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"), 'id')
#                coordL2carr = coordL2.transform_to(
#                    sunpy.coordinates.frames.HeliographicCarrington)
#                coordL2ston = coordL2.transform_to(
#                    sunpy.coordinates.frames.HeliographicStonyhurst)
#                myfitsL2[0].header['CRLT_OBS'] = coordL2carr.lat.deg
#                myfitsL2[0].header['CRLN_OBS'] = coordL2carr.lon.deg
#                myfitsL2[0].header['HGLT_OBS'] = coordL2ston.lat.deg
#                myfitsL2[0].header['HGLN_OBS'] = coordL2ston.lon.deg
#                hdrL2 = myfitsL2[0].header
#        headers = [hdra2, hdrb2, hdrL2]
#    else:
#        headers = [hdra2, hdrb2]
#        ims = [ima2, imb2]

    