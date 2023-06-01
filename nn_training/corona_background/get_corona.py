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

    OPTIONS:
        diff: Set to True to return a time differential corona.
        imsize: Set to [x,y] to imsize the output image to that size
        rnd_rot: Set to rotate the ouput image by a random angle around the central pixel

    '''
    # CONSTANTS
    #files
    data_path = '/gehme/data'
    secchipath = data_path + '/stereo/secchi/L1'

    # main
    # STEREO A
    if sat==0:
        p0 = ['/a/img/cor2/20110317/20110317_115400_14c2A.fts', '/a/img/cor2/20130424/20130424_055400_14c2A.fts']
        p1 = ['/a/img/cor2/20110317/20110317_122400_14c2A.fts', '/a/img/cor2/20130424/20130424_062400_14c2A.fts']           
        i = np.random.randint(low=0, high=len(p0)-1)
        p0=secchipath + p0[i]
        p1=secchipath + p1[i]
        i0, h0 = sunpy.io._fits.read(p0)[0]
        if diff:
            i1, h1 = sunpy.io._fits.read(p1)[0]
            oimg = i1-i0
        else:
            oimg = i1
    # STEREO B
    elif sat==1:
        p0 = ['/b/img/cor2/20110317/20110317_123900_14c2B.fts', '/b/img/cor2/20130424/20130424_065400_14c2B.fts']
        p1 = ['/b/img/cor2/20110317/20110317_125400_14c2B.fts', '/b/img/cor2/20130424/20130424_072400_14c2B.fts']        
        i = np.random.randint(low=0, high=len(p0)-1)           
        p0=secchipath + p0[i]
        p1=secchipath + p1[i]    
        i0, h0 = sunpy.io._fits.read(p0)[0]
        if diff:
            i1, h1 = sunpy.io._fits.read(p1)[0]
            oimg = i1-i0
        else:
            oimg = i1
    else:
        os.error('Input instrument not recognized, check value of sat')
    
    if rnd_rot:
        oimg = scipy.ndimage.rotate(oimg, np.random.randint(low=0, high=360), reshape=False)
 
    if imsize is not None:
        oimg = rebin(oimg,imsize,operation='mean') 
        
    return oimg, h0



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

    