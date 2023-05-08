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


def rnd_samples(rng, n):
# gernerate n random (uniform dist) float samples from range rnd[0] to rnd[1]    
    return (rng[1] - rng[0]) * np.random.random(n) + rng[0]

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
    DATA_PATH = '/gehme/data'
    secchipath = DATA_PATH + '/stereo/secchi/L1'


    # main
    # STEREO A
    if sat==0:
        p0 = secchipath + '/a/img/cor2/20110317/20110317_115400_14c2A.fts'
        p1 = secchipath + '/a/img/cor2/20110317/20110317_122400_14c2A.fts'        
        i0, h0 = sunpy.io._fits.read(p0)[0]
        if diff:
            i1, h1 = sunpy.io._fits.read(p1)[0]
            oimg = i1-i0
        else:
            oimg = i1
    # STEREO B
    elif sat==1:
        p0 = secchipath + '/b/img/cor2/20110317/20110317_123900_14c2B.fts' 
        p1 = secchipath + '/b/img/cor2/20110317/20110317_125400_14c2B.fts'    
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

    return oimg

