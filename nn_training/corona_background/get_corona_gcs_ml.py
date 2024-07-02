#!/usr/bin/env python
# coding: utf-8

# ## Librerias
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import scipy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ext_libs.rebin import rebin
from astropy.io import fits


def find_matches(paths, tolerance):
    """
    Finds matches of dates and times in the files of the comparison paths
    """
    matches = []
    # listir all paths
    cor2b_path, cor2a_path, lasco_path = paths
    # list all files in the paths
    cor2b_files = [file for file in os.listdir(cor2b_path) if file.endswith('.fits')]
    cor2a_files = [file for file in os.listdir(cor2a_path) if file.endswith('.fits')]
    lasco_files = [file for file in os.listdir(lasco_path) if file.endswith('.fits')]

    for cor2b_file in cor2b_files:
        # Gets the date and time of the file
        cor2b_timestamp = datetime.strptime(cor2b_file.split("_")[0] + "_" + cor2b_file.split("_")[1], "%Y%m%d_%H%M%S")
        # Check for matches
        cor2a_timestamps = [datetime.strptime(cor2a_file.split("_")[0] + "_" + cor2a_file.split("_")[1], "%Y%m%d_%H%M%S") for cor2a_file in cor2a_files]
        cor2a_timestamps_diff = [abs(cor2b_timestamp - cor2a_timestamp) for cor2a_timestamp in cor2a_timestamps]
        cor2a_file = cor2a_files[np.argmin(cor2a_timestamps_diff)] if min(cor2a_timestamps_diff) <= tolerance else None
        if cor2a_file is not None:
            lasco_timestamps = [datetime.strptime(lasco_file.split(".")[0], "%Y%m%d_%H%M%S") for lasco_file in lasco_files]
            lasco_timestamps_diff = [abs(cor2b_timestamp - lasco_timestamp) for lasco_timestamp in lasco_timestamps]
            lasco_file = lasco_files[np.argmin(lasco_timestamps_diff)] if min(lasco_timestamps_diff) <= tolerance else None
            if lasco_file is not None:
                matches.append((cor2b_file, cor2a_file, lasco_file))
    return matches

def get_corona(imsize=None, rnd_rot=False, custom_headers=False, random_state=None):
    '''
    Returns a measured "quiet" (with no CME) solar corona observed by satelitte sat, the implemented instruments are

    sat=
        0: Stereo B Cor2
        1: Stereo A Cor2
        2: Lasco C2

    OPTIONS:
        diff: Set to True to return a time differential corona.
        imsize: Set to [x,y] to imsize the output image to that size
        rnd_rot: Set to rotate the ouput image by a random angle around the central pixel

    '''
    # CONSTANTS  
    COR2B_PATH = "/gehme/projects/2020_gcs_with_ml/data/corona_background_affects/cor2/cor2_b"
    COR2A_PATH = "/gehme/projects/2020_gcs_with_ml/data/corona_background_affects/cor2/cor2_a"
    LASCO_PATH = "/gehme/projects/2020_gcs_with_ml/data/corona_background_affects/lasco/c2/3VP"
    H_COR2B = "/gehme/data/stereo/secchi/L1/b/img/cor2/20130209/20130209_062400_14c2B.fts"
    H_COR2A = "/gehme/data/stereo/secchi/L1/a/img/cor2/20130209/20130209_062400_14c2A.fts" 
    H_LASCO = "/gehme/data/soho/lasco/level_1/c2/20130209/25447666.fts" 
    SIZE_OCC = [3.3, 3.3, 2.2]# Occulters size for [sat1, sat2 ,sat3] in [Rsun]
    OCC_CENTER = [(30,-15),(0,-5),(0,0)] # [(38,-15),(0,-5),(0,0)] # (y,x)
    # size_occ=[1.4, 1.4, 2]# Occulters size for [sat1, sat2 ,sat3] in [Rsun] 
    TIME_DIFF_THRESHOLD = timedelta(hours=24)

    
    # Generate list of triplets
    matches = find_matches([COR2B_PATH, COR2A_PATH, LASCO_PATH], TIME_DIFF_THRESHOLD)
    
    # Get random triplet
    if random_state is None:
        triplet = np.random.randint(0, len(matches)-1)
    else:
        triplet = random_state.randint(0, len(matches)-1)
    triplet = matches[triplet]

    cor2b_file, cor2a_file, lasco_file = triplet
    cor2b_path = COR2B_PATH + "/" + cor2b_file
    cor2a_path = COR2A_PATH + "/" + cor2a_file
    lasco_path = LASCO_PATH + "/" + lasco_file

    # Get date of triplet
    cor2b_date = datetime.strptime(cor2b_file.split("_")[0] + "_" + cor2b_file.split("_")[1], "%Y%m%d_%H%M%S")
    cor2a_date = datetime.strptime(cor2a_file.split("_")[0] + "_" + cor2a_file.split("_")[1], "%Y%m%d_%H%M%S")
    lasco_date = datetime.strptime(lasco_file.split(".")[0], "%Y%m%d_%H%M%S")

    # Read images
    cor2b_img = fits.open(cor2b_path)[0].data
    cor2a_img = fits.open(cor2a_path)[0].data
    lasco_img = fits.open(lasco_path)[0].data

    # Read headers
    if custom_headers:
        h_cor2b= fits.getheader(H_COR2B)
        h_cor2a= fits.getheader(H_COR2A)
        h_lasco= fits.getheader(H_LASCO)
    else:
        h_cor2b = fits.open(cor2b_path)[0].header
        h_cor2a = fits.open(cor2a_path)[0].header
        h_lasco = fits.open(lasco_path)[0].header

    # rotate images
    if rnd_rot:
        cor2b_img = scipy.ndimage.rotate(cor2b_img, np.random.randint(low=0, high=360), reshape=False)
        cor2a_img = scipy.ndimage.rotate(cor2a_img, np.random.randint(low=0, high=360), reshape=False)
        lasco_img = scipy.ndimage.rotate(lasco_img, np.random.randint(low=0, high=360), reshape=False)

 
    # shift images
    shift_values = OCC_CENTER[0]
    cor2b_img = np.roll(cor2b_img, shift_values[0], axis=0)
    cor2b_img = np.roll(cor2b_img, shift_values[1], axis=1)

    shift_values = OCC_CENTER[1]
    cor2a_img = np.roll(cor2a_img, shift_values[0], axis=0)
    cor2a_img = np.roll(cor2a_img, shift_values[1], axis=1)

    shift_values = OCC_CENTER[2]
    lasco_img = np.roll(lasco_img, shift_values[0], axis=0)
    lasco_img = np.roll(lasco_img, shift_values[1], axis=1)

    # rebin images
    if imsize is not None:
        cor2b_img = rebin(cor2b_img,imsize,operation='mean')
        cor2a_img = rebin(cor2a_img,imsize,operation='mean')
        lasco_img = rebin(lasco_img,imsize,operation='mean')
    
    # stack data
    imgs_data = [cor2b_img, cor2a_img, lasco_img]
    headers = [h_cor2b, h_cor2a, h_lasco]
    size_occ = SIZE_OCC
    dates = [cor2b_date, cor2a_date, lasco_date]

    return imgs_data, headers, size_occ, dates
