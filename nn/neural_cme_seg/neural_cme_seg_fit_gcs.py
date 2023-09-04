import os
import numpy as np
import cv2
import torch
import matplotlib as mpl
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from nn.utils.gcs_mask_generator import maskFromCloud
from pyGCS_raytrace import pyGCS
mpl.use('Agg')

__author__ = "Francisco Iglesias"
__copyright__ = "Grupo de Estudios en Heliofisica de Mendoza - https://sites.google.com/um.edu.ar/gehme"
__version__ = "0.1"
__maintainer__ = "Francisco Iglesias"
__email__ = "franciscoaiglesias@gmail.com"


def load_data(dpath, nsat=3):
    """
    Load all .fits files from directory dpath in order
    :param dpath: directory path
    :nsat: number of satellites (views) to use. Rearanges the lists in blocks of nsat
    :return: images and headers lists
    """
    images = []
    headers = []
    filenames = []
    for filename in sorted(os.listdir(dpath)):
        if filename.endswith(".fits"):
            filenames.append(filename)
            hdu = fits.open(os.path.join(dpath, filename))
            images.append(hdu[0].data)
            hdr = hdu[0].header
            headers.append(hdr) 

    #satpos and plotranges
    satpos, plotranges = pyGCS.processHeaders(headers)  

    # rearrange lists in blocks of nsat
    images = np.array(images)
    headers = np.array(headers)
    filenames = np.array(filenames)
    images = images.reshape(-1, nsat, images.shape[1], images.shape[2])
    headers = headers.reshape(-1, nsat)
    filenames = filenames.reshape(-1, nsat)

    return images, fnames, satpos, plotranges

def mask_error(gcs_par, satpos, plotranges, masks):
    """
    Computes the error between the masks and the GCS model
    :param gcs_par: GCS model parameters
    :param satpos: satellite position
    :param plotranges: plot ranges
    :param masks: masks to compare
    :return: error
    """
    error = 0
    for i in range(masks.shape[0]):
        mask = maskFromCloud(gcs_par, 0, satpos, plotranges, masks.shape[1:])
        error += np.sum(np.abs(mask - masks[i]))
    return error


############ Main
'''
Fits a filled masks created with GCS model to the data
'''
#Constants
dpath =  '/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_exp_paper_filtered/GCS_20130424_filter_True'
nsat = 3 # number of satellites (views) to use
imsize = [512, 512] # image size

# Load data
images, fnames, satpos, plotranges = load_data(dpath, nsat=nsat)

#
breakpoint()
