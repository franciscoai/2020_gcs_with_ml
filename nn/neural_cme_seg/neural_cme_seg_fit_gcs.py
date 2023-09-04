import os
import numpy as np
import cv2
import matplotlib as mpl
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from nn.utils.gcs_mask_generator import maskFromCloud_3d
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
    filenames = np.array(filenames)
    satpos = np.array(satpos)
    plotranges = np.array(plotranges)
    images = images.reshape(-1, nsat, images.shape[1], images.shape[2])
    filenames = filenames.reshape(-1, nsat)
    satpos = satpos.reshape(-1, nsat, satpos.shape[1])
    plotranges = plotranges.reshape(-1, nsat, plotranges.shape[1])
    # transform back to list
    images = images.tolist()
    filenames = filenames.tolist()
    satpos = satpos.tolist()
    plotranges = plotranges.tolist()

    return images, filenames, satpos, plotranges

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
        mask = maskFromCloud_3d(gcs_par, 0, satpos, plotranges, masks.shape[1:])
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

gcs_param_ini = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # initial guess for GCS model parameters
mask = maskFromCloud_3d(gcs_param_ini, satpos[0], imsize, plotranges[0]) # initial mask
