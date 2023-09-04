import os
import numpy as np
import cv2
import matplotlib as mpl
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
from scipy.io import readsav
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from nn.utils.gcs_mask_generator import maskFromCloud_3d
from pyGCS_raytrace import pyGCS
mpl.use('Agg')

__author__ = "Francisco Iglesias"
__copyright__ = "Grupo de Estudios en Heliofisica de Mendoza - https://sites.google.com/um.edu.ar/gehme"
__version__ = "0.1"
__maintainer__ = "Francisco Iglesias"
__email__ = "franciscoaiglesias@gmail.com"


def load_data(dpath):
    """
    Load all .fits files from directory dpath in order and in blocks grouped by the file basename using the last '_' separator
    :param dpath: directory path
    :return: images and headers lists
    """
    masks = []
    headers = []
    filenames = []
    files = sorted(os.listdir(dpath))
    files =[f for f in files if f.endswith(".fits")]
    for f in files:
        filenames.append(f)
        hdu = fits.open(os.path.join(dpath, f))
        masks.append(hdu[0].data)
        hdr = hdu[0].header
        headers.append(hdr) 

    #satpos and plotranges
    satpos, plotranges = pyGCS.processHeaders(headers)  

    # reshape the lists in blocks based on the file basename
    omasks =[]
    osatpos = []
    oplotranges = []
    ofilenames = []
    # splits based on the last '_' separator, keeps the first part
    files_base = [f.split('_')[0] + '_' + f.split('_')[1] for f in filenames]
    for f in np.unique(files_base):
        idx = [i for i, x in enumerate(files_base) if x == f]
        omasks.append([masks[i] for i in idx])
        osatpos.append([satpos[i] for i in idx])
        oplotranges.append([plotranges[i] for i in idx])
        ofilenames.append([filenames[i] for i in idx])
  
    return omasks, ofilenames, osatpos, oplotranges

def plot_to_png(ofile, fnames,omask, fitmask):
    """
    plots the original mask and the fitted masks to a png file
    """    
    color=['b','r','g','k','y','m','c','w','b','r','g','k','y','m','c','w']
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(omask[0]), np.nan)
    fig, axs = plt.subplots(2, 3, figsize=[20,10])
    axs = axs.ravel()
    for i in range(len(fnames)):
        axs[i].imshow(omask[i], vmin=0, vmax=1, cmap='gray', origin='lower')
        axs[i].axis('off')
        axs[i+3].imshow(fitmask[i], vmin=0, vmax=1, cmap='gray', origin='lower')        
        axs[i+3].axis('off')        
        # masked = nans.copy()
        # masked[:, :][omask[i] > 0.1] = 0              
        # axs[i].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1, origin='lower')
        # masked = nans.copy()
        # masked[:, :][fitmask[i] > 0.1] = 0
        # axs[i+3].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1, origin='lower')
    axs[0].set_title(f'Cor A: {fnames[0]}')
    axs[1].set_title(f'Cor B: {fnames[1]}')
    axs[2].set_title(f'Lasco: {fnames[2]}')   

    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

def mask_error(gcs_par, satpos, plotranges, masks, imsize):
    """
    Computes the error between the input masks and the maks from GCS model
    :param gcs_par: GCS model parameters. The param are: CMElon, CMElat, CMEtilt, k, ang, , height0, height1, height2, ...
                    For all images they are all the same except for height, wich is different for each set of three images (time instant)
    :param satpos: satellite position
    :param plotranges: plot ranges
    :param masks: masks to compare
    :return: error
    """
    error = 0
    for i in range(masks.shape[0]):
        this_gcs_par = [gcs_par[0], gcs_par[1], gcs_par[2], gcs_par[5+i], gcs_par[3], gcs_par[4]] 
        mask = maskFromCloud_3d(this_gcs_par, satpos, imsize, plotranges[0])
        error += mask - masks[i]
    return error

############ Main
'''
Fits a filled masks created with GCS model to the data
'''
#Constants
dpath =  '/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_exp_paper_filtered/GCS_20130424_filter_True'
opath = dpath + '/gcs_fit'
manual_gcs = '/gehme/projects/2019_cme_expansion/repo_fran/2020_cme_expansion/GCSs/GCS_20130424/1.sav'
imsize = [512, 512] # image size

# Load data
omask, fnames, satpos, plotranges = load_data(dpath)

#loads manual gcs for IDL .save file
#CMElon, CMElat, CMEtilt, height, k, ang
gcs_param_ini = readsav(manual_gcs, python_dict=True)
gcs_param_ini = [np.degrees(float(gcs_param_ini['sgui']['lon'])), np.degrees(float(gcs_param_ini['sgui']['lat'])), np.degrees(float(gcs_param_ini['sgui']['rot'])),
                float(gcs_param_ini['sgui']['hgt']), float(gcs_param_ini['sgui']['rat']), np.degrees(float(gcs_param_ini['sgui']['han']))]

# crate opath
os.makedirs(opath, exist_ok=True)

# fits gcs model to all images simultaneosly
mask = maskFromCloud_3d(gcs_param_ini, satpos[1], imsize, plotranges[1])
ofile = os.path.join(opath, 'gcs_fit.png')
plot_to_png(ofile, fnames[1], omask[1], mask)

breakpoint()
