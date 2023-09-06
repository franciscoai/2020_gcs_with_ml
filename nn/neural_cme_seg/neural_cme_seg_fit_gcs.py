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


def load_data(dpath, occ_size):
    """
    Load all .fits files from directory dpath in order and in blocks grouped by the file basename using the last '_' separator
    :param dpath: directory path
    :return: images and headers lists
    :occ_size: # Artifitial occulter radius in pixels. Use 0 to avoid. [Stereo a/b C1, Stereo a/b C2, Lasco C2]
    """
    masks = []
    headers = []
    filenames = []
    occ_sizes = []
    files = sorted(os.listdir(dpath))
    files =[f for f in files if f.endswith(".fits")]
    for f in files:
        filenames.append(f)
        hdu = fits.open(os.path.join(dpath, f))
        masks.append(hdu[0].data)
        hdr = hdu[0].header
        headers.append(hdr) 
        if 'OBSRVTRY' in hdr.keys():
            if ('STEREO_A' in hdr['OBSRVTRY']) or ('STEREO_B' in hdr['OBSRVTRY']):
                if 'COR1' in hdr['DETECTOR']:
                    occ_sizes.append(occ_size[0])
                elif 'COR2' in hdr['DETECTOR']:
                    occ_sizes.append(occ_size[1])
        elif 'INSTRUME' in hdr.keys():
            if 'LASCO' in hdr['INSTRUME'] and 'C2' in hdr['DETECTOR']:
                occ_sizes.append(occ_size[2])
        else:
            print('Error: Could not find the instrument name in the headers')
            breakpoint()
    #satpos and plotranges
    satpos, plotranges = pyGCS.processHeaders(headers)  

    # reshape the lists in blocks based on the file basename
    omasks =[]
    osatpos = []
    oplotranges = []
    ofilenames = []
    oocc_sizes = []
    # splits based on the last '_' separator, keeps the first part
    files_base = [f.split('_')[0] + '_' + f.split('_')[1] for f in filenames]
    for f in np.unique(files_base):
        idx = [i for i, x in enumerate(files_base) if x == f]
        omasks.append(np.array([masks[i] for i in idx]))
        osatpos.append([satpos[i] for i in idx])
        oplotranges.append([plotranges[i] for i in idx])
        ofilenames.append([filenames[i] for i in idx])
        oocc_sizes.append([occ_sizes[i] for i in idx])
  
    return omasks, ofilenames, osatpos, oplotranges, oocc_sizes

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
        # adds a cross to the center of the image
        axs[i].plot(np.shape(omask[i])[0]/2., np.shape(omask[i])[1]/2., 'x', color='r')
        axs[i+3].plot(np.shape(omask[i])[0]/2., np.shape(omask[i])[1]/2., 'x', color='r')
        # adds a grid to the image
        axs[i].grid(color='r', linestyle='-', linewidth=0.5)
        axs[i+3].grid(color='r', linestyle='-', linewidth=0.5)   
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

def gcs_mask_error(gcs_par, satpos, plotranges, masks, mask_total_px, imsize, occ_size):
    """
    Computes the error between the input masks and the maks from GCS model
    :param gcs_par: GCS model parameters. The param are: CMElon, CMElat, CMEtilt, k, ang, , height0, height1, height2, ...
                    For all images they are all the same except for height, wich is different for each set of three images (time instant)
    :param satpos: satellite position
    :param plotranges: plot ranges
    :param masks: masks to compare
    :param mask_total_px: total number of px in each mask
    :param imsize: image size
    :param occ_size: occulter size
    :return: error
    """
    error = []
    for i in range(len(masks)):
        this_gcs_par = [gcs_par[0], gcs_par[1], gcs_par[2], gcs_par[5+i], gcs_par[3], gcs_par[4]] 
        mask = maskFromCloud_3d(this_gcs_par, satpos[i], imsize, plotranges[i], occ_size=occ_size[i])
        error.append(np.sum(np.abs(np.array(mask) - masks[i]), axis=(1,2))/mask_total_px[i])
        #print(gcs_par, np.mean(error))
    return np.array(error).flatten()

############ Main
'''
Fits a filled masks created with GCS model to the data
'''
#Constants
dpath =  '/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_exp_paper_filtered/GCS_20130424_filter_True'
opath = dpath + '/gcs_fit'
manual_gcs = '/gehme/projects/2019_cme_expansion/repo_fran/2020_cme_expansion/GCSs/GCS_20130424/1.sav'
imsize = [512, 512] # image size
gcs_par_range = [[-180,180],[-90,90],[-90,90],[1,50],[0.1,0.9], [1,80]] # bounds for the fit gcs parameters
occ_size = [90,35,75] # Artifitial occulter radius in pixels. Use 0 to avoid. [Stereo a/b C1, Stereo a/b C2, Lasco C2]

# Load data
meas_masks, fnames, satpos, plotranges, occ_sizes= load_data(dpath, occ_size)
mask_total_px = [np.sum(m, axis=(1,2)) for m in meas_masks] # total number of px in the mask
#loads manual gcs for IDL .save file
#CMElon, CMElat, CMEtilt, height, k, ang
gcs_param_ini = readsav(manual_gcs, python_dict=True)
gcs_param_ini = [np.degrees(float(gcs_param_ini['sgui']['lon'])), np.degrees(float(gcs_param_ini['sgui']['lat'])), np.degrees(float(gcs_param_ini['sgui']['rot'])),
                float(gcs_param_ini['sgui']['hgt']), float(gcs_param_ini['sgui']['rat']), np.degrees(float(gcs_param_ini['sgui']['han']))]

# crate opath
os.makedirs(opath, exist_ok=True)

# fits gcs model to all images simultaneosly
ini_cond =  np.array([gcs_param_ini[0], gcs_param_ini[1], gcs_param_ini[2], gcs_param_ini[4], gcs_param_ini[5]])
ini_cond = np.append(ini_cond, np.arange(len(meas_masks))+2.)
up_bounds= np.array([gcs_par_range[0][1], gcs_par_range[1][1], gcs_par_range[2][1], gcs_par_range[4][1], gcs_par_range[5][1]])
up_bounds= np.append(up_bounds, np.full(len(meas_masks), gcs_par_range[3][1]))
low_bounds= np.array([gcs_par_range[0][0], gcs_par_range[1][0], gcs_par_range[2][0], gcs_par_range[4][0], gcs_par_range[5][0]])
low_bounds= np.append(low_bounds, np.full(len(meas_masks), gcs_par_range[3][0]))

#scales = np.append(np.array([1, 1, 1, 0.001, 0.001]), np.full(len(meas_masks), 0.001))

#ini_cond = low_bounds + (up_bounds - low_bounds)/2.*np.random.rand(len(ini_cond))

print('Fitting GCS model with initial conditions: ', ini_cond)
fit=least_squares(gcs_mask_error, ini_cond , method='trf', 
                  kwargs={'satpos': satpos, 'plotranges': plotranges, 'masks': meas_masks, 'imsize': imsize, 'mask_total_px':mask_total_px, 'occ_size':occ_sizes}, 
                  verbose=2, bounds=(low_bounds,up_bounds), diff_step=1., xtol=1e-11) #, x_scale=scales)
print('The fit parameters are: ', fit.x)

# plots the fit mask along with the original masks
for i in range(len(meas_masks)):
    gcs_param = [fit.x[0], fit.x[1], fit.x[2], fit.x[5+i], fit.x[3], fit.x[4]]
    mask = maskFromCloud_3d(gcs_param, satpos[i], imsize, plotranges[i], occ_size=occ_sizes[i])
    cfname = fnames[i][0].split('_')[0] + '_' + fnames[i][0].split('_')[1]
    ofile = os.path.join(opath, f'{cfname}_gcs_fit.png')
    plot_to_png(ofile, fnames[i], meas_masks[i], mask)
