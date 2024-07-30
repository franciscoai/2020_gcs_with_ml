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
import pickle
mpl.use('Agg')

__author__ = "Francisco Iglesias"
__copyright__ = "Grupo de Estudios en Heliofisica de Mendoza - https://sites.google.com/um.edu.ar/gehme"
__version__ = "0.1"
__maintainer__ = "Francisco Iglesias"
__email__ = "franciscoaiglesias@gmail.com"


def load_data(dpath, occ_size, select=None):
    """
    Load all .fits files from directory dpath in order and in blocks grouped by the file basename using the last '_' separator
    :param dpath: directory path
    :return: images and headers lists
    :occ_size: # Artifitial occulter radius in pixels. Use 0 to avoid. [Stereo a/b C1, Stereo a/b C2, Lasco C2]
    :select: select the time instants to return, in order as read from dpath
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
            if 'COR1' in hdr['DETECTOR']:
                #occ_sizes.append(occ_size[0]) 90 maybe?
                print("occulter size is missing")
            if 'COR2' in hdr['DETECTOR']:
                if ('STEREO_A' in hdr['OBSRVTRY']):
                    occ_sizes.append(occ_size[0])
                if ('STEREO_B' in hdr['OBSRVTRY']):
                    occ_sizes.append(occ_size[1])
        elif 'INSTRUME' in hdr.keys():
            if 'LASCO' in hdr['INSTRUME'] and 'C2' in hdr['DETECTOR']:
                occ_sizes.append(occ_size[2])
        else:
            print('Error: Could not find the instrument name in the headers')
            breakpoint()
    #satpos and plotranges
        
        #satpos, plotranges = pyGCS.processHeaders([hdr])
        #breakpoint()
    satpos, plotranges = pyGCS.processHeaders(headers)  
    
    #loads mask properties from headers
    masks_prop = []
    #for i in range(len(headers)):
    #    prop = [headers[i]['NN_SCORE'], headers[i]['NN_C_ANG'], headers[i]['NN_W_ANG'], headers[i]['NN_APEX']]
    #    masks_prop.append(prop)

    # reshape the lists in blocks based on the file basename
    omasks =[]
    osatpos = []
    oplotranges = []
    ofilenames = []
    oocc_sizes = []
    omasks_prop = []
    # splits based on the last '_' separator, keeps the first part
    num_of_us = f.count('_')
    if num_of_us >=1:
        files_base = ["".join(f.rsplit('_')[0:-1]) for f in filenames]
    else:
        print('Error: The filenames of a single time instant must have the same basename and end with _0 (cor A), _1 (cor B) or _2 (lasco))')
    #breakpoint()
    for f in np.unique(files_base):
        idx = [i for i, x in enumerate(files_base) if x == f]
        omasks.append(np.array([masks[i] for i in idx]))
        #breakpoint()
        osatpos.append([satpos[i] for i in idx])
        oplotranges.append([plotranges[i] for i in idx])
        ofilenames.append([filenames[i] for i in idx])
        oocc_sizes.append([occ_sizes[i] for i in idx])
        omasks_prop.append([masks_prop[i] for i in idx])
    
    if select is not None:
        omasks = [omasks[i] for i in select]
        osatpos = [osatpos[i] for i in select]
        oplotranges = [oplotranges[i] for i in select]
        ofilenames = [ofilenames[i] for i in select]
        oocc_sizes = [oocc_sizes[i] for i in select]
        omasks_prop = [omasks_prop[i] for i in select]
    return omasks, ofilenames, osatpos, oplotranges, oocc_sizes, omasks_prop

def plot_to_png(ofile, fnames,omask, fitmask, manual_mask):
    """
    plots the original mask and the fitted masks to a png file
    """    
    color=['b','r','g','k','y','m','c','w','b','r','g','k','y','m','c','w']
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(omask[0]), np.nan)
    fig, axs = plt.subplots(3, 3, figsize=[10,10])
    axs = axs.ravel()
    for i in range(len(fnames)):
        axs[i].imshow(omask[i], vmin=0, vmax=1, cmap='gray', origin='lower')
        axs[i].axis('off')
        axs[i+3].imshow(fitmask[i], vmin=0, vmax=1, cmap='gray', origin='lower')        
        axs[i+3].axis('off')   
        axs[i+6].imshow(manual_mask[i], vmin=0, vmax=1, cmap='gray', origin='lower')  
        axs[i+6].axis('off')
        # adds a cross to the center of the image
        axs[i].plot(np.shape(omask[i])[0]/2., np.shape(omask[i])[1]/2., 'x', color='r')
        axs[i+3].plot(np.shape(omask[i])[0]/2., np.shape(omask[i])[1]/2., 'x', color='r')
        axs[i+6].plot(np.shape(omask[i])[0]/2., np.shape(omask[i])[1]/2., 'x', color='r')
        # masked = nans.copy()
        # masked[:, :][omask[i] > 0.1] = 0              
        # axs[i].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1, origin='lower')
        # masked = nans.copy()
        # masked[:, :][fitmask[i] > 0.1] = 0
        # axs[i+3].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1, origin='lower')
    axs[0].set_title(f'Cor A: {fnames[0]}')
    axs[1].set_title(f'Cor B: {fnames[1]}')
    axs[2].set_title(f'Lasco: {fnames[2]}')   
    axs[0].set_ylabel('Neural mask')
    axs[3].set_ylabel('Fit to neural mask')
    axs[6].set_ylabel('Manual fit')
    #plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

def plot_gcs_param_vs_time(fit_par, gcs_manual_par, opath, ylim=None):
    """
    Plots the 6 gcs parameters vs time
    :param fit_par: fit parameters, list of [CMElon, CMElat, CMEtilt,height, k, ang]
    :param gcs_manual_par: manual gcs parameters, list of [CMElon, CMElat, CMEtilt,height, k, ang]
    :param opath: output path
    :param ylim: y limits
    """
    fig, axs = plt.subplots(3, 2, figsize=[12,8])
    axs = axs.ravel()
    for t in range(len(fit_par)):
        for i in range(6):
            axs[i].plot(t, fit_par[t][i], 'o', color='r', label='fit')
            axs[i].plot(t, gcs_manual_par[t][i], 'o', color='b', label='manual')
            if ylim is not None:
                axs[i].set_ylim(ylim[i])
            if t==0 and i==0:
                axs[i].legend()
    axs[0].set_ylabel('CMElon')
    axs[1].set_ylabel('CMElat')
    axs[2].set_ylabel('CMEtilt')
    axs[3].set_ylabel('height')
    axs[4].set_ylabel('k')
    axs[5].set_ylabel('ang')
    plt.tight_layout()
    plt.savefig(os.path.join(opath, 'gcs_fit_vs_manual.png'))
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
        #breakpoint()
        error.append(np.mean((np.array(mask) - masks[i]), axis=(1,2))**2 /mask_total_px[i])
        #print(gcs_par, np.mean(error))
    return np.array(error).flatten()

############ Main
'''
Fits a filled masks created with GCS model to the data
'''
#Constants
dpath =  '/gehme/projects/2023_eeggl_validation/niemela_project/gcs_20100403_mask'
opath = dpath + '/gcs_fit'
select = [-4, -3, -2, -1] # select the time instants to fit, in order as read from dpath
manual_gcs = '/gehme/projects/2023_eeggl_validation/niemela_project/GCS_20130424'
imsize = [512, 512] # image size
gcs_par_range = [[-180,180],[-90,90],[-90,90],[1,50],[0.1,0.9], [1,80]] # bounds for the fit gcs parameters
occ_size = [50,75,90] # Artifitial occulter radius in pixels. Use 0 to avoid. [Stereo-A C2, Stereo-B C2, Lasco-C2]

# Load data
meas_masks, fnames, satpos, plotranges, occ_sizes, masks_prop= load_data(dpath, occ_size, select=select)
breakpoint()
mask_total_px = [np.sum(m, axis=(1,2)) for m in meas_masks] # total number of px in the mask

#loads manual gcs for IDL .sav file
#CMElon, CMElat, CMEtilt, height, k, ang
gcs_files= sorted(os.listdir(manual_gcs))
gcs_files =[f for f in gcs_files if f.endswith(".sav")]
gcs_files = sorted(gcs_files, key=lambda x: x[0] != 'm')
gcs_manual_par = []
for f in gcs_files:
    temp = readsav(os.path.join(manual_gcs, f))
    #breakpoint()
    gcs_manual_par.append([np.degrees(float(temp['sgui']['lon'])), np.degrees(float(temp['sgui']['lat'])), np.degrees(float(temp['sgui']['rot'])),
                float(temp['sgui']['hgt']), float(temp['sgui']['rat']), np.degrees(float(temp['sgui']['han']))])
# keeps only select
gcs_manual_par = [gcs_manual_par[i] for i in select]

# read /gehme/projects/2023_eeggl_validation/niemela_project/gcs_events
# guardar en gcs_manual



# crate opath
os.makedirs(opath, exist_ok=True)

# fits gcs model to all images simultaneosly
# bounds
up_bounds= np.array([gcs_par_range[0][1], gcs_par_range[1][1], gcs_par_range[2][1], gcs_par_range[4][1], gcs_par_range[5][1]])
up_bounds= np.append(up_bounds, np.full(len(meas_masks), gcs_par_range[3][1]))
low_bounds= np.array([gcs_par_range[0][0], gcs_par_range[1][0], gcs_par_range[2][0], gcs_par_range[4][0], gcs_par_range[5][0]])
low_bounds= np.append(low_bounds, np.full(len(meas_masks), gcs_par_range[3][0]))
#inital  conditions from masks_prop
# gcs_param_ini = gcs_manual_par[-2]
# ini_cond =  np.array([gcs_param_ini[0], gcs_param_ini[1], gcs_param_ini[2], gcs_param_ini[4], gcs_param_ini[5]])
# ini_cond = np.append(ini_cond, np.arange(len(meas_masks))+gcs_param_ini[3])
#breakpoint()
# CMElat from LASCO maks CPA
ini_lat = np.median([masks_prop[i][2][1] for i in range(len(masks_prop))])
print(ini_lat)
# change from 0 to 360 to -90 to 90
if ini_lat > 90 and ini_lat < 180:
    ini_lat = 180 - ini_lat 
elif ini_lat > 180 and ini_lat < 270:
    ini_lat = -(ini_lat - 180)
elif ini_lat > 270 and ini_lat < 360:
    ini_lat -= 360

# ang from the min mask AW
ini_ang = np.min([masks_prop[i][j][2] for i in range(len(masks_prop)) for j in range(len(masks_prop[i]))])/2.
# heights from each LASCO mask height
ini_height = [masks_prop[i][2][3] for i in range(len(masks_prop))]
# k at half the bounds
ini_k = (gcs_par_range[4][0] + gcs_par_range[4][1])/2.
# CMElon from LASCO mask CPA
ini_lon = np.median([masks_prop[2][0]])
if ini_lon < 90 or ini_lon > 270:
    ini_lon = 90
else:
    ini_lon = -90
ini_cond = np.array([ini_lon, ini_lat, 0, ini_k, ini_ang]+ini_height).flatten()
# if initial conditions are outside bounds use the closest
if np.any(ini_cond < low_bounds) or np.any(ini_cond > up_bounds):
    print('Warning: Initial conditions are outside bounds. Using closest bounds')
    ini_cond = np.clip(ini_cond, low_bounds, up_bounds)
#breakpoint()
print('Fitting GCS model with initial conditions: ', ini_cond)
#usar metodo lm
fit=least_squares(gcs_mask_error, ini_cond , method='trf', 
                kwargs={'satpos': satpos, 'plotranges': plotranges, 'masks': meas_masks, 'imsize': imsize, 'mask_total_px':mask_total_px, 'occ_size':occ_sizes}, 
                verbose=2, bounds=(low_bounds,up_bounds), diff_step=.5, xtol=1e-15) #, x_scale=scales)
ini_cond = fit.x
fit=least_squares(gcs_mask_error, ini_cond , method='trf', 
                kwargs={'satpos': satpos, 'plotranges': plotranges, 'masks': meas_masks, 'imsize': imsize, 'mask_total_px':mask_total_px, 'occ_size':occ_sizes}, 
                verbose=2, bounds=(low_bounds,up_bounds), diff_step=.5, xtol=1e-15) #, x_scale=scales)
print('The fit parameters are: ', fit.x)

# saves to pickle
with open(os.path.join(opath, 'gcs_fit.pkl'), 'wb') as f:
    pickle.dump(fit, f)

# plots manual and fit gcs param vs time
gcs_fit_par = []
for i in range(len(meas_masks)):
    gcs_fit_par.append([fit.x[0], fit.x[1], -fit.x[2], fit.x[5+i], fit.x[3], fit.x[4]]) # TODO the tilt signs seems to be OPOSITE in pyGCS???
plot_gcs_param_vs_time(gcs_fit_par, gcs_manual_par, opath, ylim=gcs_par_range)

# plots the fit mask along with the original masks
for i in range(len(meas_masks)):
    gcs_param = [fit.x[0], fit.x[1], fit.x[2], fit.x[5+i], fit.x[3], fit.x[4]]
    mask = maskFromCloud_3d(gcs_param, satpos[i], imsize, plotranges[i], occ_size=occ_sizes[i])
    mask_manual = maskFromCloud_3d(gcs_manual_par[i], satpos[i], imsize, plotranges[i], occ_size=occ_sizes[i])
    cfname = fnames[i][0].split('_')[0] + '_' + fnames[i][0].split('_')[1]
    ofile = os.path.join(opath, f'{cfname}_gcs_fit.png')
    plot_to_png(ofile, fnames[i], meas_masks[i], mask, mask_manual)
