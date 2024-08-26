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
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
from sunpy.coordinates.sun import carrington_rotation_number
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
    dates = []
    files = sorted(os.listdir(dpath))
    files =[f for f in files if f.endswith(".fits")]
    #breakpoint()
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
                dates.append(datetime.strptime(hdr['date-obs'], '%Y-%m-%dT%H:%M:%S.%f'))
        elif 'INSTRUME' in hdr.keys():
            if 'LASCO' in hdr['INSTRUME'] and 'C2' in hdr['DETECTOR']:
                occ_sizes.append(occ_size[2])
                dates.append(datetime.strptime(hdr['date-obs'], '%Y-%m-%dT%H:%M:%S.%f'))
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
        #omasks_prop.append([masks_prop[i] for i in idx])
    if select is not None:
        omasks = [omasks[i] for i in select]
        osatpos = [osatpos[i] for i in select]
        oplotranges = [oplotranges[i] for i in select]
        ofilenames = [ofilenames[i] for i in select]
        oocc_sizes = [oocc_sizes[i] for i in select]
        dates      = [dates[i] for i in select]
        #omasks_prop = [omasks_prop[i] for i in select]
    return omasks, ofilenames, osatpos, oplotranges, oocc_sizes, omasks_prop,dates

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
        axs[i+2].imshow(fitmask[i], vmin=0, vmax=1, cmap='gray', origin='lower')        
        axs[i+2].axis('off')   
        axs[i+4].imshow(manual_mask[i], vmin=0, vmax=1, cmap='gray', origin='lower')  
        axs[i+4].axis('off')
        # adds a cross to the center of the image
        axs[i].plot(np.shape(omask[i])[0]/2., np.shape(omask[i])[1]/2., 'x', color='r')
        axs[i+2].plot(np.shape(omask[i])[0]/2., np.shape(omask[i])[1]/2., 'x', color='r')
        axs[i+4].plot(np.shape(omask[i])[0]/2., np.shape(omask[i])[1]/2., 'x', color='r')
        # masked = nans.copy()
        # masked[:, :][omask[i] > 0.1] = 0              
        # axs[i].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1, origin='lower')
        # masked = nans.copy()
        # masked[:, :][fitmask[i] > 0.1] = 0
        # axs[i+3].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1, origin='lower')
    axs[0].set_title(f'Cor A: {fnames[0]}')
    axs[1].set_title(f'Cor B: {fnames[1]}')
    #axs[2].set_title(f'Lasco: {fnames[2]}')   
    axs[0].set_ylabel('Neural mask')
    axs[2].set_ylabel('Fit to neural mask')
    axs[4].set_ylabel('Manual fit')
    #plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

def plot_to_png_diego(ofile, fnames,omask, fitmask, manual_mask):
    """
    plots the original mask and the fitted masks to a png file
    """    
    color=['b','r','g','k','y','m','c','w','b','r','g','k','y','m','c','w']
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(omask[0]), np.nan)
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=[10,10])
    axs = axs.ravel()
    #j=0
    for j in range(len(fnames)):
        #axs[j].imshow(omask[j][0,:,:], vmin=0, vmax=1, cmap='gray', origin='lower')
        #fliped_image = np.flip(np.flip(omask[j][0,:,:],axis=0),axis=1)
        fliped_image = np.flip(omask[j][0,:,:],axis=0)
        axs[j].imshow(fliped_image, vmin=0, vmax=1, cmap='gray', origin='lower')
        axs[j].axis('off')
        axs[j+2].imshow(fitmask[j][0], vmin=0, vmax=1, cmap='Reds', origin='lower')        
        axs[j+2].axis('off')   
        axs[j+4].imshow(manual_mask[j][0], vmin=0, vmax=1, cmap='Blues', origin='lower')  
        axs[j+4].axis('off')
        #j=j+3
        # adds a cross to the center of the image
        
        axs[j].plot(  np.shape(omask[j][0,:,:])[0]/2., np.shape(omask[j][0,:,:])[1]/2., 'x', color='r')
        axs[j+2].plot(np.shape(omask[j][0,:,:])[0]/2., np.shape(omask[j][0,:,:])[1]/2., 'x', color='r')
        axs[j+4].plot(np.shape(omask[j][0,:,:])[0]/2., np.shape(omask[j][0,:,:])[1]/2., 'x', color='r')
        # masked = nans.copy()
        # masked[:, :][omask[i] > 0.1] = 0              
        # axs[i].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1, origin='lower')
        # masked = nans.copy()
        # masked[:, :][fitmask[i] > 0.1] = 0
        # axs[i+3].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1, origin='lower')
    axs[0].set_title(f'Cor A: {fnames[0]}')
    axs[1].set_title(f'Cor B: {fnames[1]}')
    #axs[2].set_title(f'Lasco: {fnames[2]}')   
    axs[0].set_ylabel('Neural mask')
    axs[1].set_ylabel('Fit to neural mask')
    #axs[4].set_ylabel('Manual fit')
    #plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

def plot_gcs_param_vs_time(fit_par, gcs_manual_par,dates,tiempos, opath, ylim=None):
    """
    Plots the 6 gcs parameters vs time
    :param fit_par: fit parameters, list of [CMElon, CMElat, CMEtilt,height, k, ang]
    :dates correspond to fit_par
    :tiempos correspond to gcs_manual_par
    :param gcs_manual_par: manual gcs parameters, list of [CMElon, CMElat, CMEtilt,height, k, ang]
    :param opath: output path
    :param ylim: y limits
    """
    fig, axs = plt.subplots(3, 2, figsize=[12,8])
    axs = axs.ravel()
    xfmt = mdates.DateFormatter('%H:%M')
    x_dates   = [mdates.date2num(dt) for dt in dates]
    x_tiempos = [mdates.date2num(dt) for dt in tiempos]
    for t in range(len(fit_par)):
        for i in range(6):
            axs[i].plot(x_dates[t], fit_par[t][i], 'o', color='r', label='fit')
            #axs[i].plot(tiempos[t], gcs_manual_par[t][i], 'x', color='b', label='manual')
            if ylim is not None:
                axs[i].set_ylim(ylim[i])
            if t==0 and i==0:
                axs[i].legend()
            axs[i].xaxis.set_major_formatter(xfmt)

    for t in range(len(gcs_manual_par)):
        for i in range(6):
            #axs[i].plot(x_dates[t], fit_par[t][i], 'o', color='r', label='fit')
            axs[i].plot(x_tiempos[t], gcs_manual_par[t][i], 'x', color='b', label='manual')
            #if ylim is not None:
            #    axs[i].set_ylim(ylim[i])
            if t==0 and i==0:
                axs[i].legend()
            axs[i].xaxis.set_major_formatter(xfmt)

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
        error.append(np.mean((np.array(mask) - masks[i]), axis=(1,2))**2)# /mask_total_px[i])
        print(gcs_par, np.mean(error))
    return np.array(error).flatten()

def rms_difference(array1, array2):
    """Calculates the RMS value of the element-wise difference between two arrays.
    Args:
        array1: The first NumPy array.
        array2: The second NumPy array.
    Returns:
        The RMS value of the difference.
    """
    diff = array1 - array2
    return np.sqrt(np.mean(diff**2))

def gcs_mask_error_diego(gcs_par, satpos, plotranges, masks, imsize, occ_size):
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
    
    #Para cada par o triplete de imagenes, que conrresponden al mismo tiempo, asignarle la misma altura.
    #En caso contrario se ajustan tantas alturas como satpos haya para ese mismo tiempo.
    # usar gcs_par[5+j] con j=j+1 para los i pares. 
    j=0
    for i in range(len(masks)):
        #this_gcs_par = [gcs_par[0], gcs_par[1], gcs_par[2], gcs_par[5+i], gcs_par[3], gcs_par[4]] 
        this_gcs_par = [gcs_par[0], gcs_par[1], gcs_par[2], gcs_par[5+j], gcs_par[3], gcs_par[4]]
        mask = maskFromCloud_3d(this_gcs_par, satpos[i], imsize, plotranges[i], occ_size=occ_size[i])

        #RMS
        #error.append(np.mean((np.array(mask) - masks[i]), axis=(1,2))**2)# /mask_total_px[i])
        #error.append( rms_difference( np.array(mask[0]), np.array(masks[i][0]) ) )
        error.append( rms_difference( np.array(mask[0]), np.flip(masks[i][0],axis=0) ) )
                
        #IOU^-1
        fliped_image = np.flip(masks[i][0],axis=0)
        intersection = np.logical_and(mask[0], fliped_image)
        union = np.logical_or(mask[0], fliped_image)
        #error.append(np.sum(intersection) / np.sum(union))
        #error.append(np.sum(union)/np.sum(intersection))
        
        #agregar opcion de vector que permita pesar entre o y 1 cor2a - b y C2.
        print(gcs_par, np.mean(error))
        if i%2==1:
            j=j+1
    return np.array(error).flatten()

def ckcarr_to_stony(carr_ck,observer_time):
    #christina Kay carrington coordinate system es similar to carrington pero 
    #observer_time = "2010-04-03T10:54:00"
    observer_time = observer_time.isoformat()
    #L0 = 360*(carrington_rotation_number(observer_time) - int(carrington_rotation_number(observer_time))) 
    if carr_ck <0:
        carr = carr_ck + 360 #solo si carr_ck es negativo
    else:
        carr = carr_ck
    if carr >180:
        L0 = 360*(carrington_rotation_number(observer_time) - int(carrington_rotation_number(observer_time))-1)
    else:
        L0 = 360*(carrington_rotation_number(observer_time) - int(carrington_rotation_number(observer_time)))
    stony = carr + L0
    return stony

def stony_to_ckcarr(stony, observer_time):
    #observer_time = "2010-04-03T10:54:00"
    observer_time = observer_time.isoformat()
    L0 = 360*(carrington_rotation_number(observer_time) - int(carrington_rotation_number(observer_time))-1)
    carr = stony - L0
    if carr >360:
        carr = carr - 360
    if carr>180:
        carr_ck = carr - 360
    else:
        carr_ck = carr
    return carr_ck

def sensitivity_test(gcs_par_minimum, satpos, plotranges, masks, imsize, occ_size,aux):
    fig, axs = plt.subplots(4, 2, figsize=[12,8])
    axs = axs.ravel()
    #breakpoint()
    label=['CMElon', 'CMElat', 'CMEtilt', 'k', 'ang', 'height0', 'height1', 'height2']
    #for por instantes de tiempo
    minimum_error = np.sum(gcs_mask_error_diego(gcs_par_minimum, satpos, plotranges, masks, imsize, occ_size))
    for i in range(len(gcs_par_minimum)):
        gcs_par = gcs_par_minimum.copy()
        #entender plot ranges, ver en el caso del ploteo de params vs tiempo, ver el eje y.
        x = np.linspace(gcs_par[i]*0.3, gcs_par[i]*1.7, num=200)
        y=[]
        for j in x:
            gcs_par[i] = j
            #breakpoint()
            #y.append(np.mean(gcs_mask_error_diego(gcs_par, satpos, plotranges, masks, imsize, occ_size)))
            y.append(np.sum(gcs_mask_error_diego(gcs_par, satpos, plotranges, masks, imsize, occ_size)))
        axs[i].plot(x, y, 'o', color='r', label=label[i])
        axs[i].set_xlabel(label[i])
        axs[i].axvline(x=gcs_par_minimum[i], color='blue')
        axs[i].axhline(y=minimum_error, color='blue')
        axs[i].axhline(y=minimum_error*1.10, color='blue', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(opath, aux+'.png'))
    plt.close()
    #breakpoint()





############ Main
'''
Fits a filled masks created with GCS model to the data
'''
#Constants
dpath =  '/gehme/projects/2023_eeggl_validation/niemela_project/gcs_20100403_mask'
opath = dpath + '/gcs_fit'
#aux='' #if not using modified masks
aux = '/modified_masks'
dpath = dpath + aux
opath = opath + aux

#select=[4,5]#4,5,7,8,11,12] #None # select the time instants to fit, in order as read from dpath
select=None
manual_gcs_path = '/gehme/projects/2023_eeggl_validation/repo_diego/2020_gcs_with_ml/nn/neural_cme_seg/applications/niemela_proyect/'
imsize = [512, 512] # image size
gcs_par_range = [[-180,180],[-90,90],[-90,90],[4,20],[0.1,0.9], [1,80]] # bounds for the fit gcs parameters
occ_size = [50,75,90] # Artifitial occulter radius in pixels. Use 0 to avoid. [Stereo-A C2, Stereo-B C2, Lasco-C2]
Event_Number = 3
# Load data
meas_masks, fnames, satpos, plotranges, occ_sizes, masks_prop, dates= load_data(dpath, occ_size, select=select)
mask_total_px = [np.sum(m, axis=(1,2)) for m in meas_masks] # total number of positive pixels in the mask
csv_file = 'Event_list.csv'
df = pd.read_csv(manual_gcs_path+csv_file)

#loads manual gcs for IDL .sav file
#CMElon, CMElat, CMEtilt, height, k, ang

gcs_hebe_par = []
gcs_tony_par = []
time_hebe    = []
time_tony    = []
#Read df in triplets
# Read df in triplets
gcs_manual_par = []
df_event = df[df['Event_Number']==Event_Number]


for i in range(1, len(df_event), 3):
    print,i
    
    #quiero seleccionar la 2da linea de cada tripleta, que es donde ponemos la informacion.
    gcs_hebe_par.append([[float(str(df_event.iloc[i]['Long_H']).replace(",", "."))],
                        [float(str(df_event.iloc[i]['Lat_H']).replace(",", "."))],
                        [float(str(df_event.iloc[i]['Tilt_H']).replace(",", "."))],
                        [float(str(df_event.iloc[i]['Height_H']).replace(",", "."))],
                        [float(str(df_event.iloc[i]['Ratio_H']).replace(",", "."))],
                        [float(str(df_event.iloc[i]['Half_Angle_H']).replace(",", "."))] ] )
    time_hebe.append(datetime.strptime(df_event.iloc[i]['Date'],'%Y-%m-%dT%H-%M'))  
    gcs_tony_par.append([[float(str(df_event.iloc[i]['Long']).replace(",", "."))],
                        [float(str(df_event.iloc[i]['Lat']).replace(",", "."))],
                        [float(str(df_event.iloc[i]['Tilt']).replace(",", "."))],
                        [float(str(df_event.iloc[i]['Height']).replace(",", "."))],
                        [float(str(df_event.iloc[i]['Ratio']).replace(",", "."))],
                        [float(str(df_event.iloc[i]['Half_Angle']).replace(",", "."))] ] )
    time_tony.append(datetime.strptime(df_event.iloc[i]['Date'],'%Y-%m-%dT%H-%M')) 


# read /gehme/projects/2023_eeggl_validation/niemela_project/gcs_events
# guardar en gcs_manual
select_hebe = [1,3,5]#[1,3,5]
gcs_hebe_par = [gcs_hebe_par[index] for index in select_hebe]
time_hebe    = [time_hebe[index] for index in select_hebe]
# crate opath
os.makedirs(opath, exist_ok=True)

# bounds
up_bounds= np.array([gcs_par_range[0][1], gcs_par_range[1][1], gcs_par_range[2][1], gcs_par_range[4][1], gcs_par_range[5][1]])
#up_bounds= np.append(up_bounds, np.full(len(meas_masks), gcs_par_range[3][1]))
up_bounds= np.append(up_bounds, np.full(int(len(meas_masks)/2), gcs_par_range[3][1]))
low_bounds= np.array([gcs_par_range[0][0], gcs_par_range[1][0], gcs_par_range[2][0], gcs_par_range[4][0], gcs_par_range[5][0]])
#low_bounds= np.append(low_bounds, np.full(len(meas_masks), gcs_par_range[3][0]))
low_bounds= np.append(low_bounds, np.full(int(len(meas_masks)/2), gcs_par_range[3][0]))

#bound de abajo habian servido para 1 solo instante de tiempo.
#up_bounds = np.array([gcs_par_range[0][1], gcs_par_range[1][1], gcs_par_range[2][1], gcs_par_range[4][1], gcs_par_range[5][1], gcs_par_range[3][1]])
#low_bounds= np.array([gcs_par_range[0][0], gcs_par_range[1][0], gcs_par_range[2][0], gcs_par_range[4][0], gcs_par_range[5][0], gcs_par_range[3][0]])

#inital  conditions from masks_prop
# gcs_param_ini = gcs_manual_par[-2]
# ini_cond =  np.array([gcs_param_ini[0], gcs_param_ini[1], gcs_param_ini[2], gcs_param_ini[4], gcs_param_ini[5]])
# ini_cond = np.append(ini_cond, np.arange(len(meas_masks))+gcs_param_ini[3])
#breakpoint()
# CMElat from LASCO maks CPA
#ini_lat = np.median([masks_prop[i][2][1] for i in range(len(masks_prop))])
#print(ini_lat)
# change from 0 to 360 to -90 to 90
#if ini_lat > 90 and ini_lat < 180:
#    ini_lat = 180 - ini_lat 
#elif ini_lat > 180 and ini_lat < 270:
#    ini_lat = -(ini_lat - 180)
#elif ini_lat > 270 and ini_lat < 360:
#    ini_lat -= 360

# ang from the min mask AW
#ini_ang = np.min([masks_prop[i][j][2] for i in range(len(masks_prop)) for j in range(len(masks_prop[i]))])/2.
# heights from each LASCO mask height
#ini_height = [masks_prop[i][2][3] for i in range(len(masks_prop))]
# k at half the bounds
#ini_k = (gcs_par_range[4][0] + gcs_par_range[4][1])/2.
# CMElon from LASCO mask CPA
#ini_lon = np.median([masks_prop[2][0]])
#if ini_lon < 90 or ini_lon > 270:
#    ini_lon = 90
#else:
#    ini_lon = -90

#gcs height vs time, second order fit
altura  = [gcs_hebe_par[i][3][0] for i in range(len(gcs_hebe_par)) if np.isnan(gcs_hebe_par[i][3][0]) == False]
if len(altura)>1:
    tiempos = [time_hebe[i] for i in range(len(gcs_hebe_par)) if np.isnan(gcs_hebe_par[i][3][0]) == False]
    gcs_manual_par = [gcs_hebe_par[i] for i in range(len(gcs_hebe_par)) if np.isnan(gcs_hebe_par[i][3][0]) == False]
    tiempos_to_fit = np.array([(date - tiempos[0]).total_seconds() for date in tiempos])
    #breakpoint()
    coefs = np.polyfit(tiempos_to_fit,altura, 2)
    poly_func = np.poly1d(coefs)
    plot_gcs_height_fit = False
    if plot_gcs_height_fit:
        # Generate fitted values for plotting
        x_fit = np.linspace(tiempos_to_fit.min(), tiempos_to_fit.max(), 100)
        y_fit = poly_func(x_fit)
        plt.plot(tiempos_to_fit,altura,'o')
        plt.plot(x_fit,y_fit,'x')
        plt.savefig(os.path.join(opath, 'ajuste_altura2.png'))
        plt.close()
    #height from masks estimated using the function poly_func "gcs_height(time)" Como quiero una altura por cada instante de tiempo, 
    # me quedo unicamente con los valores impares
    ini_height = poly_func(np.array([( asd- tiempos[0]).total_seconds() for index, asd in enumerate(dates) if (index%2)==0]))
if len(altura)==1:
    tiempos = [time_hebe[i] for i in range(len(gcs_hebe_par)) if np.isnan(gcs_hebe_par[i][3][0]) == False]
    gcs_manual_par = [gcs_hebe_par[i] for i in range(len(gcs_hebe_par)) if np.isnan(gcs_hebe_par[i][3][0]) == False]
    ini_height = np.array(np.full(2,altura))

#estimating initial conditions from gcs_hebe_par
#breakpoint()
ini_lon    = np.nanmean([gcs_hebe_par[i][0] for i in range(len(gcs_hebe_par))])
#si csv usa stony, debo convertirlo a carrington de CK.
ini_lon = stony_to_ckcarr(ini_lon, tiempos[0])

ini_lat    = np.nanmean([gcs_hebe_par[i][1] for i in range(len(gcs_hebe_par))])
ini_tilt   = np.nanmean([gcs_hebe_par[i][2] for i in range(len(gcs_hebe_par))])
#ini_height = np.nanmean([gcs_hebe_par[i][3] for i in range(len(gcs_hebe_par))])
ini_k      = np.nanmean([gcs_hebe_par[i][4] for i in range(len(gcs_hebe_par))])
ini_ang    = np.nanmean([gcs_hebe_par[i][5] for i in range(len(gcs_hebe_par))])


#Pensar que el input ahora esta siendo el ajuste de Hebe y esta en stony, debe pasar a carrington y de ahi a ckcarr.
ini_cond_0 = np.array([ini_lon, ini_lat, ini_tilt, ini_k, ini_ang]+ini_height.tolist()).flatten()
#ini_cond_0 = ini_cond_0[:-1]
breakpoint()
print('Fitting GCS model with initial conditions: ', ini_cond_0)
#usar metodo lm
fit=least_squares(gcs_mask_error_diego, ini_cond_0 , method='trf', 
                kwargs={'satpos': satpos, 'plotranges': plotranges, 'masks': meas_masks, 'imsize': imsize, 'occ_size':occ_sizes}, 
                verbose=2, bounds=(low_bounds,up_bounds), diff_step=.5, xtol=1e-15) #, x_scale=scales)
ini_cond = fit.x
fit=least_squares(gcs_mask_error_diego, ini_cond , method='trf', 
                kwargs={'satpos': satpos, 'plotranges': plotranges, 'masks': meas_masks, 'imsize': imsize, 'occ_size':occ_sizes}, 
                verbose=2, bounds=(low_bounds,up_bounds), diff_step=.5, xtol=1e-15) #, x_scale=scales)
print('The fit parameters are: ', fit.x)
breakpoint()
# saves to pickle
with open(os.path.join(opath, 'gcs_fit.pkl'), 'wb') as f:
    pickle.dump(fit, f)

# plots manual and fit gcs param vs time
gcs_fit_par = []
gcs_fit_par_ckunits = []
j=0
for i in range(len(meas_masks)):
    #gcs_fit_par.append([fit.x[0], fit.x[1], fit.x[2], fit.x[5+i], fit.x[3], fit.x[4]])
    lon_stony = ckcarr_to_stony(fit.x[0],tiempos[j])
    gcs_fit_par.append([lon_stony, fit.x[1], fit.x[2], fit.x[5+j], fit.x[3], fit.x[4]]) 
    #gcs_fit_par_ckunits.append([fit.x[0], fit.x[1], fit.x[2], fit.x[5+j], fit.x[3], fit.x[4]]) 
    #breakpoint()
    if i%2==1:
        j=j+1
#lon in gcs_par_range should be in the range 0,360 because it is carrington coordinate system and not carrington_CK.
plot_gcs_param_vs_time(gcs_fit_par, gcs_manual_par, dates,tiempos,opath, ylim=gcs_par_range)

for i in range(len(gcs_manual_par)):
    gcs_manual_par[i][0] = stony_to_ckcarr(gcs_manual_par[i][0],tiempos[i])

# plots the fit mask along with the original masks
gcs_manual_par_plot = [item for item in gcs_manual_par for _ in range(2)]

i=0
for j in range(len(tiempos)):
    #gcs_param = [fit.x[0], fit.x[1], fit.x[2], fit.x[5+i], fit.x[3], fit.x[4]]
    gcs_param = [fit.x[0], fit.x[1], fit.x[2], fit.x[5+j], fit.x[3], fit.x[4]]
    mask_A = maskFromCloud_3d(gcs_param, satpos[i], imsize, plotranges[i], occ_size=occ_sizes[i])
    #gcs_param = [fit.x[0], fit.x[1], fit.x[2], fit.x[5+i+1], fit.x[3], fit.x[4]]
    mask_B = maskFromCloud_3d(gcs_param, satpos[i+1], imsize, plotranges[i+1], occ_size=occ_sizes[i+1])
    gcs_manual_par_plot_i = np.array(gcs_manual_par_plot[i]).flatten().tolist()
    mask_manual_A = maskFromCloud_3d(gcs_manual_par_plot_i, satpos[i], imsize, plotranges[i], occ_size=occ_sizes[i])
    mask_manual_B = maskFromCloud_3d(gcs_manual_par_plot_i, satpos[i+1], imsize, plotranges[i+1], occ_size=occ_sizes[i+1])
    cfname = fnames[i][0].split('_')[0] + '_' + fnames[i][0].split('_')[1]
    ofile = os.path.join(opath, f'{cfname}_gcs_fit.png')
    plot_to_png_diego(ofile, [fnames[i],fnames[i+1]], [meas_masks[i],meas_masks[i+1]], [mask_A,mask_B], [mask_manual_A,mask_manual_B])
    i=i+2


gcs_par_minimum = fit.x.copy() #gcs_fit_par_ckunits[0].copy()
aux='testeo_sensibilidad_mean'
aux='testeo_sensibilidad_sum'
aux2=['t0','t1','t2']

aux_file = aux


sensitivity_test(gcs_par_minimum, satpos, plotranges, meas_masks, imsize, occ_sizes,aux_file)







breakpoint()
i=0
for j in range(len(tiempos)):
    gcs_par_aux = gcs_par_minimum[:-2]
    gcs_par_aux[5] = gcs_par_minimum[5+int(i/2)]
    satpos_aux     = satpos[i:i+1]
    plotranges_aux = plotranges[i:i+1]
    meas_masks_aux = meas_masks[i:i+1]
    imsize_aux     = imsize
    occ_size_aux   = occ_sizes[i:i+1]
    aux_file = aux+aux2[j]
    sensitivity_test(gcs_par_aux, satpos_aux, plotranges_aux, meas_masks_aux, imsize_aux, occ_size_aux,aux_file)
    i=i+2
breakpoint()
