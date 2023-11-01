#!/usr/bin/env python
# coding: utf-8

# ## Librerias
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pyGCS_raytrace import pyGCS
from pyGCS_raytrace.rtraytracewcs import rtraytracewcs
from nn_training.get_cme_mask import get_cme_mask,get_mask_cloud
#from nn_training.corona_background.get_corona import get_corona
from nn_training.corona_background.get_corona_gcs_ml import get_corona
import numpy as np
import datetime
import matplotlib.pyplot as plt
from astropy.io import fits
import sunpy
#from sunpy.coordinates.ephemeris import get_horizons_coord
import sunpy.map
import pandas as pd
#from sunpy.sun.constants import radius as _RSUN
#from ext_libs.rebin import rebin
from nn_training.low_freq_map import low_freq_map
import scipy
from nn.utils.coord_transformation import deg2px, center_rSun_pixel, pnt2arr


def get_params(par_names, par_rng): #par_names = ['CMElon', 'CMElat', 'CMEtilt', 'height', 'k','ang', 'level_cme'] # par names
                         #            [[-180,180],[-70,70],[-90,90],[5,10],[0.2,0.6],[10,60],[7e2,1e3]]
    params = []
    for i in range(len(par_names[0:-2])):
        params.append(np.random.uniform(low=par_rng[i][0], high=par_rng[i][1]))
    if params in used_params:
        params = get_params(par_names, par_rng)
    return params

def save_png(array, ofile=None, range=None):
    '''
    pltos array to an image in ofile without borders axes or anything else
    ofile: if not give only the image object is generated and returned
    range: defines color scale limits [vmin,vmax]
    '''    
    fig = plt.figure(figsize=(4,4), facecolor='white')
    if range is not None:
        vmin=range[0]
        vmax=range[1]
    else:
        vmin=None
        vmax=None
    plt.imshow(array, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)#, aspect='auto')#,extent=plotranges[sat])
    plt.axis('off')         
    if ofile is not None:
        fig.savefig(ofile, facecolor='white', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return 1
    else:
        return fig
    
def save_cases(succesful_cases, otype, iter_counter):
    folder = os.path.join(OPATH, str(iter_counter))
    mask_folder = os.path.join(folder, "mask")
    os.makedirs(folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    for i in range(len(succesful_cases)):
        if otype == "fits":
            cme_mask = succesful_cases[i][0]
            cme_mask.writeto(mask_folder +'/sat{}.fits'.format(i+1), overwrite=True)
            if synth_int_image:
                cme = succesful_cases[i][1]
                cme.writeto(folder +'/sat{}_btot.fits'.format(i+1), overwrite=True)
            occ = succesful_cases[i][2]
            occ.writeto(mask_folder +'/sat{}_occ.fits'.format(i+1), overwrite=True)
        elif not mesh and otype == "png":
            # mask for cme
            cme_mask = succesful_cases[i][0]
            ofile = mask_folder +'/sat{}.png'.format(i+1)
            fig=save_png(cme_mask,ofile=ofile, range=[0, 1])
            # mask for occulter
            arr = succesful_cases[i][1]
            ofile = mask_folder +'/sat{}_occ.png'.format(i+1)
            fig=save_png(arr,ofile=ofile, range=[0, 1])
            if synth_int_image:
                btot = succesful_cases[i][2]
                m = np.mean(btot)
                sd = np.std(btot)
                vmin=m-im_range*sd
                vmax=m+im_range*sd
                ofile=folder +'/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_btot.png'.format(
                    params[0], params[1], params[2], params[3], params[4], params[5], i+1)
                fig=save_png(btot,ofile=ofile, range=[vmin, vmax])
        elif mesh and otype == "png":
            mask = succesful_cases[i][0]
            ofile = mask_folder +'/sat{}.png'.format(i+1)
            fig=save_png(mask,ofile=ofile, range=[0, 1])
            arr_cloud = succesful_cases[i][1]
            ofile = folder +'/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_mesh.png'.format(
                params[0], params[1], params[2], params[3], params[4], params[5], i+1)
            fig=save_png(arr_cloud,ofile=ofile, range=[0, 1])

def checkFailure(failure_counter, n_sat, min_nviews):
    if failure_counter > n_sat - min_nviews:
        print(f'WARNING: Too many failures, skipping all views...')
        return True

def pygcsWrapper(params, satpos):
    return pyGCS.getGCS(params[0], params[1], params[2], params[3], params[4], params[5], satpos)

def raytracewcsWrapper(header, params, imsize, occrad, in_sig, out_sig, nel):
    return rtraytracewcs(header, params[0], params[1], params[2], params[3], params[4], params[5], imsize=imsize, occrad=occrad, in_sig=in_sig, out_sig=out_sig, nel=nel)


######Main

# CONSTANTS
#files
DATA_PATH = '/gehme/data'
OPATH = '/gehme-gpu/projects/2020_gcs_with_ml/data/gcs_ml_3VP' #'/gehme/projects/2020_gcs_with_ml/data/cme_seg_training'
OVERWRITE = True # set to True to overwrite existing files
n_sat = 3 #number of satellites to  use [Cor2 A, Cor2 B, Lasco C2]
min_nviews = 3 # minimum number succesful views 

# GCS parameters [first 6]
# The other parameters are:
# level_cme: CME intensity level relative to the mean background corona
par_names = ['CMElon', 'CMElat', 'CMEtilt', 'height', 'k','ang', 'level_cme', 'satpos', 'plotranges'] # par names
par_units = ['deg', 'deg', 'deg', 'Rsun','','deg',''] # par units
par_rng = [[-180,180],[-70,70],[-90,90],[3,10],[0.2,0.6],[10,60],[7e2,1e3]] # min-max ranges of each parameter in par_names {2.5,7} heights
par_num = int(1e5)  # total number of samples that will be generated for each param (there are nsat images per param combination)
rnd_par=False # when true it apllies a fixed seed to generate the same parameters for each run
seed = 47 # seed to use when rnd_par is True
same_corona=True # Set to True use a single corona back for all par_num cases
same_position=True # Set to True to use the same set of satteite positions(not necesarly the same background image)

# Syntethic image options
synth_int_image=False # set to True to generate a synthetic image with a cme
plot_mask_rtc_only=False # set to True to plot only the mask from raytracing, synth_int_image must be True
imsize=np.array([512, 512], dtype='int32') # output image size
level_occ=0. #mean level of the occulter relative to the background level
cme_noise= [0,2.] #gaussian noise level of cme image. [mean, sd], both expressed in fractions of the cme-only image mean level. Set mean to None to avoid
occ_noise = [0,30.] # occulter gaussian noise. [mean, sd] both expressed in fractions of the abs mean background level. Set mean to None to avoid
mesh=True # set to also save a png with the GCSmesh (only for otype='png')
otype="png" # set the ouput file type: 'png' or 'fits'
im_range=2. # range of the color scale of the output final syntethyc image in std dev around the mean
back_rnd_rot=False # set to randomly rotate the background image around its center
inner_hole_mask=True #Set to True to make the cme mask excludes the inner void of the gcs (if visible) 
mask_from_cloud=False #True to calculete mask from clouds, False to do it from ratraycing total brigthness image
two_cmes = False # set to include two cme per image on some (random) cases


# initialize vars
used_params = []
iter_counter = -1
if not rnd_par:
    np.random.seed(seed)

OPATH = OPATH + "_size_" + str(par_num) + "_seed_" + str(seed)
if os.path.exists(OPATH) and OVERWRITE:
    os.system("rm -r " + OPATH)
os.makedirs(OPATH, exist_ok=True)

# Save configuration to .CSV
date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_')
configfile_name = OPATH + '/' + date_str+'Set_Parameters.csv'
succesful_df = pd.DataFrame(columns=par_names)

# defining lists
satpos_all = []
plotranges_all = []

# generate views
back_corona=[]
headers=[]
size_occ=[]
while iter_counter != par_num:

    # while vars
    successful_cases = []
    success_counter = 0
    failure_counter = 0
    too_many_failures = False

    # get parameters
    params = get_params(par_names, par_rng)

    #get background corona,headers and occulter size
    if same_corona==False or len(back_corona)==0:
        for sat in range(n_sat):
            if sat == 0:
                a,b,c,obs_datetime=get_corona(sat,imsize=imsize, custom_headers=same_position)
            else:
                a,b,c,_=get_corona(sat,imsize=imsize, obs_datetime=obs_datetime, custom_headers=same_position)
            back_corona.append(a)
            headers.append(b)
            size_occ.append(c)
    
    # Get the location of sats and gcs:
    satpos, plotranges = pyGCS.processHeaders(headers)


    if not mask_from_cloud:
        # Computes with mask from cloud just to see if there is any failure
        for sat in range(n_sat):
            if too_many_failures:
                break
            #defining ranges and radius of the occulter
            x = np.linspace(plotranges[sat][0], plotranges[sat][1], num=imsize[0])
            y = np.linspace(plotranges[sat][2], plotranges[sat][3], num=imsize[1])
            xx, yy = np.meshgrid(x, y)
            x_cS, y_cS = center_rSun_pixel(headers, plotranges, sat)  
            r = np.sqrt((xx - x_cS)**2 + (yy - y_cS)**2)
            #mask for cme outer envelope
            clouds = pygcsWrapper(params, satpos)             
            x = clouds[sat, :, 1]
            y = clouds[0, :, 2]
            p_x,p_y=deg2px(x,y,plotranges,imsize, sat)
            mask=get_mask_cloud(p_x,p_y,imsize)
            #check for null mask
            mask[r <= size_occ[sat]] = 0

            if len(np.array(np.where(mask==1)).flatten())/len(mask.flatten())<0.005: # only if there is a cme that covers more than 0.5% of the image
                print(f'WARNING: CME mask is null because it is probably behind the occulter')
                failure_counter += 1
                too_many_failures = checkFailure(failure_counter, n_sat, min_nviews)
                break
            

    for sat in range(n_sat):
        print(f'Generating image {sat+1} of {n_sat} for iteration {iter_counter+1} of {par_num}')
        if too_many_failures:
            break
        #defining ranges and radius of the occulter
        x = np.linspace(plotranges[sat][0], plotranges[sat][1], num=imsize[0])
        y = np.linspace(plotranges[sat][2], plotranges[sat][3], num=imsize[1])
        xx, yy = np.meshgrid(x, y)
        x_cS, y_cS = center_rSun_pixel(headers, plotranges, sat)  
        r = np.sqrt((xx - x_cS)**2 + (yy - y_cS)**2)

        if mask_from_cloud:
            #mask for cme outer envelope
            clouds = pygcsWrapper(params, satpos)             
            x = clouds[sat, :, 1]
            y = clouds[0, :, 2]
            p_x,p_y=deg2px(x,y,plotranges,imsize, sat)
            mask=get_mask_cloud(p_x,p_y,imsize)
        else:
            btot_mask = raytracewcsWrapper(headers[sat], params, imsize=imsize, occrad=size_occ[sat], in_sig=1., out_sig=0.1, nel=1e5)     
            cme_npix= len(btot_mask[btot_mask>0].flatten())
            if cme_npix<=0:
                print(f'WARNING: CME raytracing did not work')
                failure_counter += 1
                too_many_failures = checkFailure(failure_counter, n_sat, min_nviews)
                breakpoint()
                break
            mask = get_cme_mask(btot_mask,inner_cme=inner_hole_mask)          
            mask_npix= len(mask[mask>0].flatten())
            if mask_npix/cme_npix<0.8:
                print(f'WARNING: CME mask is too small compared to cme brigthness image')
                failure_counter += 1
                too_many_failures = checkFailure(failure_counter, n_sat, min_nviews)
                break
        
        #check for null mask
        mask[r <= size_occ[sat]] = 0  
        if len(np.array(np.where(mask==1)).flatten())/len(mask.flatten())<0.005: # only if there is a cme that covers more than 0.5% of the image
            print(f'WARNING: CME mask is null because it is probably behind the occulter')
            failure_counter += 1
            too_many_failures = checkFailure(failure_counter, n_sat, min_nviews)
            break

        #mask for occulter
        arr = np.zeros(xx.shape)
        arr[r <= size_occ[sat]] = 1

        #Total intensity (Btot) figure from raytrace:
        if synth_int_image:  
            btot0 = raytracewcsWrapper(headers[sat], params, imsize=imsize, occrad=size_occ[sat], in_sig=0.5, out_sig=0.25, nel=1e5)
                      
            #adds a flux rope-like structure
            height_diff = np.random.uniform(low=0.55, high=0.85)
            aspect_ratio_frope = np.random.uniform(low=0.09, high=0.14)
            int_frope = np.random.uniform(low=2, high=10, size=2) * np.random.choice([-1,1], size=2)
            btot1 = rtraytracewcs(headers[sat], params[0], params[1], params[2], params[3]*height_diff,aspect_ratio_frope, params[4], imsize=imsize, occrad=size_occ[sat], in_sig=0.7, out_sig=0.1, nel=1e5)
            btot11 = rtraytracewcs(headers[sat], params[0], params[1], params[2], params[3]*height_diff*1.2,aspect_ratio_frope, params[4], imsize=imsize, occrad=size_occ[sat], in_sig=0.7, out_sig=0.1, nel=1e5)
            btot = btot0 + int_frope[0]*btot1-int_frope[1]*btot11

            #background corona
            back = back_corona[sat]
            if back_rnd_rot:
                back =  scipy.ndimage.rotate(back, np.random.randint(low=0, high=360), reshape=False)
            level_back = np.mean(back)

            #cme
            #adds a random patchy spatial variation of the cme only 
            var_map = low_freq_map(dim=[imsize[0],imsize[1],1],off=[1],var=[1.5],func=[13])
            btot = var_map*(btot/np.max(btot))*params[6]*level_back
            #adds noise
            if cme_noise[0] is not None :
                m = np.mean(btot)
                noise=np.random.normal(loc=cme_noise[0]*m, scale=cme_noise[1]*np.abs(m), size=imsize)
                btot[btot > 0]+=noise[btot > 0]    
            #Randomly adds the previous CME to have two in one image
            sceond_mask = None
            if two_cmes and np.random.choice([True,False,False]): # only for sat 0 for now
                if mask_prev is None:
                    btot_prev = btot
                    mask_prev = mask
                else:
                    cbtot = np.array(btot)
                    btot += btot_prev
                    btot_prev = cbtot
                    cbtot = 0
                    sceond_mask = np.array(mask_prev)
                    mask_prev = mask

            #adds background
            btot+=back
            if plot_mask_rtc_only:
                btot = np.array(btot_mask)
            #adds occulter
            if occ_noise[0] is not None:
                noise=np.random.normal(loc=occ_noise[0]*level_back, scale=occ_noise[1]*np.abs(level_back), size=imsize)[r <= size_occ[sat]]
            else:
                noise =0        
            btot[r <= size_occ[sat]] = level_occ*level_back + noise
        
        case_stuff =[]
        if otype=="fits":
            cme_mask = fits.PrimaryHDU(mask)
            case_stuff.append(cme_mask)
            if synth_int_image:
                cme = fits.PrimaryHDU(btot)
                case_stuff.append(cme)
            occ = fits.PrimaryHDU(arr)
            case_stuff.append(occ)
        elif not mesh and otype == "png":
            case_stuff.append(mask)
            case_stuff.append(arr)
            if synth_int_image:
                case_stuff.append(btot)
        elif mesh and otype=='png':
            clouds = pygcsWrapper(params, satpos)                
            x = clouds[sat, :, 1]
            y = clouds[0, :, 2]    
            arr_cloud=pnt2arr(x,y,plotranges,imsize, sat)
            case_stuff.append(mask)
            case_stuff.append(arr_cloud)
        
        successful_cases.append(case_stuff)
        success_counter += 1

    if success_counter >= min_nviews:

        iter_counter += 1

        # save cases
        save_cases(successful_cases, otype, iter_counter)

        used_params.append(params)
        params.append(satpos)
        params.append(plotranges)
        succesful_df.loc[iter_counter] = params

#save to csv
succesful_df.to_csv(configfile_name)

