#!/usr/bin/env python
# coding: utf-8

# ## Librerias
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pyGCS_raytrace import pyGCS
from pyGCS_raytrace.rtraytracewcs import rtraytracewcs
from nn_training.get_cme_mask import get_cme_mask,get_mask_cloud
from nn_training.corona_background.get_corona import get_corona
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

def center_rSun_pixel(headers, plotranges, sat):
    '''
    Gets the location of Suncenter 
    '''    
    x_cS = (headers[sat]['CRPIX1']*plotranges[sat][sat]*2) / \
        headers[sat]['NAXIS1'] - plotranges[sat][sat]
    y_cS = (headers[sat]['CRPIX2']*plotranges[sat][sat]*2) / \
        headers[sat]['NAXIS2'] - plotranges[sat][sat]
    return x_cS, y_cS

def deg2px(x,y,plotranges,imsize):
    '''
    Computes spatial plate scale in both dimensions
    '''    
    scale_x = (plotranges[sat][1]-plotranges[sat][0])/imsize[0]
    scale_y =(plotranges[sat][3]-plotranges[sat][2])/imsize[1]
    x_px=[]
    y_px=[]    
    for i in range(len(x)):
        v_x= (np.round((x[i]-plotranges[sat][0])/scale_x)).astype("int") 
        v_y= (np.round((y[i]-plotranges[sat][2])/scale_y)).astype("int")
        if v_x<512 and v_y<512:
            x_px.append(v_x)
            y_px.append(v_y)
    return(y_px,x_px)

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

def pnt2arr(x,y,plotranges,imsize):
    '''
    Returns an array
    points:list of (x,y) points
    imsize: size of the output array
    '''
    p_x,p_y=deg2px(x,y,plotranges,imsize)
    points=[]
    for i in range(len(p_x)):
        points.append([p_x[i],p_y[i]])       
    arr=np.zeros(imsize)
    for i in range(len(points)):
        arr[points[i][0], points[i][1]] = 1
    return arr

######Main

# CONSTANTS
#files
DATA_PATH = '/gehme/data'
OPATH = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_training_v3' #'/gehme/projects/2020_gcs_with_ml/data/forwardGCS_test'
n_sat = 3 #number of satellites to  use [Cor A, Cor B, Lasco]

# GCS parameters [first 6]
# The other parameters are:
# level_cme: CME intensity level relative to the mean background corona
par_names = ['CMElon', 'CMElat', 'CMEtilt', 'height', 'k','ang', 'level_cme'] # par names
par_units = ['deg', 'deg', 'deg', 'Rsun','','deg',''] # par units
par_rng = [[-180,180],[-70,70],[-90,90],[8,30],[0.2,0.6], [10,60],[7e2,1e3]] # min-max ranges of each parameter in par_names
par_num = 2000  # total number of samples that will be generated for each param (there are nsat images per param combination)
rnd_par=True # set to randomnly shuffle the generated parameters linspace 

# Syntethic image options
imsize=np.array([512, 512], dtype='int32') # output image size
size_occ = [2.6, 3.7, 2] # Occulters size for [sat1, sat2 ,sat3] in [Rsun] 3.7
level_occ=0. #mean level of the occulter relative to the background level
cme_noise= [0,2.] #gaussian noise level of cme image. [mean, sd], both expressed in fractions of the cme-only image mean level. Set mean to None to avoid
occ_noise = [0,30.] # occulter gaussian noise. [mean, sd] both expressed in fractions of the abs mean background level. Set mean to None to avoid
mesh=False # set to also save a png with the GCSmesh (only for otype='png')
otype="png" # set the ouput file type: 'png' or 'fits'
im_range=2. # range of the color scale of the output final syntethyc image in std dev around the mean
back_rnd_rot=True # set to randomly rotate the background image around its center
inner_cme=False #Set to True to make the cme mask excludes the inner void of the gcs (if visible) 
mask_from_cloud=False #True to calculete mask from clouds, False to do it from ratraycing total brigthness image
two_cmes = True # set to include two cme per image on some (random) cases

# generate param arrays
par_num = [par_num] * len(par_rng)
all_par = []
for (rng, num) in zip(par_rng, par_num):
    cpar = np.linspace(rng[0],rng[1], num)
    if rnd_par:
        np.random.shuffle(cpar)
    all_par.append(cpar)

# Save configuration to .CSV
os.makedirs(OPATH, exist_ok=True)
date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_')
configfile_name = OPATH + '/' + date_str+'Set_Parameters.csv'
set = pd.DataFrame(np.column_stack(all_par), columns=par_names)
set.to_csv(configfile_name)
df = pd.DataFrame(pd.read_csv(configfile_name))
sat_info=[]
mask_prev = None
# generate views
for row in range(len(df)):
    for sat in range(n_sat):
        #back corona, temporal must be change to have a different corona per image#same_corona,hdr1
        sat_info.append(get_corona(sat,imsize=imsize))
    same_corona = [sat_info[0][0],sat_info[1][0]]
    headers = [sat_info[0][1],sat_info[1][1]]

    print(f'Saving image pair {row} of {len(df)-1}')
    for sat in range(n_sat):
        # Get the location of sats and gcs:
        satpos, plotranges = pyGCS.processHeaders(headers)
        
        # if row !=8 :
        #     break
        #defining ranges and radius of the occulter
        x = np.linspace(plotranges[sat][0], plotranges[sat][1], num=imsize[0])
        y = np.linspace(plotranges[sat][2], plotranges[sat][3], num=imsize[1])
        xx, yy = np.meshgrid(x, y)
        x_cS, y_cS = center_rSun_pixel(headers, plotranges, sat)  
        r = np.sqrt((xx - x_cS)**2 + (yy - y_cS)**2)

        if mask_from_cloud:
            #mask for cme outer envelope
            clouds = pyGCS.getGCS(df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], satpos)                
            x = clouds[sat, :, 1]
            y = clouds[0, :, 2]
            p_x,p_y=deg2px(x,y,plotranges,imsize)
            mask=get_mask_cloud(p_x,p_y,imsize,OPATH)
        else:
            btot_mask = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], imsize=imsize, occrad=size_occ[sat], in_sig=1., out_sig=0.1, nel=1e5)     
            mask = get_cme_mask(btot_mask,inner_cme=inner_cme)
            #check for too small mask
            cme_npix= len(btot_mask[btot_mask>0].flatten())
            mask_npix= len(mask[mask>0].flatten())
            if mask_npix/cme_npix<0.9:
                print(f'WARNING: CME number {row} mask is too small compared to cme brigthness image, skipping all views...')
                break
        
        #check for null mask
        mask[r <= size_occ[sat]] = 0  
        if len(np.array(np.where(mask==1)).flatten())/len(mask.flatten())<0.005: # only if there is a cme that covers more than 0.5% of the image
            print(f'WARNING: CME number {row} mask is null because it is probably behind the occulter, skipping all views...')
            break

        #Total intensity (Btot) figure from raytrace:               
        btot0 = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], imsize=imsize, occrad=size_occ[sat], in_sig=0.5, out_sig=0.25, nel=1e5)
                      
        #adds a flux rope-like structure
        height_diff = np.random.uniform(low=0.55, high=0.85)
        aspect_ratio_frope = np.random.uniform(low=0.09, high=0.14)
        int_frope = np.random.uniform(low=2, high=10, size=2) * np.random.choice([-1,1], size=2)
        btot1 = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row]*height_diff,aspect_ratio_frope, df['ang'][row], imsize=imsize, occrad=size_occ[sat], in_sig=0.7, out_sig=0.1, nel=1e5)
        btot11 = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row]*height_diff*1.2,aspect_ratio_frope, df['ang'][row], imsize=imsize, occrad=size_occ[sat], in_sig=0.7, out_sig=0.1, nel=1e5)
        btot = btot0 + int_frope[0]*btot1-int_frope[1]*btot11

        #mask for occulter
        arr = np.zeros(xx.shape)
        arr[r <= size_occ[sat]] = 1    

        #background corona
        back = same_corona[sat]
        if back_rnd_rot:
            back =  scipy.ndimage.rotate(back, np.random.randint(low=0, high=360), reshape=False)
        level_back = np.mean(back)               

        #cme
        #adds a random patchy spatial variation of the cme only 
        var_map = low_freq_map(dim=[imsize[0],imsize[1],1],off=[1],var=[1.5],func=[13])
        btot = var_map*(btot/np.max(btot))*df['level_cme'][row]*level_back
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
        #adds occulter
        if occ_noise[0] is not None:
            noise=np.random.normal(loc=occ_noise[0]*level_back, scale=occ_noise[1]*np.abs(level_back), size=imsize)[r <= size_occ[sat]]
        else:
            noise =0        
        btot[r <= size_occ[sat]] = level_occ*level_back + noise

        #creating folders for each case
        folder = os.path.join(OPATH, str(row*len(satpos)+sat))
        if os.path.exists(folder):
            os.system("rm -r " + folder) 
        os.makedirs(folder)
        mask_folder = os.path.join(folder, "mask")
        os.makedirs(mask_folder) 

        if otype=="fits":
            #mask for cme
            cme_mask = fits.PrimaryHDU(mask)
            cme_mask.writeto(mask_folder +'/2.fits'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), overwrite=True)
            if sceond_mask is not None:
                cme_mask = fits.PrimaryHDU(sceond_mask)
                cme_mask.writeto(mask_folder +'/3.fits'.format(
                    df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), overwrite=True)               
            #cme
            cme = fits.PrimaryHDU(btot)
            cme.writeto(folder +'/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_btot.fits'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), overwrite=True)
            #mask for occulter
            occ = fits.PrimaryHDU(arr)
            occ.writeto(mask_folder +'/1.fits'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1), overwrite=True)
            
        elif otype =="png":       
            #mask for cme
            ofile = mask_folder +'/2.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1)
            fig=save_png(mask,ofile=ofile, range=[0, 1])
            if sceond_mask is not None:      
                ofile = mask_folder +'/3.png'.format(
                    df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1)
                fig=save_png(sceond_mask,ofile=ofile, range=[0, 1])                      
            #mask for occulter
            ofile = mask_folder +'/1.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1)       
            fig=save_png(arr,ofile=ofile, range=[0, 1])
            #full image
            m = np.mean(btot[mask>0])
            sd = np.std(btot[mask>0])
            vmin=m-im_range*sd
            vmax=m+im_range*sd
            ofile=folder +'/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_btot.png'.format(
                  df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1)
            fig=save_png(btot,ofile=ofile, range=[vmin, vmax])
        else:
            print("otype value not recognized")    

        if mesh and otype=='png':
            # overplot  GCS mesh to cme figure
            clouds = pyGCS.getGCS(df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], satpos)                
            x = clouds[sat, :, 1]
            y = clouds[0, :, 2]         
            arr_cloud=pnt2arr(x,y,plotranges,imsize)
            ofile = folder +'/{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}_mesh.png'.format(
                df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1)
            fig=save_png(arr_cloud,ofile=ofile, range=[0, 1])     


