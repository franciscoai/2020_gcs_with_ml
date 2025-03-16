#!/usr/bin/env python
# coding: utf-8

# ## Librerias
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pyGCS_raytrace import pyGCS
from pyGCS_raytrace.rtraytracewcs import rtraytracewcs
from nn_training.get_cme_mask import get_mask_cloud,get_cme_mask
from nn_training.corona_background.get_corona import get_corona
import numpy as np
import datetime
import matplotlib.pyplot as plt
from astropy.io import fits
import sunpy.map
import pandas as pd
from nn_training.low_freq_map import low_freq_map
import scipy
from nn.utils.coord_transformation import deg2px, center_rSun_pixel, pnt2arr
import concurrent.futures
from scipy.stats import kstest,chisquare,ks_2samp
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
mpl.use('Agg')

def save_png(array, ofile=None, range=None):
    '''
    pltos array to an image in ofile without borders axes or anything else
    ofile: if not give only the image object is generated and returned
    range: defines color scale limits [vmin,vmax]
    '''    
    if range is not None:
        vmin=range[0]
        vmax=range[1]
    else:
        vmin=None
        vmax=None    
    if ofile is not None:
        plt.imsave(ofile, array, cmap="gray", vmin=vmin, vmax=vmax)
        return 1
    else:
        return None


def get_random_lognormal(size):
    '''
    Generates a random number of cme desplacement in Rsun based on a lognormal distribution for cme velocity
    '''
    mu, sigma = np.log(433), 0.5267# mean and standard deviation of CME velocity from table 1 Yurchyshyn 2005
    while True:
        cme_vel = np.random.lognormal(mu, sigma, 1) # in km/s
        #Cota minima y maxima en distribucion de velocidades segun Vourlidas et al. 2010 
        #200 y 1800 km/s
        if cme_vel[0] > 200 and cme_vel[0] < 1800:        
            cme_vel = cme_vel/695700 # in Rsun/s
            cadencia = 15 # in min
            cadencia = cadencia*60 # in s
            height_diff = cme_vel[0] * cadencia # displacement in Rsun, assuming 15 min cadence.
            return height_diff


def flter_non_zero(back,btot,mask):
    '''
    Filters out zero values in both back and btot ndarrays. Returns the filtered arrays.
    '''
    A=back[mask==1].copy().flatten()
    B=btot[mask==1].copy().flatten()
    if np.any(A==0) or np.any(B==0):
        non_zero = (A != 0) & (B != 0)
        A = A[non_zero]
        B = B[non_zero]
    area = np.sum(mask)
    return A,B,area

def statistic_values(A,B=None):
    '''
    Calculates mean, median and std of A and/or B.
    factor size: amount of std consider to calculate the amount of values outside that range.

    output: mean_A,median_A,std_A [,mean_B,median_B,std_B]
    '''
    mean_A   = np.mean(A)
    median_A = np.median(A)
    std_A    = np.std(A)
    if B is not None:
        mean_B   = np.mean(B)
        std_B    = np.std(B)
        median_B = np.median(B)
        return mean_A,median_A,std_A,mean_B,median_B,std_B
    else:
        return mean_A,median_A,std_A

def area_outside_std(A,factor_std=2,positive_only=False,percentile=False,cota=False):
    '''
    Calculates the amount of values outside a certain range in terms of std.
    factor size: amount of std consider to calculate the amount of values outside that range.
    '''
    median_A = np.median(A)
    std_A  = np.std(A)

    if cota and not percentile:
        if positive_only:
            return len(A[A > cota[1]])
        else:
            return len(A[(A < cota[0]) | (A > cota[1])])

    if percentile and not cota:
        percentiles = np.percentile(A, [percentile[0],percentile[1]])
        if positive_only:
            return len(A[A > percentiles[1]])
        else:
            return len(A[(A < percentiles[0]) | (A > percentiles[1])])
    
    if not percentile and not cota:
        if positive_only:
            return len(A[A > median_A + factor_std*std_A])
        else:
            return len(A[(A < median_A - factor_std*std_A) | (A > median_A + factor_std*std_A)])

def area_outside_thresh(A,cota=False):
    '''
    Calculates the amount of values outside a certain value
    cota should be a list.
    '''
    results = []
    for j in cota:
        results.append(len(A[A > j]))

def factor(X, Y):
    """
    Ment to estimate the scale factor. We follow Howard et al. 2018. Density in the front of CME follows a power law with a factor of -3.
    Therefore, Ne(r_0) = N0 * r_0**-3 and  Ne(r_0-desplazamiento)= N0 * (r_0-desplazamiento)**-3.
    Therefore Ne(r_0-desplazamiento) / Ne(r_0) = (1-desplazamiento/r_0)**-3
    X: desplacement in Rsun
    Y: Height in Rsun
    """
    return (1-X/Y)**-3

def stats_caclulations(variable, filter, percentil=[15,85]):
    '''
    variable: Btot Background or CME
    filter: mask CME, mask leading edge.
    '''
    filtered_variable = variable[filter.astype(np.int64)]
    selection = ~np.isnan(filtered_variable)
    mean = np.mean(filtered_variable[selection])
    median = np.median(filtered_variable[selection])
    std = np.std(filtered_variable[selection])
    percentil_1 = np.percentile(filtered_variable[selection], percentil[0])
    percentil_2 = np.percentile(filtered_variable[selection], percentil[1])
    return [mean, median, std, percentil_1, percentil_2]


def plot_histogram(back,btot,mask,filename,plot_ratio_histo=False,plot_ratio_histo2=True,selected_cota=9):
    '''
    Plots histograms of back and btot inside the mask
    '''
    if plot_ratio_histo:
        A,B,area=flter_non_zero(back,btot,mask)
        #Ratio of the histograms
        num_bins = 80  
        ax = plt.gca()
        min_val = np.percentile(B/A,2) 
        max_val = np.percentile(B/A,98)
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        plt.hist(B/A, bins=bin_edges, alpha=0.5, label='Btot/Back', color='red')
        mean_BA,median_BA,std_BA = statistic_values(B/A)
        area_outside_double   = area_outside_std(B/A, percentile=[15,85])
        area_outside_positive = area_outside_std(B/A, percentile=[15,85], positive_only=True)
        percentile_superior = np.percentile(B/A,85)
        percentile_inferior = np.percentile(B/A,15)
        cota = [-1,1]
        cota = [x * selected_cota +1 for x in cota]
        area_cota_7   = area_outside_std(B/A, cota=[0,7], positive_only=True)
        area_cota_positive = area_outside_std(B/A, cota=cota, positive_only=True)
        breakpoint()
        plt.text(0.03, 0.95 ,f'\nMean: {mean_BA:.3f}\nStd: {std_BA:.3f}\nMedian: {median_BA:.3f}',transform=ax.transAxes,fontsize=10,color='blue',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
        plt.text(0.03, 0.75 ,f'\nper 85: {percentile_superior:.1f}\nper 15: {percentile_inferior:.1f}',transform=ax.transAxes,fontsize=10,color='green',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
        plt.text(0.70, 0.75 ,f'\nArea Tot: {area:.1f}',transform=ax.transAxes,fontsize=10,color='red',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
        plt.text(0.70, 0.68 ,f'\n cota 7+/Area: {area_cota_7/area:.3f}',transform=ax.transAxes,fontsize=10,color='blue',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
        plt.text(0.70, 0.64 ,f'\n cota10+ /Area: {area_cota_positive/area:.3f}',transform=ax.transAxes,fontsize=10,color='blue',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
        plt.axvline(x=1, color='red', linestyle='-', linewidth=2, label='x = 1')
        plt.axvline(x=percentile_superior, color='green', linestyle='--', linewidth=2, label='percentiles')
        plt.axvline(x=percentile_inferior, color='green', linestyle='--', linewidth=2)
        plt.axvline(x=cota[0], color='blue', linestyle='--', linewidth=2, label=f'cotas{cota[0],cota[1]}')
        plt.axvline(x=cota[1], color='blue', linestyle='--', linewidth=2)
        plt.legend(loc='upper right')
        plt.savefig(filename.replace('histo.png','histo_ratio.png'))
        plt.close()


    if plot_ratio_histo2: 
        back_on_mask,btot_on_mask,area=flter_non_zero(back,btot,mask)
        SNR = btot_on_mask / back_on_mask - 1
        mean_BA,median_BA,std_BA = statistic_values(SNR)
                
        fig, ax = plt.subplots(figsize=(8, 6))
        num_bins = 80  
        ax = plt.gca()
        min_val = np.percentile(SNR,2) 
        max_val = np.percentile(SNR,99)
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        n, bins, _ = ax.hist(SNR, bins=bin_edges, facecolor='red', edgecolor='red', alpha=0.5)
        ax.set_xlabel('SNR')
        ax.set_ylabel('Frequency')
        ax.set_xlim(min_val, max_val)

        ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper right', borderpad=1)
        ax_inset.hist(SNR, bins=bin_edges, facecolor='red', edgecolor='red', alpha=0.5)
        cota_superior = np.percentile(SNR,99)
        cota = [1,3,5,7,10]
        ax_inset.set_xlim(1, cota_superior)
        _max_ylim = n[ (bins[:-1] >= 1) & (bins[:-1] <= cota_superior) ]
        if _max_ylim.size == 0:
            _max_ylim = 10
        max_ylim = np.max(_max_ylim)
        ax_inset.set_ylim(0, max_ylim * 1.1 )
        ax.axvline(x=0, color='red', linestyle='-', linewidth=2, label='x = 1')
        percentile_superior = np.percentile(SNR,85)
        percentile_inferior = np.percentile(SNR,15)
        ax.axvline(x=percentile_superior, color='green', linestyle='--', linewidth=2)
        ax.axvline(x=percentile_inferior, color='green', linestyle='--', linewidth=2)
        for j in cota:
            ax.axvline(x=j, color='blue', linestyle='--', linewidth=2)
        #ax.axvline(x=cota[0], color='blue', linestyle='--', linewidth=2)
        #ax.axvline(x=cota[1], color='blue', linestyle='--', linewidth=2)
        area_threshold = area_outside_thresh(SNR, cota=[1,3,5,7,10])
        ax.text(0.03, 0.95 ,f'\nMean: {mean_BA:.3f}\nStd: {std_BA:.3f}\nMedian: {median_BA:.3f}',transform=ax.transAxes,fontsize=10,color='blue',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
        ax.text(0.03, 0.75 ,f'\nper 85: {percentile_superior:.1f}\nper 15: {percentile_inferior:.1f}',transform=ax.transAxes,fontsize=10,color='green',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
        ax.text(0.70, 0.65 ,f'\nArea Tot: {area:.1f}',transform=ax.transAxes,fontsize=10,color='red',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
        ax.text(0.70, 0.60 ,f'\n ratio1 + /Area: {area_threshold[0]/area:.3f}',transform=ax.transAxes,fontsize=10,color='blue',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
        ax.text(0.70, 0.55 ,f'\n ratio3 + /Area: {area_threshold[1]/area:.3f}',transform=ax.transAxes,fontsize=10,color='blue',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
        ax.text(0.70, 0.50 ,f'\n ratio5 + /Area: {area_threshold[2]/area:.3f}',transform=ax.transAxes,fontsize=10,color='blue',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
        ax.text(0.70, 0.45 ,f'\n ratio7 + /Area: {area_threshold[3]/area:.3f}',transform=ax.transAxes,fontsize=10,color='blue',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
        ax.text(0.70, 0.40 ,f'\n ratio10 + /Area: {area_threshold[4]/area:.3f}',transform=ax.transAxes,fontsize=10,color='blue',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
        plt.savefig(filename.replace('histo.png','histo_ratio.png'))
        plt.close()


######Main

# CONSTANTS
#paths
DATA_PATH = '/gehme/data'
#OPATH = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_20240912'
OPATH = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_20250212'
OPATH = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_20250304'
opath_fstructure='check'#'run' # use 'check' to save all the ouput images in the same dir together for easier checkout
                         # use 'run' to save each image in a different folder as required for training dataset
#Syntethic image options
# morphology
diff_int_cme=True # set to use a differential intensity CME image
add_flux_rope = True # set to add a flux rope-like structure to the cme image
par_names = ['CMElon', 'CMElat', 'CMEtilt', 'height', 'k','ang', 'level_cme'] # GCS parameters plus CME intensity level
par_units = ['deg', 'deg', 'deg', 'Rsun','','deg','frac of back sdev'] # par units
par_rng = [[-180,180],[-70,70],[-90,90],[1.5,20],[0.2,0.6], [5,65],[2,7]] 
par_num = 50#300000  # total number of GCS samples that will be generated. n_sat images are generated per GCS sample.
rnd_par=True # set to randomnly shuffle the generated parameters linspace 
#background
n_sat = 3 #number of satellites to  use [Cor2 A, Cor2 B, Lasco C2]
back_rnd_rot=False # set to randomly rotate the background image around its center
same_corona=False # Set to True use a single corona back for all par_num cases
same_position=True # Set to True to use the same set of satelite positions(not necesarly the same background image)
rnd_int_occ=0.1 # set to randomly increase the internal occulter size by a max fraction of its size. Use None to keep the constant size given by get_corona
rnd_ext_occ=0.1 # set to randomly reduce the external occulter size by a max fraction of its size. Use None to keep the constant size given by get_corona
#noise
cme_noise= True # set to add poissonian noise to the cme image
#relative int levels
level_occ='min' # level of the occulter relative to the background level. Note that png images are saved in 0-255 scale
                # The image range is mapped to 0-255 using the value given by im_range. 
                # Use 'min' to set the occulter to the minimum value, thus appearing as 0 in png images
im_range=3 # range of the color scale of the output final syntethyc image in std dev around the mean

# Output images to save
otype="png" # set the ouput file type: 'png' or 'fits'
imsize=np.array([512, 512], dtype='int32') # output image size
add_mesh_image=False # adds the GCS mesh on top of the image (only for otype='png')
cme_only_image=False # set to True to save an addditional image with only the cme without the background corona
back_only_image = False # set to True to save an addditional image with only the background corona without the cme
save_masks = True # set to True to save the masks images
save_background = True#True # set to True to save the background images
save_leading_edge = True # set to True to save the leading edge images
save_only_cme_mask = False # set to True to save only the cme mask and not the occulter masks. save_masks must be True
inner_hole_mask=False #Set to True to produce a mask that contains the inner hole of the GCS (if visible)
mask_from_cloud=False #True #True to calculete mask from clouds, False to do it from an independent ratraycing total brigthness image
two_cmes = False # set to include two cme per image on some (random) cases
show_middle_cross = False #False # set to show the middle cross of the image

#### main
if opath_fstructure=='check':
    save_masks = True #False
    cme_only_image = False
    back_only_image = False
    add_mesh_image = True
    rnd_par=True#False 
    save_only_cme_mask = True
    diff_int_cme = True #False
    par_rng = [[-180,180],[-70,70],[-90,90],[1.5,20],[0.2,0.6], [5,65],[2,7]] 
    #par_rng = [[-0,0.1],[0,0.1],[90,90],[19,20],[0.15,0.15], [70,71],[10,10]] 
    inner_hole_mask= False
    show_middle_cross = True
    add_flux_rope = True

os.makedirs(OPATH, exist_ok=True)
# saves a copy of this script to the output folder
os.system("cp " + os.path.realpath(__file__) + " " + OPATH)
# generate param arrays
par_num = [par_num] * len(par_rng)
all_par = []
for (rng, num) in zip(par_rng, par_num):
    cpar = np.linspace(rng[0],rng[1], num)
    if rnd_par:
        np.random.shuffle(cpar)
    all_par.append(cpar)

# Save param to .CSV
date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_')
configfile_name = OPATH + '/' + date_str+'Set_Parameters.csv'
set = pd.DataFrame(np.column_stack(all_par), columns=par_names)
set.to_csv(configfile_name)
df = pd.DataFrame(pd.read_csv(configfile_name))

mask_prev = None
satpos_all = []
plotranges_all = []
sceond_mask = None
mask_aw_all = []
apex_all = []
scl_fac_fr_all = []
def_fac_all = []
exp_fac_all = []
aspect_ratio_frope_all = []
status_all = []
halo_count = 0
ok_cases = 0
usr_center_off = None
median_btot_over_back_all = []
filter_area_threshold_all = []
sinthetic_params_Bt_out_all = []
sinthetic_params_Bt_RD_all = []
sinthetic_params_FR_out_all = []
sinthetic_params_FR_RD_all = []
stats_btot_mask_all = []
stats_back_mask_all = []
stats_cme_mask_all = []
stats_btot_mask_outer_all = []
stats_back_mask_outer_all = []
stats_cme_mask_outer_all = []
folder_name_all = []

def create_sintetic_image(row):
    global ok_cases, halo_count,satpos_all, plotranges_all, mask_aw,sceond_mask
    #get background corona,headers and occulter size
    if same_corona==False or row==0:
        back_corona=[]
        headers=[]
        size_occ=[]
        size_occ_ext=[]
        occ_center = []
        for sat in range(n_sat):
            a,b,c,d,e=get_corona(sat, imsize=imsize, custom_headers=same_position)
            back_corona.append(a)
            headers.append(b)
            if rnd_int_occ is not None:
                c=c*np.random.uniform(low=1., high=1.+rnd_int_occ)
            if rnd_ext_occ is not None:
                d=d*np.random.uniform(low=1.-rnd_ext_occ, high=1.)
            size_occ.append(c)
            size_occ_ext.append(d)
            occ_center.append(e)
    # Get the location of sats and gcs: 
    satpos, plotranges = pyGCS.processHeaders(headers)
    mask_aw_sat     = [np.nan for i in range(n_sat)]
    area_sat        = [np.nan for i in range(n_sat)]
    apex_sat        = [np.nan for i in range(n_sat)]
    height_diff_sat = [np.nan for i in range(n_sat)]
    scl_fac_fr_sat  = [np.nan for i in range(n_sat)]
    def_fac_sat     = [np.nan for i in range(n_sat)]
    exp_fac_sat     = [np.nan for i in range(n_sat)]
    aspect_ratio_frope_sat = [np.nan for i in range(n_sat)]
    status_sat      = ['' for i in range(n_sat)]
    median_btot_over_back_sat = [np.nan for i in range(n_sat)]
    filter_area_threshold_sat = [np.nan for i in range(n_sat)]
    sinthetic_params_Bt_out_sat = [np.nan for i in range(n_sat)]
    sinthetic_params_Bt_RD_sat  = [np.nan for i in range(n_sat)]
    sinthetic_params_FR_out_sat = [np.nan for i in range(n_sat)]
    sinthetic_params_FR_RD_sat  = [np.nan for i in range(n_sat)]
    folder_name_sat             = [np.nan for i in range(n_sat)]
    stats_btot_mask_sat         = [np.nan for i in range(n_sat)]
    stats_back_mask_sat         = [np.nan for i in range(n_sat)]
    stats_cme_mask_sat          = [np.nan for i in range(n_sat)]
    stats_btot_mask_outer_sat   = [np.nan for i in range(n_sat)]
    stats_back_mask_outer_sat   = [np.nan for i in range(n_sat)]
    stats_cme_mask_outer_sat    = [np.nan for i in range(n_sat)]

    print(f'Saving image pair {row} of {df.index.stop-1}')

    for sat in range(n_sat):
        # for sat==2 (LASCO) we divide the current height by 16/6 to match the scale of the other satellites
        if sat==0:
            occ_size_1024 = 100
        elif sat==1:
            occ_size_1024 = 120
        elif sat==2:
            occ_size_1024 = 150
        if sat==2:
            df.loc[row, 'height'] = df.loc[row, 'height']/2.667
        #defining ranges and radius of the occulter
        x = np.linspace(plotranges[sat][0], plotranges[sat][1], num=imsize[0])
        y = np.linspace(plotranges[sat][2], plotranges[sat][3], num=imsize[1])
        xx, yy = np.meshgrid(x, y)
        x_cS, y_cS = center_rSun_pixel(headers, plotranges, sat) 
        # ad hoc correction for the occulter center
        x_cS, y_cS = x_cS+occ_center[sat][0], y_cS+occ_center[sat][1]       
        r = np.sqrt((xx - x_cS)**2 + (yy - y_cS)**2)
        phi = np.arctan2(yy - y_cS, xx - x_cS)
        if mask_from_cloud: 
            #mask for cme outer envelope
            clouds = pyGCS.getGCS(df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row],
                                    satpos, do_rotate_lat=[False, False, True])                         
            x = clouds[sat, :, 1]
            y = clouds[sat, :, 2] # sat has always been 0 in this line, but why ???

            p_x,p_y=deg2px(x,y,plotranges,imsize, sat)
            mask=get_mask_cloud(p_x,p_y,imsize)
        else:
            btot_mask = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row], df['k'][row],
                                      df['ang'][row], imsize=imsize, occrad=size_occ[sat]*0.9, in_sig=1., out_sig=0.0001, nel=1e5, usr_center_off=usr_center_off,
                                      losrange=np.array([-1*df['height'][row] - 2.0 , df['height'][row] + 2.0]))     
            cme_npix= len(btot_mask[btot_mask>0].flatten())
            if cme_npix<=0:
                print("\033[93m WARNING: CME number {} raytracing did not work\033".format(row))                
                break          
            mask = get_cme_mask(btot_mask,inner_cme=inner_hole_mask,occ_size=occ_size_1024)          
            mask_npix= len(mask[mask>0].flatten())
            if mask_npix/cme_npix<0.5:                        
                print("\033[93m WARNING: CME number {} mask is too small compared to cme brigthness image, skipping all views...\033".format(row))
                break
            #breakpoint()
        #adds occulter to the masks and checks for null masks
        mask[r <= size_occ[sat]] = 0  
        mask[r >= size_occ_ext[sat]] = 0
        if len(np.array(np.where(mask==1)).flatten())/len(mask.flatten())<0.005: # only if there is a cme that covers more than 0.5% of the image
            print("\033[93m WARNING: CME number {} mask is null because it is probably behind the occulter, skipping all views...\033".format(row))
            break
       
        # adds mask angluar width to list
        aw   = np.percentile(phi[mask==1],98)-np.percentile(phi[mask==1],2)
        apex = np.percentile(r[mask==1],99)
        #mask_aw_sat.append(aw)
        mask_aw_sat[sat]=aw
        apex_sat[sat]   =apex
        # check for halos, i.e. maks that have pixels with polar angles that span more than 120 deg
        if aw>2*np.pi/3:
            halo_count += 1
            print(f'CME number {row} looks like a halo')

        #masks for occulters
        occ_mask = np.zeros(xx.shape)
        occ_mask[r <= size_occ[sat]] = 1    
        occ_mask_ext = np.zeros(xx.shape)
        occ_mask_ext[r >= size_occ_ext[sat]] = 1 

        if opath_fstructure=='run':
            #creating folders for each case
            folder = os.path.join(OPATH, str(row*len(satpos)+sat))
            if os.path.exists(folder):
                os.system("rm -r " + folder) 
            os.makedirs(folder)
            mask_folder = os.path.join(folder, "mask")
            os.makedirs(mask_folder) 
            #if save_background:
            #    back_folder = os.path.join(folder, "back")
            #    os.makedirs(back_folder)        
        elif opath_fstructure=='check':
            # save all images in the same folder
            folder = OPATH
            mask_folder = folder
        else:
            print('ERROR: opath_fstructure value not recognized')
            sys.exit(1)

        #Total intensity (Btot) figure from raytrace:      
        print(headers[sat]['TELESCOP'], headers[sat]['CRPIX1'], headers[sat]['CRPIX2'],headers[sat]['INSTRUME'], 
              headers[sat]['NAXIS1'], headers[sat]['NAXIS2'], headers[sat]['CRVAL1'], headers[sat]['CRVAL2'], headers[sat]['CROTA'])
        btot = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row],
                             imsize=imsize, occrad=size_occ[sat], in_sig=1.0, out_sig=0.2, nel=1e5, usr_center_off=usr_center_off,
                             losrange=np.array([-1*df['height'][row] - 2.0 , df['height'][row] + 2.0]))
        sinthetic_params_Bt_out_sat[sat] = [1, df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row]]

        btot_outer = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row],
                             imsize=imsize, occrad=size_occ[sat], in_sig=1.0, out_sig=0.075, nel=1e5, usr_center_off=usr_center_off,
                             losrange=np.array([-1*df['height'][row] - 2.0 , df['height'][row] + 2.0]))
        mask_outer_for_filter = get_cme_mask(btot_outer,inner_cme=inner_hole_mask,occ_size=occ_size_1024)
        #diff intensity by adding a second, smaller GCS
        if diff_int_cme:
            # self-similar expansion -> height increase
            height_diff = get_random_lognormal(1)
            if height_diff > df['height'][row]:
                height_diff = df['height'][row]*height_diff
            scl_fac = factor(height_diff, df['height'][row])
            scl_fac_fr_sat[sat] = scl_fac
            #scl_fac = np.random.uniform(low=1.00, high=1.3) # int scaling factor
            def_fac = np.random.uniform(low=0.95, high=1.05, size=3) # deflection and rotation factors [lat, lon, tilt]
            def_fac_sat[sat] = def_fac
            exp_fac = np.random.uniform(low=0.90, high=1,    size=2) # non-self-similar expansion factor [k, ang]
            exp_fac_sat[sat] = exp_fac
            # avoids lat and lon overturns 
            if def_fac[0]*df['CMElon'][row]>180:
                fr_lon = 180
            else:
                fr_lon = def_fac[0]*df['CMElon'][row]
            if def_fac[1]*df['CMElat'][row]>90:
                fr_lat = 90
            else:
                fr_lat = def_fac[1]*df['CMElat'][row]
            btot -= scl_fac*rtraytracewcs(headers[sat], fr_lon, fr_lat ,def_fac[2]*df['CMEtilt'][row],
                                          df['height'][row]-height_diff,df['k'][row]*exp_fac[0], df['ang'][row]*exp_fac[1], imsize=imsize, 
                                          occrad=size_occ[sat], in_sig=1.0, out_sig=0.2, nel=1e5, usr_center_off=usr_center_off,
                                          losrange=np.array([-1*df['height'][row] - 2.0 , df['height'][row] + 2.0]))
            sinthetic_params_Bt_RD_sat[sat] = [scl_fac, fr_lon, fr_lat,def_fac[2]*df['CMEtilt'][row], df['height'][row]-height_diff,df['k'][row]*exp_fac[0], df['ang'][row]*exp_fac[1]]

            btot_inner = rtraytracewcs(headers[sat], fr_lon, fr_lat ,def_fac[2]*df['CMEtilt'][row],
                                          df['height'][row]-height_diff,df['k'][row]*exp_fac[0], df['ang'][row]*exp_fac[1], imsize=imsize, 
                                          occrad=size_occ[sat], in_sig=1.0, out_sig=0.0001, nel=1e5, usr_center_off=usr_center_off,
                                          losrange=np.array([-1*df['height'][row] - 2.0 , df['height'][row] + 2.0]))
            mask_inner_for_filter = get_cme_mask(btot_inner,inner_cme=inner_hole_mask,occ_size=occ_size_1024)

        #adds a flux rope-like structure
        if add_flux_rope:
            if df['height'][row] < 3:
                frope_height_diff = np.random.uniform(low=0.55, high=0.85) # random flux rope height
            else:
                frope_height_diff = np.random.uniform(low=0.45, high=0.75)
            aspect_ratio_frope = np.random.uniform(low=0.2, high=0.4) # random % of k
            aspect_ratio_frope_sat[sat] = aspect_ratio_frope
            scl_fac_fr = np.random.uniform(low=0, high=2) # int scaling factor
            scl_fac_fr_sat[sat]=scl_fac_fr
            btot += scl_fac_fr*rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row]*frope_height_diff,
                                          df['k'][row]*aspect_ratio_frope, df['ang'][row], imsize=imsize,
                                          occrad=size_occ[sat], in_sig=2., out_sig=0.2, nel=1e5, usr_center_off=usr_center_off,
                                          losrange=np.array([-1*df['height'][row] - 2.0 , df['height'][row] + 2.0]))    
            sinthetic_params_FR_out_sat[sat] = [scl_fac_fr, df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row]*frope_height_diff, df['k'][row]*aspect_ratio_frope, df['ang'][row]]
            # uses a differential flux rope
            if diff_int_cme:
                btot -= scl_fac*scl_fac_fr*rtraytracewcs(headers[sat], fr_lon, fr_lat,def_fac[2]*df['CMEtilt'][row], 
                                              df['height'][row]*frope_height_diff-height_diff,df['k'][row]*aspect_ratio_frope*exp_fac[0],df['ang'][row]*exp_fac[1],
                                              imsize=imsize, occrad=size_occ[sat], in_sig=2., out_sig=0.2, nel=1e5, usr_center_off=usr_center_off,
                                              losrange=np.array([-1*df['height'][row] - 2.0 , df['height'][row] + 2.0]))    
                sinthetic_params_FR_RD_sat[sat] = [scl_fac*scl_fac_fr, fr_lon, fr_lat,def_fac[2]*df['CMEtilt'][row], df['height'][row]*frope_height_diff-height_diff, df['k'][row]*aspect_ratio_frope*exp_fac[0], df['ang'][row]*exp_fac[1]]
        #background corona
        back = back_corona[sat]
        if back_rnd_rot:
            back =  scipy.ndimage.rotate(back, np.random.randint(low=0, high=360), reshape=False)
        # final colo rscale is based on the background std w/o occulters
        mean_back = np.mean(back[(r >= size_occ[sat] ) & (r <= size_occ_ext[sat])]) 
        sd_back = np.std(back[(r >= size_occ[sat] ) & (r <= size_occ_ext[sat])])
        # color scale for ouput image
        vmin_back=mean_back-im_range*sd_back
        vmax_back=mean_back+im_range*sd_back
                
        #cme
        #adds a random patchy spatial variation of the cme only
        var_map = low_freq_map(dim=[imsize[0],imsize[1],1],off=[1],var=[1.5],func=[13])        
        var_map = (var_map-np.min(var_map))/(np.max(var_map)-np.min(var_map)) # normalizes to 0-1
        btot *= var_map

        # normalizes CME btot to 0-df['level_cme'][row]*sd_back
        btot = (btot-np.min(btot))/(np.max(btot)-np.min(btot))*df['level_cme'][row]*sd_back

        # checks NaNs in btot
        m_abs_btot=np.abs(np.mean(btot))
        if m_abs_btot is np.nan:
            print("\033[93m WARNING: CME number {} has NaNs in btot, skipping all views...".format(row))            
            #mask_aw_sat.pop()# removes the last appended mask_aw_sat
            break

        #adds poissonian noise to btot
        if cme_noise is not None :
            btot += np.random.poisson(lam=m_abs_btot, size=imsize) - m_abs_btot

        # subtracts the mean value outside the mask
        btot = btot - np.mean(btot[mask==0])
        #Randomly adds the previous CME to have two in one image
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

        # output files base name
        ofile_name = '{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_{:08.3f}_sat{}'.format(
                    df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], sat+1)
        
        # adds row number at the begining of the file name
        if opath_fstructure=='check':
            ofile_name = '{:09d}_'.format(row) + ofile_name

        # saves btot png image
        if cme_only_image:
            ofile=folder + '/' + ofile_name + '_btot_cme_only.png'                 
            fig=save_png(btot,ofile=ofile, range=[-np.max(btot), np.max(btot)])

        # saves back png image
        if back_only_image:
            ofile=folder + '/' + ofile_name + '_btot_back_only.png'         
            fig=save_png(back,ofile=ofile, range=[vmin_back, vmax_back])

        btot_original = btot.copy()
        #adds background
        btot+=back

        #adds occulter 
        if level_occ == 'min':
            btot[r <= size_occ[sat]]     = vmin_back
            btot[r >= size_occ_ext[sat]] = vmin_back
            back[r <= size_occ[sat]]     = vmin_back
            back[r >= size_occ_ext[sat]] = vmin_back
        else: 
            btot[r <= size_occ[sat]]     = level_occ*mean_back 
            btot[r >= size_occ_ext[sat]] = level_occ*mean_back
            back[r <= size_occ[sat]]     = level_occ*mean_back
            back[r >= size_occ_ext[sat]] = level_occ*mean_back

        mask_outer_for_filter2=mask_outer_for_filter.copy()
        mask_outer_for_filter2 [mask_inner_for_filter==1] = 0
        mask_outer_for_filter2[r <= size_occ[sat]] = 0  
        mask_outer_for_filter2[r >= size_occ_ext[sat]] = 0
        #check if the background on the mask area differs from btot +back
        back_on_mask,btot_on_mask,area=flter_non_zero(back,btot,mask_outer_for_filter2)
        #Signal to Noise ratio definition. SNR = (Btot/Back) - 1 evaluated on a specific mask.
        #This is a similar idea than Hinrichs et al. 2021
        SNR = btot_on_mask / back_on_mask - 1
        mean_BA,median_BA,std_BA = statistic_values(SNR)
        #selected_cota=10#5
        #cota = [-1,1]
        #cota = [x * selected_cota for x in cota]
        #area_threshold   = area_outside_std(B/A, cota=cota,positive_only=True)
        area_threshold = area_outside_thresh(SNR, cota=[1,3,5,7,10])

        stats_btot_mask_sat[sat] = stats_caclulations(btot         , mask)
        stats_cme_mask_sat[sat]  = stats_caclulations(btot_original, mask)
        stats_back_mask_sat[sat] = stats_caclulations(back         , mask)
        stats_btot_mask_outer_sat[sat] = stats_caclulations(btot         , mask_outer_for_filter2)
        stats_cme_mask_outer_sat[sat]  = stats_caclulations(btot_original, mask_outer_for_filter2)
        stats_back_mask_outer_sat[sat] = stats_caclulations(back         , mask_outer_for_filter2)
        folder_name_sat[sat] = folder

        aux=''
        status='ok'
        if  np.abs(median_BA) < 0.2:
            if area_threshold[1]/area <= 0.02:
                if opath_fstructure=='check':
                    aux='/rejected'
                status='rejected'
        median_btot_over_back_sat[sat] = median_BA
        filter_area_threshold_sat[sat] = area_threshold/area
        status_sat[sat] = status

        #saves images
        if otype=="fits":
            if save_masks:
                #mask for cme
                cme_mask = fits.PrimaryHDU(mask)
                cme_mask.writeto(mask_folder +'/'+ofile_name+'_mask_2.fits', overwrite=True)
                if sceond_mask is not None:
                    cme_mask = fits.PrimaryHDU(sceond_mask)
                    cme_mask.writeto(mask_folder +'/'+ofile_name+'_mask_3.fits', overwrite=True)               
                #mask for occulter
                occ = fits.PrimaryHDU(occ_mask)
                occ.writeto(mask_folder +'/'+ofile_name+'_mask_0.fits', overwrite=True)
                #mask for ext occulter
                occ = fits.PrimaryHDU(occ_mask_ext)
                occ.writeto(mask_folder +'/'+ofile_name+'_mask_1.fits', overwrite=True)            
            #full image
            cme = fits.PrimaryHDU(btot)
            cme.writeto(folder + '/' + ofile_name+'_btot.fits', overwrite=True)            
            
        elif otype =="png": 
            if save_masks:      
                #mask for cme
                ofile = mask_folder +aux+'/'+ofile_name+'_mask_2.png'
                fig=save_png(mask,ofile=ofile, range=[0, 1])
                if save_leading_edge:
                    ofile = mask_folder +aux+'/'+ofile_name+'_mask_3.png' 
                    fig=save_png(mask_outer_for_filter2,ofile=ofile, range=[0, 1])
                if save_only_cme_mask == False:   
                    if sceond_mask is not None: 
                        ofile = mask_folder +'/'+ofile_name+'_mask_3.png'
                        fig=save_png(sceond_mask,ofile=ofile, range=[0, 1])                      
                    #mask for occulter
                    ofile = mask_folder +'/'+ofile_name+'_mask_0.png'      
                    fig=save_png(occ_mask,ofile=ofile, range=[0, 1])
                    #mask for ext occulter
                    ofile = mask_folder +'/'+ofile_name+'_mask_1.png'
                    fig=save_png(occ_mask_ext,ofile=ofile, range=[0, 1])
            if save_background:
                ofile=folder +aux+ '/' + ofile_name+'_back.png'
                fig=save_png(back,ofile=ofile, range=[vmin_back, vmax_back])
            #full image
            ofile=folder +aux+ '/' + ofile_name+'_btot.png'
            btot_clean = btot.copy()
            # adds middle cross to btot
            if show_middle_cross:
                btot[imsize[0]//2-1:imsize[0]//2+1,:] = vmin_back
                btot[:,imsize[1]//2-1:imsize[1]//2+1] = vmin_back
            if add_mesh_image and otype=='png' and mask_from_cloud:     
                arr_cloud=pnt2arr(x,y,plotranges,imsize, sat)       
                fig=save_png(btot+arr_cloud*vmin_back,ofile=ofile, range=[vmin_back, vmax_back])
            else:
                fig=save_png(btot,ofile=ofile, range=[vmin_back, vmax_back])
            if opath_fstructure=='check':
                plot_histogram(back,btot_clean,mask_outer_for_filter2,folder +aux+ '/' + ofile_name+'_histo.png')
        else:
            print("otype value not recognized")    
        # saves ok case
        ok_cases += 1    

    #save satpos and plotranges
    #satpos_all.append(satpos)
    #plotranges_all.append(plotranges)
    # saves aw
    #mask_aw.append(mask_aw_sat)
    return satpos, plotranges, mask_aw_sat, halo_count, ok_cases, apex_sat, scl_fac_fr_sat, def_fac_sat, exp_fac_sat, aspect_ratio_frope_sat, status_sat, median_btot_over_back_sat, filter_area_threshold_sat, sinthetic_params_Bt_out_sat, sinthetic_params_Bt_RD_sat, sinthetic_params_FR_out_sat, sinthetic_params_FR_RD_sat, stats_btot_mask_sat, stats_back_mask_sat, stats_cme_mask_sat, stats_btot_mask_outer_sat, stats_back_mask_outer_sat, stats_cme_mask_outer_sat, folder_name_sat

def print_stuffs():
    global satpos_all, plotranges_all, mask_aw_all,apex_all,scl_fac_fr_all,def_fac_all,exp_fac_all,aspect_ratio_frope_all,median_btot_over_back_all,filter_area_threshold_all,status_all,halo_count_tot,ok_cases_tot

    print(f'Total Number of OK cases: {ok_cases_tot}')
    print(f'Total Number of aborted cases: {df.index.stop*n_sat -1-ok_cases_tot}')
    print(f'Total Number of halos: {halo_count_tot}')
    #add satpos and plotranges to dataframe and save csv
    df['satpos'] = satpos_all
    df['plotranges'] = plotranges_all
    # add halo count to dataframe
    df['mask AW'] = mask_aw_all
    df['apex'] = apex_all
    df['scl_fac_fr'] = scl_fac_fr_all
    df['def_fac_lon_lat_tilt'] = def_fac_all
    df['exp_fac_k_ang'] = exp_fac_all
    df['aspect ratio'] = aspect_ratio_frope_all
    df['status'] = status_all
    df['median_btot_over_back'] = median_btot_over_back_all
    df['filter_area_threshold'] = filter_area_threshold_all
    df.to_csv(configfile_name)

num_cpus = os.cpu_count()
########### end of paralelize
MAX_WORKERS = num_cpus -2# number of workers for parallel processing
futures = []
index = 0
halo_count_tot = 0
ok_cases_tot = 0
# generate views

########### paralelize this for loop
with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #while len(futures) != MAX_WORKERS: # Task loader
    futures = [executor.submit(create_sintetic_image, row) for row in df.index]
    for future in concurrent.futures.as_completed(futures):
        satpos_f, plotranges_f, mask_aw_sat_f, halo_count_f, ok_cases_f,apex_sat_f, scl_fac_fr_sat_f, def_fac_sat_f, exp_fac_sat_f, aspect_ratio_frope_sat_f, status_sat_f,median_btot_over_back_sat_f,filter_area_threshold_sat_f, sinthetic_params_Bt_out_sat_f, sinthetic_params_Bt_RD_sat_f, sinthetic_params_FR_out_sat_f, sinthetic_params_FR_RD_sat_f, stats_btot_mask_sat_f, stats_back_mask_sat_f, stats_cme_mask_sat_f, stats_btot_mask_outer_sat_f, stats_back_mask_outer_sat_f, stats_cme_mask_outer_sat_f, folder_name_sat_f = future.result()
        satpos_all.append(satpos_f)
        plotranges_all.append(plotranges_f)
        mask_aw_all.append(mask_aw_sat_f)
        apex_all.append(apex_sat_f)
        scl_fac_fr_all.append(scl_fac_fr_sat_f)
        def_fac_all.append(def_fac_sat_f)
        exp_fac_all.append(exp_fac_sat_f)
        status_all.append(status_sat_f)
        aspect_ratio_frope_all.append(aspect_ratio_frope_sat_f)
        median_btot_over_back_all.append(median_btot_over_back_sat_f)
        filter_area_threshold_all.append(filter_area_threshold_sat_f)
        sinthetic_params_Bt_out_all.append(sinthetic_params_Bt_out_sat_f)
        sinthetic_params_Bt_RD_all.append(sinthetic_params_Bt_RD_sat_f)
        sinthetic_params_FR_out_all.append(sinthetic_params_FR_out_sat_f)
        sinthetic_params_FR_RD_all.append(sinthetic_params_FR_RD_sat_f)
        stats_btot_mask_all.append(stats_btot_mask_sat_f)
        stats_back_mask_all.append(stats_back_mask_sat_f)
        stats_cme_mask_all.append(stats_cme_mask_sat_f)
        stats_btot_mask_outer_all.append(stats_btot_mask_outer_sat_f)
        stats_back_mask_outer_all.append(stats_back_mask_outer_sat_f)
        stats_cme_mask_outer_all.append(stats_cme_mask_outer_sat_f)
        folder_name_all.append(folder_name_sat_f)
        halo_count_tot = halo_count_f +halo_count_tot
        ok_cases_tot = ok_cases_f + ok_cases_tot
print_stuffs()
print('Finished :-)')