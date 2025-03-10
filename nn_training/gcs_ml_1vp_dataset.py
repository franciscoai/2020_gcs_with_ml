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
from scipy.stats import kstest,chisquare,ks_2samp

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

def plot_histogram(back,btot,mask,filename):
    '''
    Plots histograms of back and btot inside the mask
    '''
    A=back[mask==1].copy().flatten()
    B=btot[mask==1].copy().flatten()
    min_val = np.min([np.percentile(A,2) ,np.percentile(B,2)])
    max_val = np.max([np.percentile(A,98),np.percentile(B,98)])
    num_bins = 50  
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    plt.hist(A, bins=bin_edges, alpha=0.5, label='Back', color='blue')  
    plt.hist(B, bins=bin_edges, alpha=0.5, label='Btot', color='green') 
    plt.legend(loc='upper right')

    ax = plt.gca()
    y_max = ax.get_ylim()[1]
    mean_A = np.mean(A)
    median_A = np.median(A)
    std_A = np.std(A)
    mean_B = np.mean(B)
    std_B = np.std(B)
    median_B = np.median(B)

    bins = np.histogram_bin_edges(np.hstack((B, A)), bins=100)
    hist1 = np.histogram(A, bins=bins)
    hist2 = np.histogram(B, bins=bins)
    bin_midpoints1 = (hist1[1][:-1] + hist1[1][1:]) / 2
    data1 = np.repeat(bin_midpoints1, hist1[0].astype(int))
    bin_midpoints2 = (hist2[1][:-1] + hist2[1][1:]) / 2
    data2 = np.repeat(bin_midpoints2, hist2[0].astype(int))
    p_value = ks_2samp(data1, data2).pvalue

    plt.text(0.03, 0.95 ,f'Back:\nMean: {mean_A:.2e}\nStd: {std_A:.2e}\nMedian: {median_A:.2e}',transform=ax.transAxes,fontsize=10,color='blue',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
    plt.text(0.33, 0.95 ,f'Btot:\nMean: {mean_B:.2e}\nStd: {std_B:.2e}\nMedian: {median_B:.2e}',transform=ax.transAxes,fontsize=10,color='green',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
    plt.text(0.63, 0.65 ,f'\nP-value: {p_value:.2e}',transform=ax.transAxes,fontsize=10,color='red',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label='x = 1')
    plt.legend(loc='upper right')
    #mask_folder2 = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_20250212/test/'
    plt.savefig(filename)
    plt.close()

    #Ratio of the histograms
    if np.any(A==0) or np.any(B==0):
        non_zero = (A != 0) & (B != 0)
        A = A[non_zero]
        B = B[non_zero]
    min_val = np.percentile(B/A,2) 
    max_val = np.percentile(B/A,98)
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    plt.hist(B/A, bins=bin_edges, alpha=0.5, label='Btot/Back', color='red')
    mean_BA = np.mean(B/A)
    median_BA = np.median(B/A)
    std_BA   = np.std(B/A)
    plt.text(0.03, 0.95 ,f'\nMean: {mean_BA:.2e}\nStd: {std_BA:.2e}\nMedian: {median_BA:.2e}',transform=ax.transAxes,fontsize=10,color='blue',verticalalignment='top',bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) 
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label='x = 1')
    plt.legend(loc='upper right')
    plt.savefig(filename.replace('histo.png','histo_ratio.png'))
    plt.close()

######Main

# CONSTANTS
#paths
DATA_PATH = '/gehme/data'
#OPATH = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_20240830'
OPATH = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_20250212'
opath_fstructure='check'#'run' # use 'check' to save all the ouput images in the same dir together for easier checkout
                         # use 'run' to save each image in a different folder as required for training dataset
#Syntethic image options
# morphology
diff_int_cme=True # set to use a differential intensity CME image
add_flux_rope = True # set to add a flux rope-like structure to the cme image
par_names = ['CMElon', 'CMElat', 'CMEtilt', 'height', 'k','ang', 'level_cme'] # GCS parameters plus CME intensity level
par_units = ['deg', 'deg', 'deg', 'Rsun','','deg','frac of back sdev'] # par units
par_rng = [[-180,180],[-70,70],[-90,90],[1.5,20],[0.2,0.6], [5,65],[2,7]] 
par_num = 50  # total number of GCS samples that will be generated. n_sat images are generated per GCS sample.
rnd_par=True # set to randomnly shuffle the generated parameters linspace 
#background
n_sat = 3 #number of satellites to  use [Cor2 A, Cor2 B, Lasco C2]
back_rnd_rot=False # set to randomly rotate the background image around its center
same_corona=False # Set to True use a single corona back for all par_num cases
same_position=True # Set to True to use the same set of satteite positions(not necesarly the same background image)
rnd_int_occ=0.1 # set to randomly increase the internal occulter size by a max fraction of its size. Use None to keep the constant size given by get_corona
rnd_ext_occ=0.1 # set to randomly reduce the external occulter size by a max fraction of its size. Use None to keep the constant size given by get_corona
#noise
cme_noise= True # set to add poissonian noise to the cme image
#relative int levels
level_occ='min' # level of the occulter relative to the background level. Note that png images are saved in 0-255 scale
                # The image range is mapped to 0-255 suning the value given by im_range. 
                # Use 'min' to set the occulter to the minimum value, thus appearing as 0 in png images
im_range=3 # range of the color scale of the output final syntethyc image in std dev around the mean

# Output images to save
otype="png" # set the ouput file type: 'png' or 'fits'
imsize=np.array([512, 512], dtype='int32') # output image size
add_mesh_image=False # adds the GCS mesh on top of the image (only for otype='png')
cme_only_image=False # set to True to save an addditional image with only the cme without the background corona
back_only_image = False # set to True to save an addditional image with only the background corona without the cme
save_masks = True # set to True to save the masks images
save_only_cme_mask = False # set to True to save only the cme mask and not the occulter masks. save_masks must be True
inner_hole_mask=False #Set to True to produce a mask that contains the inner hole of the GCS (if visible)
mask_from_cloud=False #True #True to calculete mask from clouds, False to do it from ratraycing total brigthness image
two_cmes = False # set to include two cme per image on some (random) cases
show_middle_cross = False # set to show the middle cross of the image

#### main
if opath_fstructure=='check':
    save_masks = True #False
    cme_only_image = False
    back_only_image = False
    add_mesh_image = True
    rnd_par=True # False 
    save_only_cme_mask = True
    diff_int_cme = True #False
    par_rng = [[-180,180],[-70,70],[-90,90],[1.5,20],[0.2,0.6], [5,65],[2,7]]
    #par_rng = [[180,180],[70,70],[90,90],[20,20],[0.6,0.6], [65,65],[10,10]]
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
mask_aw = []
halo_count = 0
ok_cases = 0
usr_center_off = None

# generate views
########### paralelize this for loop
for row in df.index:
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
    mask_aw_sat = []
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
                                      df['ang'][row], imsize=imsize, occrad=size_occ[sat]*0.9, in_sig=1., out_sig=0.0001, nel=1e5, usr_center_off=usr_center_off)     
            cme_npix= len(btot_mask[btot_mask>0].flatten())
            if cme_npix<=0:
                print("\033[93m WARNING: CME number {} raytracing did not work\033".format(row))                
                break          
            mask = get_cme_mask(btot_mask,inner_cme=inner_hole_mask,occ_size=occ_size_1024)          
            mask_npix= len(mask[mask>0].flatten())
            if mask_npix/cme_npix<0.5:                        
                print("\033[93m WARNING: CME number {} mask is too small compared to cme brigthness image, skipping all views...\033".format(row))
                break
        #adds occulter to the masks and checks for null masks
        mask[r <= size_occ[sat]] = 0  
        mask[r >= size_occ_ext[sat]] = 0
        if len(np.array(np.where(mask==1)).flatten())/len(mask.flatten())<0.005: # only if there is a cme that covers more than 0.5% of the image
            print("\033[93m WARNING: CME number {} mask is null because it is probably behind the occulter, skipping all views...\033".format(row))
            break
    
        # adds mask angluar width to list
        aw = np.percentile(phi[mask==1],95)-np.percentile(phi[mask==1],5)
        mask_aw_sat.append(aw)
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
                             imsize=imsize, occrad=size_occ[sat], in_sig=0.5, out_sig=0.1, nel=1e5, usr_center_off=usr_center_off)

        #diff intensity by adding a second, smaller GCS
        if diff_int_cme:
            # self-similar expansion -> height increase
            height_diff = get_random_lognormal(1)
            if height_diff > df['height'][row]:
                height_diff = df['height'][row]*height_diff
                breakpoint()
            scl_fac = np.random.uniform(low=0.90, high=1.3) # int scaling factor
            def_fac = np.random.uniform(low=0.95, high=1.05, size=3) # deflection and rotation factors [lat, lon, tilt]
            exp_fac = np.random.uniform(low=0.9, high=1, size=2) # non-self-similar expansion factor [k, ang]
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
                                          occrad=size_occ[sat], in_sig=0.5, out_sig=0.1, nel=1e5, usr_center_off=usr_center_off)
        
        #adds a flux rope-like structure
        if add_flux_rope:
            if df['height'][row] < 3:
                frope_height_diff = np.random.uniform(low=0.55, high=0.85) # random flux rope height
            else:
                frope_height_diff = np.random.uniform(low=0.45, high=0.75)
            aspect_ratio_frope = np.random.uniform(low=0.2, high=0.4) # random aspect ratio
            scl_fac_fr = np.random.uniform(low=0, high=2) # int scaling factor
            btot += scl_fac_fr*rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row]*frope_height_diff,
                                          df['k'][row]*aspect_ratio_frope, df['ang'][row], imsize=imsize, 
                                          occrad=size_occ[sat], in_sig=2., out_sig=0.2, nel=1e5, usr_center_off=usr_center_off)    
            # uses a differential flux rope
            if diff_int_cme:
                btot -= scl_fac*scl_fac_fr*rtraytracewcs(headers[sat], fr_lon, fr_lat,def_fac[2]*df['CMEtilt'][row], 
                                              df['height'][row]*frope_height_diff-height_diff,df['k'][row]*aspect_ratio_frope*exp_fac[0],df['ang'][row]*exp_fac[1],
                                              imsize=imsize, occrad=size_occ[sat], in_sig=2., out_sig=0.2, nel=1e5, usr_center_off=usr_center_off)    
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
            mask_aw_sat.pop()# removes the last appended mask_aw_sat
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

        #check if the background on the mask area differs from btot +back
        btot_pixels = btot[mask==1].flatten()
        back_pixels = back[mask==1].flatten()
        bins = np.histogram_bin_edges(np.hstack((btot_pixels, back_pixels)), bins=100)
        # Important: Use the same bins for both histograms
        hist1 = np.histogram(btot_pixels, bins=bins)
        hist2 = np.histogram(back_pixels, bins=bins)
        bin_midpoints1 = (hist1[1][:-1] + hist1[1][1:]) / 2
        data1 = np.repeat(bin_midpoints1, hist1[0].astype(int)) # Repeat midpoints based on histogram counts.
        bin_midpoints2 = (hist2[1][:-1] + hist2[1][1:]) / 2
        data2 = np.repeat(bin_midpoints2, hist2[0].astype(int))
        p_valor = ks_2samp(data1, data2).pvalue

        
        if  p_valor < 0.05:
            #a p-value < 0.05 indicates that the two distributions are significantly different'
            aux=''
        else:
            #the idea is to skip this cases, since the CME is not visible
            aux='/rejected'
        string_pvalue= "{:.2e}".format(p_valor)

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

            #full image
            ofile=folder +aux+ '/' + ofile_name+'_btot.png'
            # adds middle cross to btot
            if show_middle_cross:
                btot[imsize[0]//2-1:imsize[0]//2+1,:] = vmin_back
                btot[:,imsize[1]//2-1:imsize[1]//2+1] = vmin_back
            if add_mesh_image and otype=='png' and mask_from_cloud:     
                arr_cloud=pnt2arr(x,y,plotranges,imsize, sat)       
                fig=save_png(btot+arr_cloud*vmin_back,ofile=ofile, range=[vmin_back, vmax_back])
            else:
                fig=save_png(btot,ofile=ofile, range=[vmin_back, vmax_back])

            plot_histogram(back,btot,mask,folder +aux+ '/' + ofile_name+'_histo.png')
        else:
            print("otype value not recognized")    
        # saves ok case
        ok_cases += 1    

    #save satpos and plotranges
    satpos_all.append(satpos)
    plotranges_all.append(plotranges)
    # saves aw
    mask_aw.append(mask_aw_sat)

########### end of paralelize

print(f'Total Number of OK cases: {ok_cases}')
print(f'Total Number of aborted cases: {df.index.stop*n_sat -1-ok_cases}')
print(f'Total Number of halos: {halo_count}')
#add satpos and plotranges to dataframe and save csv
df['satpos'] = satpos_all
df['plotranges'] = plotranges_all
# add halo count to dataframe
df['mask AW per sat'] = mask_aw
df.to_csv(configfile_name)