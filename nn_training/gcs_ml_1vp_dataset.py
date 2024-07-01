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
import sunpy.map
import pandas as pd
from nn_training.low_freq_map import low_freq_map
import scipy
from nn.utils.coord_transformation import deg2px, center_rSun_pixel, pnt2arr

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

######Main

# CONSTANTS
#paths
DATA_PATH = '/gehme/data'
OPATH = '/gehme/projects/2020_gcs_with_ml/data/cme_seg_20240701'
opath_fstructure='run' # use 'check' to save all the ouput images in the same dir together for easier checkout
                         # use 'run' to save each image in a different folder as required for training dataset
#Syntethic image options
# morphology
diff_int_cme=True # set to use a differential intensity CME image
add_flux_rope = True # set to add a flux rope-like structure to the cme image
par_names = ['CMElon', 'CMElat', 'CMEtilt', 'height', 'k','ang', 'level_cme'] # GCS parameters plus CME intensity level
par_units = ['deg', 'deg', 'deg', 'Rsun','','deg','frac of back sdev'] # par units
par_rng = [[-180,180],[-70,70],[-90,90],[1.5,20],[0.2,0.6], [5,65],[2,7]] 
par_num = 80000 # total number of GCS samples that will be generated. n_sat images are generated per GCS sample.
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
mesh_only_image=False # set to also save a png with the GCSmesh (only for otype='png')
cme_only_image=False # set to True to save an addditional image with only the cme without the background corona
back_only_image = False # set to True to save an addditional image with only the background corona without the cme
save_masks = True # set to True to save the masks images
inner_hole_mask=False #Set to True to make the cme mask excludes the inner void of the gcs (if visible) 
mask_from_cloud=True #True to calculete mask from clouds, False to do it from ratraycing total brigthness image
two_cmes = False # set to include two cme per image on some (random) cases

#### main
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

# generate views
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
        if sat==2:
            df['height'][row] = df['height'][row]/2.667
        #defining ranges and radius of the occulter
        x = np.linspace(plotranges[sat][0], plotranges[sat][1], num=imsize[0])
        y = np.linspace(plotranges[sat][2], plotranges[sat][3], num=imsize[1])
        xx, yy = np.meshgrid(x, y)
        x_cS, y_cS = center_rSun_pixel(headers, plotranges, sat) 
        # corrects for the occulter center
        x_cS, y_cS = x_cS+occ_center[sat][0], y_cS+occ_center[sat][1]       
        r = np.sqrt((xx - x_cS)**2 + (yy - y_cS)**2)
        phi = np.arctan2(yy - y_cS, xx - x_cS)
        if mask_from_cloud:
            #mask for cme outer envelope
            clouds = pyGCS.getGCS(df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], satpos)                
            x = clouds[sat, :, 1]
            y = clouds[0, :, 2]
            p_x,p_y=deg2px(x,y,plotranges,imsize, sat)
            mask=get_mask_cloud(p_x,p_y,imsize)
        else:
            btot_mask = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row], df['k'][row],
                                      df['ang'][row], imsize=imsize, occrad=size_occ[sat], in_sig=1., out_sig=0.1, nel=1e5)     
            cme_npix= len(btot_mask[btot_mask>0].flatten())
            if cme_npix<=0:
                print(f'WARNING: CME number {row} raytracing did not work')
                break          
            mask = get_cme_mask(btot_mask,inner_cme=inner_hole_mask)          
            mask_npix= len(mask[mask>0].flatten())
            if mask_npix/cme_npix<0.9:
                print(f'WARNING: CME number {row} mask is too small compared to cme brigthness image, skipping all views...')
                break

        #adds occulter to the masks and checks for null masks
        mask[r <= size_occ[sat]] = 0  
        mask[r >= size_occ_ext[sat]] = 0
        if len(np.array(np.where(mask==1)).flatten())/len(mask.flatten())<0.005: # only if there is a cme that covers more than 0.5% of the image
            print(f'WARNING: CME number {row} mask is null because it is probably behind the occulter, skipping all views...')
            break
       
        # adds mask angluar width to list
        aw = np.max(phi[mask==1])-np.min(phi[mask==1])
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
        btot = rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row],
                             imsize=imsize, occrad=size_occ[sat], in_sig=0.5, out_sig=0.25, nel=1e5)

        #diff intensity by adding a second, smaller GCS
        if diff_int_cme:
            # self-similar expansion -> height increase
            height_diff = np.random.uniform(low=0, high=1) # in Sr. 1 Sr is covered in 15 min at 1500 mk/s
            if height_diff > df['height'][row]:
                height_diff = df['height'][row]*height_diff
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
                                          occrad=size_occ[sat], in_sig=0.8, out_sig=0.2, nel=1e5)
        
        #adds a flux rope-like structure
        if add_flux_rope:
            if df['height'][row] < 3:
                frope_height_diff = np.random.uniform(low=0.55, high=0.85) # random flux rope height
            else:
                frope_height_diff = np.random.uniform(low=0.45, high=0.75)
            aspect_ratio_frope = np.random.uniform(low=0.4, high=0.8) # random aspect ratio
            scl_fac_fr = np.random.uniform(low=0, high=2) # int scaling factor
            btot += scl_fac_fr*rtraytracewcs(headers[sat], df['CMElon'][row], df['CMElat'][row],df['CMEtilt'][row], df['height'][row]*frope_height_diff,
                                          df['k'][row]*aspect_ratio_frope, df['ang'][row], imsize=imsize, occrad=size_occ[sat], in_sig=0.8, out_sig=0.2, nel=1e5)    
            # uses a differential flux rope
            if diff_int_cme:
                btot -= scl_fac*scl_fac_fr*rtraytracewcs(headers[sat], fr_lon, fr_lat,def_fac[2]*df['CMEtilt'][row], 
                                              df['height'][row]*frope_height_diff-height_diff,df['k'][row]*aspect_ratio_frope*exp_fac[0],df['ang'][row]*exp_fac[1],
                                              imsize=imsize, occrad=size_occ[sat], in_sig=0.8, out_sig=0.2, nel=1e5)    
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

        #adds poissonian noise to btot
        if cme_noise is not None :
            btot += np.random.poisson(lam=np.abs(np.mean(btot)), size=imsize) - np.abs(np.mean(btot))

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
            ofile_name = '{:04d}_'.format(row) + ofile_name

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
        else: 
            btot[r <= size_occ[sat]]     = level_occ*mean_back 
            btot[r >= size_occ_ext[sat]] = level_occ*mean_back 

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
                ofile = mask_folder +'/'+ofile_name+'_mask_2.png'
                fig=save_png(mask,ofile=ofile, range=[0, 1])
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
            ofile=folder + '/' + ofile_name+'_btot.png'
            fig=save_png(btot,ofile=ofile, range=[vmin_back, vmax_back])
        else:
            print("otype value not recognized")    

        if mesh_only_image and otype=='png':
            # overplot GCS mesh to cme figure
            clouds = pyGCS.getGCS(df['CMElon'][row], df['CMElat'][row], df['CMEtilt'][row], df['height'][row], df['k'][row], df['ang'][row], satpos)                
            x = clouds[sat, :, 1]
            y = clouds[0, :, 2]         
            arr_cloud=pnt2arr(x,y,plotranges,imsize, sat)
            ofile = folder + '/' + ofile_name+'_mesh.png'
            fig=save_png(arr_cloud,ofile=ofile, range=[0, 1]) 
        # saves ok case
        ok_cases += 1    

    #save satpos and plotranges
    satpos_all.append(satpos)
    plotranges_all.append(plotranges)
    # saves aw
    mask_aw.append(mask_aw_sat)
print(f'Total Number of OK cases: {ok_cases}')
print(f'Total Number of aborted cases: {df.index.stop*n_sat -1-ok_cases}')
print(f'Total Number of halos: {halo_count}')
#add satpos and plotranges to dataframe and save csv
df['satpos'] = satpos_all
df['plotranges'] = plotranges_all
# add halo count to dataframe
df['mask AW per sat'] = mask_aw
df.to_csv(configfile_name)