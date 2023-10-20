
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from ext_libs.rebin import rebin
from pyGCS_raytrace import pyGCS
from nn.utils.coord_transformation import deg2px
from nn_training.get_cme_mask import get_mask_cloud
from nn.utils.gcs_mask_generator import maskFromCloud
from nn.neural_cme_seg.neural_cme_seg import neural_cme_segmentation

def read_fits(file_path, header=False, imageSize=[512,512]):
    try:       
        img = fits.open(file_path)[0].data
        img = rebin(img, imageSize)   
        if header:
            return img, fits.open(file_path)[0].header
        else:
            return img 
    except:
        print(f'WARNING. I could not find file {file_path}')
        return None

def get_NN(odir,sat):
    df_list=[]
    odir=odir+sat+"/"
    ext_folders = os.listdir(odir)
    for ext_folder in ext_folders:
        odir_filter=odir+ext_folder+"/filtered"
        if os.path.exists(odir_filter):
            csv_path=odir_filter+"/"+ext_folder+"_filtered_stats"
            df=pd.read_csv(csv_path)
            df_list.append(df)
    
    df_full = pd.concat(df_list, ignore_index=True)
    return df_full


def get_kincat(kincat_orig_dir,sat):
    col_names=["HEL","CME","PRE_DATE","PRE_TIME","END_DATE","END_TIME","CARLON","STONEY","LAT","TILT", "ASP_RATIO","H_ANGLE","DATE","TIME","APEX_SPEED","CME_MASS","SPEED_FPF","FPF_LON","FPF_LAT","SPEED_SSEF","SSEF_LON","SSEF_LAT","SPEED_HMF","HMF_LON","HMF_LAT"]
    df = pd.read_csv(kincat_orig_dir, sep = "\t")
    df=df.drop([0,1])
    df.columns=col_names
    df = df.reset_index(drop=True)
    df['PRE_DATE_TIME'] = pd.to_datetime(df['PRE_DATE'] + ' ' + df['PRE_TIME'])
    df['END_DATE_TIME'] = pd.to_datetime(df['END_DATE'] + ' ' + df['END_TIME'])
    return df

def filter_param(NN, kincat):
    kincat_cols=["PRE_DATE","PRE_TIME","END_DATE","END_TIME"]
    columns=["NN_DATE_TIME","HEL","CME","PRE_DATE","PRE_TIME","LAST_DATE","LAST_TIME","CARLON","STONEY","LAT","TILT", "ASP_RATIO","H_ANGLE","DATE","TIME","APEX_SPEED","CME_MASS","SPEED_FPF","FPF_LON","FPF_LAT","SPEED_SSEF","SSEF_LON","SSEF_LAT","SPEED_HMF","HMF_LON","HMF_LAT",'PRE_DATE_TIME','END_DATE_TIME']


    NN['DATE_TIME'] = pd.to_datetime(NN['DATE_TIME'])
    for cols in kincat_cols:
        if cols.endswith("DATE"):
            kincat[cols] = pd.to_datetime(kincat[cols],format='%Y-%m-%d')
        else:
            kincat[cols] = pd.to_datetime(kincat[cols],format="%H:%M")

    NN.sort_values(by='DATE_TIME', inplace=True)
    kincat.sort_values(by='PRE_DATE', inplace=True)
    NN = NN.reset_index(drop=True)
    kincat=kincat.reset_index(drop=True)
    
    df=[]
    for i in range(len(NN["DATE_TIME"])):
        
        nn_date = pd.to_datetime(NN["DATE_TIME"][i].date())
        coincidences=kincat.loc[(kincat['PRE_DATE'] <= nn_date) & (nn_date <= kincat['END_DATE'])]
        param= coincidences.values.tolist()
        if param:
            param[0].insert(0,NN["DATE_TIME"][i])
            df.append(param[0])
    df_full= pd.DataFrame(df, columns=columns)
    return df_full  

def _rec2pol(mask,imsize,mask_threshold):
        '''
        Converts the x,y mask to polar coordinates
        Only pixels above the mask_threshold are considered
        TODO: Consider arbitrary image center
        '''
        nans = np.full(imsize, np.nan)
        pol_mask=[]
        #creates an array with zero value inside the mask and Nan value outside             
        masked = nans.copy()
        masked[:, :][mask > mask_threshold] = 0   
        #calculates geometric center of the image
        height, width = masked.shape
        center_x = width // 2
        center_y = height // 2
        #calculates distance to the point and the angle for the positive x axis
        for x in range(width):
            for y in range(height):
                value=masked[x,y]
                if not np.isnan(value):
                    x_dist = (x-center_x)
                    y_dist = (y-center_y)
                    distance= np.sqrt(x_dist**2 + y_dist**2)
                    angle = np.arctan2(x_dist,y_dist)
                    if angle<0:
                        angle+=2*np.pi
                    pol_mask.append([distance,angle])
        return pol_mask


def _compute_mask_prop(mask,imsize,mask_threshold, plate_scl=1, filter_halos=True, occulter_size=0):    
    '''
    Computes the CPA, AW and apex radius for the given 'CME' masks.
    If the mask label is not "CME" and if is not a Halo and filter_halos=True, it returns None
    plate_scl: if defined the apex radius is scaled based on the plate scale of the image. If 0, no filter is applied
    filter_halos: if True, filters the masks with boxes center within the occulter size, and wide_ angle > MAX_WIDE_ANG
    TODO: Compute apex in Rs; Use generic image center
    '''
    max_wide_ang = np.radians(270.) # maximum angular width [deg]
    prop_list=[]
    pol_mask=_rec2pol(mask,imsize,mask_threshold)
    if (pol_mask is not None):            
        angles = [s[1] for s in pol_mask]
        if len(angles)>0:
            # checks for the case where the cpa is close to 0 or 2pi
            if np.max(angles)-np.min(angles) >= 0.9*2*np.pi:
                angles = [s-2*np.pi if s>np.pi else s for s in angles]
            cpa_ang= np.median(angles)
            if cpa_ang < 0:
                cpa_ang += 2*np.pi
            wide_ang=np.abs(np.percentile(angles, 95)-np.percentile(angles, 5))
            #calculates diferent angles an parameters
            distance = [s[0] for s in pol_mask]
            distance_abs= max(distance, key=abs)
            idx_dist = distance.index(distance_abs)
            apex_dist= distance[idx_dist] * plate_scl
                                                            
            if filter_halos:
                if wide_ang < max_wide_ang:
                    prop_list.append([i,cpa_ang, wide_ang, apex_dist])  
                else:
                    prop_list.append([i, np.nan, np.nan, np.nan])
            else:
                prop_list.append([i,cpa_ang, wide_ang, apex_dist])
    else:
        prop_list.append([i, np.nan, np.nan, np.nan])

    return prop_list         

def plot_masks(img,parameters, satpos, plotranges,imsize, opath, namefile):
    
    mask_infered = maskFromCloud(parameters, sat=0, satpos=[satpos], imsize=imsize, plotranges=[plotranges])
    #mask_infered[occulter_mask > 0] = 0  # Setting 0 where the occulter is
    mask_infered=mask_infered[::-1, :]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    img = np.squeeze(img)
    ax[0].imshow(img, vmin=0, vmax=1, cmap='gray')
    ax[1].imshow(mask_infered, alpha=0.4)
    ax[2].imshow(img, vmin=0, vmax=1, cmap='gray')
    ax[2].imshow(mask_infered, alpha=0.4)
    os.makedirs(opath, exist_ok=True)
    plt.savefig(os.path.join(opath, namefile))
    plt.close()
    return mask_infered



###################################################################################### MAIN #########################################################################################
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
kincat_orig_dir=repo_dir+"/nn_training/kincat/helcatslist_20160601.txt"
odir="/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat_L1/"

sat="cor2_b"#cor2_b
imageSize=[512,512]
mod_scale=True
mask_threshold=0.6 # value to consider a pixel belongs to the objec
col_names=["DATE_TIME","MASK","CPA_ANG","WIDE_ANG","APEX_DIST"]

NN=get_NN(odir,sat)
kincat_orig= get_kincat(kincat_orig_dir,sat)
df=filter_param(NN,kincat_orig)
kincat_orig['PRE_DATE_TIME'] = pd.to_datetime(kincat_orig['PRE_DATE_TIME'])
kincat_orig['END_DATE_TIME'] = pd.to_datetime(kincat_orig['END_DATE_TIME'])

all_mask_props=[]                                     
for i in range(len(df)):
    try:
        date=pd.to_datetime(df["NN_DATE_TIME"][i])
        folder= str(date)[:-9].replace("-", "")
        file_name=folder+"_"+str(date)[11:].replace(":", "")
        path=(glob.glob(odir+sat+"/"+folder+"/"+file_name+"*"+".fits"))[0]
        opath=odir+sat+"/"+folder+"/"+"gcs_masks"
        print("Working on file "+str(file_name)+", "+str(i)+" of " +str(len(df)))

        if not os.path.exists(opath):
            os.makedirs(opath)
        img, header = read_fits(path, header=True,imageSize=imageSize)
        plt_scl = header['CDELT1']
        satpos, plotranges = pyGCS.processHeaders([header])

        #CME parameters
        result = kincat_orig.loc[((kincat_orig['PRE_DATE_TIME'] - timedelta(hours=1)) <= date) & (kincat_orig['END_DATE_TIME'] >= date)]
        parameters=[]
        apex_dist=NN.loc[NN["DATE_TIME"]==df["NN_DATE_TIME"][i], "APEX_DIST"]
        long = float(result['CARLON'].values[0].replace( ',', '.'))
        lat = float(result['LAT'].values[0].replace( ',', '.'))
        tilt = float(result['TILT'].values[0].replace( ',', '.'))
        heith = float(apex_dist* plt_scl / (header['RSUN']))
        asp_ratio = float(result['ASP_RATIO'].values[0].replace( ',', '.'))
        half_ang = float(result['H_ANGLE'].values[0].replace( ',', '.'))
        parameters.append([long,lat,tilt,heith,asp_ratio,half_ang])
        
        mask_inferd= plot_masks(img,parameters[0],satpos[0], plotranges[0],imsize=imageSize, opath=opath, namefile=f'GCS_mask_{file_name}.png')
        mask_props= _compute_mask_prop(mask_inferd,imageSize,mask_threshold)
        

        if df["NN_DATE_TIME"][i].date()!=df["NN_DATE_TIME"][i+1].date():
            mask_props[0].insert(0,df["NN_DATE_TIME"][i])
            
            all_mask_props.append(mask_props[0])
            df_mask_prop = pd.DataFrame(all_mask_props,columns=col_names)
            df_mask_prop.to_csv(opath+"/"+"GCS_mask_stats", index=False)
            all_mask_props=[]

        else:
            mask_props[0].insert(0,df["NN_DATE_TIME"][i])
            all_mask_props.append(mask_props[0])
    
    except:
        print("File not found")
            


