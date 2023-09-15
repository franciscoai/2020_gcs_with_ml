
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

def plot_masks(img,parameters, satpos, plotranges,imsize, opath, namefile):
    
    mask_infered = maskFromCloud(parameters, sat=0, satpos=[satpos], imsize=imsize, plotranges=[plotranges])
    #mask_infered[occulter_mask > 0] = 0  # Setting 0 where the occulter is
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    
    for i in range(2):
        img = np.squeeze(img)
        ax[i].imshow(img, vmin=0, vmax=1, cmap='gray')
        # if i == 0:
        #     ax[i].imshow(mask, alpha=0.4)
        #     ax[i].set_title('Target Mask')
        # else:
        ax[i].imshow(mask_infered, alpha=0.4)
        #ax[i].set_title(f'Prediction Mask: {loss}')
    
    masks_dir = os.path.join(opath, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    plt.savefig(os.path.join(masks_dir, namefile))
    plt.close() 



###################################################################################### MAIN #########################################################################################
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
kincat_orig_dir=repo_dir+"/nn_training/kincat/helcatslist_20160601.txt"
odir="/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat_L1/"

sat="cor2_a"#cor2_b
imageSize=[512,512]

NN=get_NN(odir,sat)
kincat_orig= get_kincat(kincat_orig_dir,sat)
df=filter_param(NN,kincat_orig)
kincat_orig['PRE_DATE_TIME'] = pd.to_datetime(kincat_orig['PRE_DATE_TIME'])
kincat_orig['END_DATE_TIME'] = pd.to_datetime(kincat_orig['END_DATE_TIME'])
                                             
for i in range(len(df)):
    date=pd.to_datetime(df["NN_DATE_TIME"][i])
    folder= str(date)[:-9].replace("-", "")
    file_name=folder+"_"+str(date)[11:].replace(":", "")
    path=(glob.glob(odir+sat+"/"+folder+"/"+file_name+"*"+".fits"))[0]
    opath=odir+sat+"/"+folder+"/"+file_name+"/"+"gcs_masks"
   
    if not os.path.exists(opath):
        os.makedirs(opath)
    img, header = read_fits(path, header=True,imageSize=imageSize)
    if header['NAXIS1'] != imageSize[0]:
        plt_scl = header['CDELT1'] * header['NAXIS1']/imageSize[0] 
    else:
        plt_scl = header['CDELT1']
    
    result = kincat_orig.loc[(kincat_orig['PRE_DATE_TIME'] <= date) & (kincat_orig['END_DATE_TIME'] >= date)]
   
    #CME parameters
    parameters=[]

    apex_dist=NN.loc[NN["DATE_TIME"]==df["NN_DATE_TIME"][i], "APEX_DIST"]
    long = result['CARLON']
    lat = result['LAT']
    tilt = result['TILT']
    heith = apex_dist* plt_scl / header['RSUN']
    asp_ratio = result['ASP_RATIO']
    half_ang = result['H_ANGLE']
    parameters.append([long,lat,tilt,heith,asp_ratio,half_ang])
    satpos, plotranges = pyGCS.processHeaders([header])

    
    plot_masks(img,parameters,satpos[0], plotranges[0],imsize=imageSize, opath=opath, namefile=f'GCS_mask_{file_name}.png')

 #cmelong(carlon)+, cmelat+, cme tilt+, height+, k(asp ratio)+, half ang+