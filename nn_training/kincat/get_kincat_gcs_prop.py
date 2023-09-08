
import os
import sys
import glob
import pandas as pd
from astropy.io import fits
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from ext_libs.rebin import rebin

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
    return df

def filter_param(NN, kincat):
    kincat_cols=["PRE_DATE","PRE_TIME","END_DATE","END_TIME"]
    columns=["NN_DATE_TIME","HEL","CME","PRE_DATE","PRE_TIME","LAST_DATE","LAST_TIME","CARLON","STONEY","LAT","TILT", "ASP_RATIO","H_ANGLE","DATE","TIME","APEX_SPEED","CME_MASS","SPEED_FPF","FPF_LON","FPF_LAT","SPEED_SSEF","SSEF_LON","SSEF_LAT","SPEED_HMF","HMF_LON","HMF_LAT"]


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



###################################################################################### MAIN #########################################################################################
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
kincat_orig_dir=repo_dir+"/nn_training/kincat/helcatslist_20160601.txt"
odir="/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat_L1/"
sat="cor2_a"#cor2_b
imageSize=[512,512]

NN=get_NN(odir,sat)
kincat_orig= get_kincat(kincat_orig_dir,sat)
df=filter_param(NN,kincat_orig)

for i in range(len(df)):
    
    folder= str(df["NN_DATE_TIME"][i])[:-9].replace("-", "")
    file_name=folder+"_"+str(df["NN_DATE_TIME"][i])[11:].replace(":", "")
    path=(glob.glob(odir+sat+"/"+folder+"/"+file_name+"*"+".fits"))[0]
    imga, ha = read_fits(path, header=True,imageSize=imageSize)
    if ha['NAXIS1'] != imageSize[0]:
        plt_scl = ha['CDELT1'] * ha['NAXIS1']/imageSize[0] 
    else:
        plt_scl = ha['CDELT1']
        
    apex_sr = NN["APEX_DIS"]* plt_scl / ha['RSUN']

    