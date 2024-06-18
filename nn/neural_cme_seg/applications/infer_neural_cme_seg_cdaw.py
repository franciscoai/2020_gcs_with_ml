import os
import sys
from astropy.io import fits
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from ext_libs.rebin import rebin
from nn.neural_cme_seg.neural_cme_seg import neural_cme_segmentation
import csv
import pandas as pd
from datetime import datetime, timedelta
import glob
from scipy.ndimage import gaussian_filter
import pickle
from tqdm import tqdm

__author__ = "Francisco Iglesias"
__copyright__ = "Grupo de Estudios en Heliofisica de Mendoza - https://sites.google.com/um.edu.ar/gehme"
__version__ = "0.1"
__maintainer__ = "Francisco Iglesias"
__email__ = "franciscoaiglesias@gmail.com"

def img_download():
    dfs = []
    doc_list=['20080503.185404.w018n.v0267.p245g.yht','20080503.103358.w007n.v0205.p198g.yht', '20080503.043004.w015n.v0346.p234g.yht','20080502.193138.w028n.v0191.p085g.yht','20080502.100604.w047n.v0176.p235g.yht','20080502.063004.w021n.v0124.p291g.yht','20080501.193137.w049n.v0130.p089g.yht','20080501.163005.w018n.v0178.p091g.yht','20080501.061804.w024n.v0369.p094g.yht']
    for h in doc_list:    
        df = pd.read_csv('/home/cisterna/Downloads/'+h, header=None, names=['COL'])
        col_list = df['COL'].tolist()
        col_names=['HEIGHT','DATE','TIME','ANGLE', 'TEL','FC','COL','ROW']
        for i in range(len(col_list)):
            line=col_list[i]
            if line=='# HEIGHT   DATE     TIME   ANGLE  TEL  FC    COL     ROW':
                dismiss_rows=i-1
                events = col_list[i+1:]
                data_split = [event.split() for event in events]
                df_fixed = pd.DataFrame(data_split)
                df_fixed.columns=col_names

        for j in range(0, dismiss_rows):
            line=col_list[j]
            line = line.strip().replace('#', '')
            try:
                nombre, valor = line.split(': ')
            except:
                try:
                    nombre, valor = line.split('=')
                except:
                    line = line.strip().replace(':', '')
                    nombre == line
                    valor == '-'
            df_fixed[nombre]=valor
        dfs.append(df_fixed)
    df_full = pd.concat(dfs, ignore_index=True)
    df_full['DATE_TIME'] = pd.to_datetime(df_full['DATE'] + ' ' + df_full['TIME'])
    return df_full

    

#---------------------------------------------------------- MAIN ----------------------------------------------------------
repo_dir =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

level = 'L1' # fits reduction level
filter_pol_images = True
units= ["-","-","yyyymmdd","hhmm","yyyymmdd","hhmm","deg","lon","old","deg","deg","deg", "yyyymmdd","hhmm","km/s","g","km/s","LON","LAT","km/s","LON","LAT","km/s","LON","LAT"]

imsize=[0,0]# if 0,0 no rebin its applied
imsize_nn=[512,512] #for rebin befor the nn
smooth_kernel=[2,2] #changes the gaussian filter size
occ_size = [50,55] # occulter radius in pixels. An artifitial occulter with constant value equal to the mean of the image is added before inference. Use 0 to avoid
occ_center=[[256,256],[260,247]]
sat='lasco'


#nn model parameters
model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4"
model_version="v4"
opath= model_path + "/infer_neural_cme_seg_kincat_"+level+"/"+sat
file_ext=".fits"
trained_model = '6000.torch'
mask_threshold = 0.6 # value to consider a pixel belongs to the object
scr_threshold = 0.25 # only detections with score larger than this value are considered


#main
gpu=0 # GPU to use
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unless its not available
print(f'Using device:  {device}')
os.makedirs(opath, exist_ok=True)
results = []

#loads nn model
nn_seg = neural_cme_segmentation(device, pre_trained_model = model_path + "/"+ trained_model, version=model_version)

catalogue=img_download()
breakpoint()
for i in range(len(catalogue.index)):
    print("Reading date range nª "+str(i))
    # path='/gehme/data/soho/lasco/'
    # breakpoint()
    # date= pd.to_datetime(catalogue['DATE'])
    # date= date.dt.strftime('%Y%m%d')
    # img_path=path+catalogue['TEL'][i].lower()+'/'+date+'/'
    
    
    # Create an empty DataFrame to store the paths and dates
    lasco_df = pd.DataFrame(columns=['paths',"date"])
    # Create the output directory
    odir = opath + "/c2"
    os.makedirs(odir, exist_ok=True)

    # Iterate over the paths and extract the date from the headers
    for i in tqdm(paths["paths"], desc="Preparing lasco data"):
        basename = os.path.basename(i)
        header = fits.getheader(i)
        date_obs = header["DATE-OBS"]
        time_obs = header["TIME-OBS"]
        datetime_obs = datetime.strptime(date_obs + ' ' + time_obs, '%Y/%m/%d %H:%M:%S.%f')
        lasco_df.loc[len(lasco_df.index)] = [i, datetime_obs]
    
    # Process the lasco data
    amount_counter = 0
    for i, date in tqdm(enumerate(lasco_df["date"]), desc="Processing lasco data"):
        # try:
        prev_date = date - timedelta(hours=12)
        # Delete duplicated rows
        lasco_df = lasco_df.drop_duplicates()
        count = ((lasco_df["date"] < date) & (lasco_df["date"] >= prev_date)).sum()
        if count==1:
            # Generate the diff image
            # file1 = glob.glob(lasco_df.loc[i, "paths"][0:-10] + "*")
            # file2 = glob.glob(lasco_df.loc[i + 1, "paths"][0:-10] + "*")
            file1 = [lasco_df.loc[i, "paths"]]
            file2 = [lasco_df.loc[i + 1, "paths"]]

            if len(file1)!=0 or len(file2)!=0:
                # img1= fits.open((file1[0]))[0].data
                # img2= fits.open((file2[1]))[0].data
                img1 = fits.open(file1[0])[0].data
                img2 = fits.open(file2[0])[0].data
                header= fits.getheader((file1[0]))

                # Check shape
                if img1.shape != (1024, 1024) or img2.shape != (1024, 1024):
                    continue    

                # Resize images if necessary
                if imsize[0]!=0 and imsize[1]!=0:
                    img1 = rebin(img1,imsize_nn,operation='mean') 
                    img2 = rebin(img2.data,imsize_nn,operation='mean')
                    header['NAXIS1'] = imsize_nn[0]   
                    header['NAXIS2'] = imsize_nn[1]
                
                # Calculate the difference image
                img_diff = img1 - img2
                img_diff = fits.PrimaryHDU(img_diff, header=header[0:-3])
                
                # Write the difference image
                if do_write==True:
                    imgs, masks, scrs, labels, boxes  = nn_seg.infer(img_diff.data, model_param=None, resize=False, occulter_size=0)
                    scrs = [scrs[i] for i in range(len(labels)) if labels[i] == 2]
                    scrs = np.concatenate([scrs])
                    if np.all(scrs < SCR_THRESHOLD):
                        namefile = f"{date.strftime('%Y%m%d_%H%M%S.%f')}.fits"
                        # Check if the namefile is in the drop_cases_lasco list
                        if namefile in drop_cases_lasco:
                            continue
                        amount_counter += 1
                        if write_png==True:
                            mu = np.mean(img_diff.data)
                            sd = np.std(img_diff.data)
                            plt.imsave(odir+"/"+namefile+".png", img_diff.data, cmap='gray', vmin=mu-3*sd, vmax=mu+3*sd)
                        else:
                            img_diff.writeto(odir+"/"+namefile,overwrite=True)
            else:
                continue
        if amout_limit is not None:
            if amount_counter >= amout_limit:
                break
# for j in range(len(files)-1):print(f'Processing {j} of {len(files)-1}')
#         #read fits
#         image1=read_fits(files[j],smooth_kernel=smooth_kernel)
#         image2=read_fits(files[j+1],smooth_kernel=smooth_kernel)