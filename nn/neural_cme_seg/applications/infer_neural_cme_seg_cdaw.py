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


__author__ = "Francisco Iglesias"
__copyright__ = "Grupo de Estudios en Heliofisica de Mendoza - https://sites.google.com/um.edu.ar/gehme"
__version__ = "0.1"
__maintainer__ = "Francisco Iglesias"
__email__ = "franciscoaiglesias@gmail.com"

def img_download():
    dfs = []
    doc_list=['19960131.065213.w047n.v0158.p272g.yht','19960126.091619.w027n.v0262.p090g.yht', '19960122.031101.w037n.v0267.p103g.yht','19960115.070110.w043n.v0525.p272s.yht','19960113.220830.w016n.v0290.p266s.yht','19960111.001436.w018n.v0499.p272s.yht']
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
            if line != '#COMMENT:':
                line = line.strip().replace('#', '')
                try:
                    nombre, valor = line.split(': ')
                except:
                    nombre, valor = line.split('=')
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

for i in range(len(catalogue.index)):
    print("Reading date range nª "+str(i))
    path='/gehme/data/soho/lasco/'
    breakpoint()
    date= pd.to_datetime(catalogue['DATE'])
    date= date.dt.strftime('%Y%m%d')
    img_path=path+catalogue['TEL'][i].lower()+'/'+date+'/'+
    # for j in range(len(files)-1):print(f'Processing {j} of {len(files)-1}')
    #         #read fits
    #         image1=read_fits(files[j],smooth_kernel=smooth_kernel)
    #         image2=read_fits(files[j+1],smooth_kernel=smooth_kernel)