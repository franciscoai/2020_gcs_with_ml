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
from scipy.ndimage.filters import gaussian_filter
import pickle


__author__ = "Francisco Iglesias"
__copyright__ = "Grupo de Estudios en Heliofisica de Mendoza - https://sites.google.com/um.edu.ar/gehme"
__version__ = "0.1"
__maintainer__ = "Francisco Iglesias"
__email__ = "franciscoaiglesias@gmail.com"

def read_fits(file_path,smooth_kernel=[0,0]):
    imageSize=[512,512]
    
    try:       
        
        file=glob.glob((file_path)[0:-5]+"*")
        img = fits.open(file[0])
        img=(img[0].data).astype("float32")
        img = gaussian_filter(img, sigma=smooth_kernel)

        if smooth_kernel[0]!=0 and smooth_kernel[1]!=0: 
            img = rebin(img, imageSize,operation='mean')
        
        return img  
    except:
        print(f'WARNING. I could not find file {file_path}')
        return None

def plot_to_png(ofile,orig_img, masks, title=None, labels=None, boxes=None, scores=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
    mask_threshold = 0.6 # value to consider a pixel belongs to the object
    scr_threshold = 0.6 # only detections with score larger than this value are considered
    color=['r','b','g','k','y']
    obj_labels = ['Occ', 'CME','N/A','N/A']
    #
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(orig_img[0]), np.nan)
    fig, axs = plt.subplots(1, len(orig_img)*2, figsize=(20, 10))
    axs = axs.ravel()
    for i in range(len(orig_img)):

        axs[i].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')
        axs[i].axis('off')
        axs[i+1].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')        
        axs[i+1].axis('off')        
        if boxes is not None:
            nb = 0
            for b in boxes[i]:
                if scores is not None:
                    scr = scores[i][nb]
                else:
                    scr = 0   
                if scr > scr_threshold:             
                    masked = nans.copy()
                    breakpoint()
                    masked[:, :][masks[i][nb] > mask_threshold] = nb              
                    axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1) # add mask
                    box =  mpl.patches.Rectangle(b[0:2],b[2]-b[0],b[3]-b[1], linewidth=2, edgecolor=color[nb], facecolor='none') # add box
                    axs[i+1].add_patch(box)
                    if labels is not None:
                        axs[i+1].annotate(obj_labels[labels[i][nb]]+':'+'{:.2f}'.format(scr),xy=b[0:2], fontsize=15, color=color[nb])
                nb+=1
    # axs[0].set_title(f'Cor A: {len(boxes[0])} objects detected') 
    # axs[1].set_title(f'Cor B: {len(boxes[1])} objects detected')               
    # axs[2].set_title(f'Lasco: {len(boxes[2])} objects detected')     
    #if title is not None:
    #    fig.suptitle('\n'.join([title[i]+' ; '+title[i+1] for i in range(0,len(title),2)]) , fontsize=16)   
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()
     
      
#main
#------------------------------------------------------------------Testing the CNN-----------------------------------------------------------------
repo_dir =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
helcat_db =  repo_dir + "/nn_training/kincat/helcatslist_20160601.txt" # kincat database
downloaded_files_list = repo_dir + '/nn_training/kincat/helcatslist_20160601_sta_downloaded.csv' # list of downloaded files
units= ["-","-","yyyymmdd","hhmm","yyyymmdd","hhmm","deg","lon","old","deg","deg","deg", "yyyymmdd","hhmm","km/s","g","km/s","LON","LAT","km/s","LON","LAT","km/s","LON","LAT"]
col_names=["HEL","CME","PRE_DATE","PRE_TIME","LAST_DATE","LAST_TIME","CARLON","STONEY","LAT","TILT", "ASP_RATIO","H_ANGLE","DATE","TIME","APEX_SPEED","CME_MASS","SPEED_FPF","FPF_LON","FPF_LAT","SPEED_SSEF","SSEF_LON","SSEF_LAT","SPEED_HMF","HMF_LON","HMF_LAT"]
imsize=[0,0]# if 0,0 no rebin its applied 
imsize_nn=[512,512] #for rebin befor the nn
smooth_kernel=[2,2] #changes the gaussian filter size

#nn model parameters
model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v3"
opath= model_path + "/infer_neural_cme_seg_kincat/cor2_a"
ipath=  "/gehme/projects/2020_gcs_with_ml/data/corona_back_database/cor2/cor2_a"
file_ext=".fits"
trained_model = '3999.torch'

#main
gpu=0 # GPU to use
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unless its not available
print(f'Using device:  {device}')

os.makedirs(opath, exist_ok=True)
#loads model
model_param = torch.load(model_path + "/"+ trained_model, map_location=device)
#vars to store all results
results = []



# read csv with paths and dates of the downloaded files
downloaded=pd.read_csv(downloaded_files_list)
downloaded['DATE_TIME'] = pd.to_datetime(downloaded['DATE_TIME'])
downloaded= downloaded.sort_values('DATE_TIME')
downloaded = downloaded.reset_index(drop=True)

# read helcat database and changes the column names
catalogue = pd.read_csv(helcat_db, sep = "\t")
catalogue=catalogue.drop([0,1])
catalogue.columns=col_names
catalogue = catalogue.reset_index(drop=True)

for i in range(len(catalogue.index)):
    print("Reading date range nª "+str(i))
    date_helcat = datetime.strptime((catalogue["PRE_DATE"][i]+" "+catalogue["PRE_TIME"][i]),'%Y-%m-%d %H:%M') #forms datetime object
    start_date = date_helcat - timedelta(hours=1)
    end_date= datetime.strptime((catalogue["LAST_DATE"][i]+" "+catalogue["LAST_TIME"][i]),'%Y-%m-%d %H:%M') #forms datetime object
    index=catalogue["CME"][i]
    files = downloaded[(downloaded["DATE_TIME"] < end_date) & (downloaded["DATE_TIME"] > start_date)]["PATH"]
    files =files.reset_index(drop=True)
    if len(files)>0:
        for j in range(len(files)-1):
            print(f'Processing {j} of {len(files)-1}')
            #read fits
            image1=read_fits(files[j],smooth_kernel=smooth_kernel)
            image2=read_fits(files[j+1],smooth_kernel=smooth_kernel)
            
            if (image1 is not None) and (image2 is not None):
                img=image2-image1
                f=files[j+1]
                # infers
                imga, maska, scra, labelsa, boxesa  = neural_cme_segmentation(model_param, img, device)
                #save results
                results.append({'file':f, 'img':imga, 'mask':maska, 'scr':scra, 'labels':labelsa, 'boxes':boxesa})
                #plot results
                filename=f[49:-4]
                os.makedirs(os.path.join(opath, str(index)),exist_ok=True)
                ofile = os.path.join(opath, str(index), filename )
                plot_to_png(ofile+".png", [imga], [maska], title=[f], labels=[labelsa], boxes=[boxesa], scores=[scra])
                
                #Saves mask parameters in pickle file
                with open((ofile+"_nn_seg"+'.pkl'), 'wb') as pickle_file:
                    # Utiliza pickle.dump() para escribir las variables en el archivo
                    pickle.dump(maska, pickle_file)
                    pickle.dump(scra, pickle_file)
                    pickle.dump(labelsa, pickle_file)
                    pickle.dump(boxesa, pickle_file)



