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
    mask_threshold = 0.5 # value to consider a pixel belongs to the object
    scr_threshold = 0.3 # only detections with score larger than this value are considered
    color=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w']
    obj_labels = ['Back', 'Occ','CME','N/A']
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
        if boxes[i] is not None:
            nb = 2
            for b in boxes[i]:
                if scores[i] is not None:
                    scr = scores[i]
                else:
                    scr = 0   
                if scr > scr_threshold:             
                    masked = nans.copy()
                    masked[:, :][masks[i] > mask_threshold] = nb              
                    axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1) # add mask
                    box =  mpl.patches.Rectangle(boxes[i][0:2],boxes[i][2]-boxes[i][0],boxes[i][3]-boxes[i][1], linewidth=2, edgecolor=color[nb], facecolor='none') # add box
                    axs[i+1].add_patch(box)
                    if labels[i] is not None: 
                        
                        axs[i+1].annotate(obj_labels[labels[i]]+':'+'{:.2f}'.format(scr),xy=boxes[i][0:2], fontsize=15, color=color[nb])
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
level = 'L1' # fits reduction level
filter_pol_images = True
downloaded_files_list = repo_dir + '/nn_training/kincat/helcatslist_20160601_sta_downloaded.csv' # list of downloaded files
units= ["-","-","yyyymmdd","hhmm","yyyymmdd","hhmm","deg","lon","old","deg","deg","deg", "yyyymmdd","hhmm","km/s","g","km/s","LON","LAT","km/s","LON","LAT","km/s","LON","LAT"]
col_names=["HEL","CME","PRE_DATE","PRE_TIME","LAST_DATE","LAST_TIME","CARLON","STONEY","LAT","TILT", "ASP_RATIO","H_ANGLE","DATE","TIME","APEX_SPEED","CME_MASS","SPEED_FPF","FPF_LON","FPF_LAT","SPEED_SSEF","SSEF_LON","SSEF_LAT","SPEED_HMF","HMF_LON","HMF_LAT"]
kincat_col_names=["DATE_TIME","MASK","SCORE","CPA_ANG","WIDE_ANG","APEX_DIST"]
imsize=[0,0]# if 0,0 no rebin its applied 
imsize_nn=[512,512] #for rebin befor the nn
smooth_kernel=[2,2] #changes the gaussian filter size
occ_size = 50 # occulter radius in pixels. An artifitial occulter with constant value equal to the mean of the image is added before inference. Use 0 to avoid

#nn model parameters
model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4"
model_version="v4"
opath= model_path + "/infer_neural_cme_seg_kincat_"+level+"/cor2_a"
ipath=  "/gehme/projects/2020_gcs_with_ml/data/corona_back_database/cor2/cor2_a"
file_ext=".fits"
trained_model = '6000.torch'

#main
gpu=0 # GPU to use
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unless its not available
print(f'Using device:  {device}')
os.makedirs(opath, exist_ok=True)
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

#loads nn model
nn_seg = neural_cme_segmentation(device, pre_trained_model = model_path + "/"+ trained_model, version=model_version)

for i in range(len(catalogue.index)):
    print("Reading date range nª "+str(i))
    date_helcat = datetime.strptime((catalogue["PRE_DATE"][i]+" "+catalogue["PRE_TIME"][i]),'%Y-%m-%d %H:%M') #forms datetime object
    start_date = date_helcat 
    end_date= datetime.strptime((catalogue["LAST_DATE"][i]+" "+catalogue["LAST_TIME"][i]),'%Y-%m-%d %H:%M') #forms datetime object
    #index=catalogue["CME"][i]
    
    if filter_pol_images:
        # if downloaded["DATE_TIME"] is at 08 minutes drop it
        downloaded = downloaded[~(downloaded["DATE_TIME"].dt.minute == 8)]        
    files = downloaded[(downloaded["DATE_TIME"] < end_date) & (downloaded["DATE_TIME"] > start_date)]["PATH"]
    files =files.reset_index(drop=True)
    if level == 'L1':
        files = [f.replace('L0','L1') for f in files]
        files = [f.replace('_d4c','_14c') for f in files]
    if len(files)>0:
        all_images=[]
        all_dates=[]
        all_occ_size=[]
        all_plate_scl=[]
        file_names=[]
        for j in range(len(files)-1):

            print(f'Processing {j} of {len(files)-1}')
            #read fits
            image1=read_fits(files[j],smooth_kernel=smooth_kernel)
            image2=read_fits(files[j+1],smooth_kernel=smooth_kernel)
            
            if (image1 is not None) and (image2 is not None):
                img=image2-image1
                file=glob.glob((files[j+1])[0:-5]+"*")
                header=fits.getheader(file[0])
                f=files[j+1]
                filename=f[49:-4]
                folder_name=files[0][49:-17]
                final_path=opath+"/"+folder_name+"/filtered/"
                date= datetime.strptime(filename[0:-6],'%Y%m%d_%H%M%S')
                os.makedirs(os.path.join(opath, str(folder_name)),exist_ok=True)
                ofile = os.path.join(opath, str(folder_name), filename )
                
                hdu = fits.PrimaryHDU(img, header=header)
                hdu.writeto(ofile+".fits", overwrite=True)
                if header['NAXIS1'] != imsize_nn[0]:
                    plt_scl = header['CDELT1'] * header['NAXIS1']/imsize_nn[0] 
                else:
                    plt_scl = header['CDELT1']
                all_plate_scl.append(plt_scl)
                all_images.append(img)
                all_dates.append(date)
                all_occ_size.append(occ_size)
                file_names.append(filename)
             
        all_orig_img, ok_dates, all_masks, all_scores, all_lbl, all_boxes, all_mask_prop =  nn_seg.infer_event(all_images, all_dates, filter=filter, plate_scl=all_plate_scl, occulter_size=all_occ_size,  plot_params=final_path+'mask_props')
        
        if len(all_masks)>0:
            for i in range(len(all_images)):

                plot_to_png(opath+"/"+folder_name+"/"+file_names[i]+".png", [all_orig_img[i]], [all_masks[i]], title=[file_names[i]], labels=[all_lbl[i]], boxes=[all_boxes[i]], scores=[all_scores[i]])
        
        else:
            print("No CME detected :-/")        

        data_kincat=[]
        for i in range(len(all_mask_prop)):
            if all(elemento is not None for elemento in all_mask_prop[i]):
                prop_list = all_mask_prop[i].tolist()
                prop_list.insert(0,ok_dates[i])
                data_kincat.append(prop_list)

        df = pd.DataFrame(data_kincat, columns=kincat_col_names)
        
        df.to_csv(final_path+folder_name+'_filtered_stats', index=False)

                       


