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
        print(f'WARNING. could not find file {file_path}')
        return None

def plot_to_png(ofile,orig_img, masks,all_centers,all_centerpix,mask_threshold,scr_threshold, title=None, labels=None, boxes=None, scores=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
    
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
                    axs[i+1].scatter(256, 256, color='red', marker='x', s=100)
                    
                    axs[i+1].scatter(round(all_centers[0][0]), round(all_centers[0][1]), color='blue', marker='x', s=100)
                    axs[i+1].scatter(round(all_centerpix[0][0]), round(all_centerpix[0][1]), color='green', marker='x', s=100)

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
units= ["-","-","yyyymmdd","hhmm","yyyymmdd","hhmm","deg","lon","old","deg","deg","deg", "yyyymmdd","hhmm","km/s","g","km/s","LON","LAT","km/s","LON","LAT","km/s","LON","LAT"]
col_names=["HEL","CME","PRE_DATE","PRE_TIME","LAST_DATE","LAST_TIME","CARLON","STONEY","LAT","TILT", "ASP_RATIO","H_ANGLE","DATE","TIME","APEX_SPEED","CME_MASS","SPEED_FPF","FPF_LON","FPF_LAT","SPEED_SSEF","SSEF_LON","SSEF_LAT","SPEED_HMF","HMF_LON","HMF_LAT"]
kincat_col_names=["DATE_TIME","MASK","SCORE","CPA_ANG","WIDE_ANG","APEX_DIST"]
imsize=[0,0]# if 0,0 no rebin its applied 
imsize_nn=[512,512] #for rebin befor the nn
smooth_kernel=[2,2] #changes the gaussian filter size
occ_size = 50 # occulter radius in pixels. An artifitial occulter with constant value equal to the mean of the image is added before inference. Use 0 to avoid
sat="cor2_b"#"cor2_a"

if sat=="cor2_a":
    downloaded_files_list = repo_dir + '/nn_training/kincat/helcatslist_20160601_sta_downloaded.csv' # list of downloaded files
elif sat=="cor2_b":
    downloaded_files_list = repo_dir + '/nn_training/kincat/helcatslist_20160601_stb_downloaded.csv' # list of downloaded files


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
        all_headers=[]
        all_centers=[]
        all_centerpix=[]

        folder=files[0][40:-26]
        print("WORKING ON FOLDER "+folder)
        if folder=="20070509":
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
                    
                    if header['NAXIS1'] != imsize_nn[0]:
                        plt_scl = header['CDELT1'] * header['NAXIS1']/imsize_nn[0] 
                        crpix1 = imsize_nn[0]-(header['CRPIX1'] / (header['NAXIS1']/imsize_nn[0]))
                        crpix2 = imsize_nn[1]-(header['CRPIX2'] / (header['NAXIS1']/imsize_nn[0]))
                        crpix=[crpix1,crpix2]

                    else:
                        plt_scl = header['CDELT1']
                        crpix1 = imsize_nn[0]-header['CRPIX1']
                        crpix2 = imsize_nn[1]-header['CRPIX2']
                        crpix=[crpix1,crpix2]

                    x_cen=(imsize_nn[0]/2)+(header['XCEN']/plt_scl)
                    y_cen=(imsize_nn[1]/2)+(header['YCEN']/plt_scl)
                    center=[x_cen,y_cen]
                    all_centerpix.append(crpix)
                    all_plate_scl.append(plt_scl)
                    all_centers.append(center)
                    all_images.append(img)
                    all_dates.append(date)
                    all_occ_size.append(0)
                    file_names.append(filename)
                    all_headers.append(header)
            
            if len(all_images)>=2:
                all_orig_img, ok_dates, all_masks, all_scores, all_lbl, all_boxes, all_mask_prop =  nn_seg.infer_event(all_images, all_dates, filter=filter, plate_scl=all_plate_scl, occulter_size=all_occ_size,centerpix=all_centers,  plot_params=final_path+'mask_props')
                
                if all_masks is not None:
                    zeros = np.zeros(np.shape(all_orig_img[0]))
                    for i in range(len(all_orig_img)):
                        scr = 0
                        if all_scores is not None:
                            if all_scores[i] is not None:
                                scr = all_scores[i]
                        if scr > scr_threshold:             
                            masked = zeros.copy()
                            masked[:, :][all_masks[i] > mask_threshold] = 1
                            # safe fits
                            ofile_fits = os.path.join(os.path.dirname(ofile), file_names[i]+'.fits')
                            h0 = all_headers[i]

                            # adapts hdr because we use smaller im size
                            sz_ratio = np.array(masked.shape)/np.array([h0['NAXIS1'], h0['NAXIS2']])
                            h0['NAXIS1'] = masked.shape[0]
                            h0['NAXIS2'] = masked.shape[1]
                            h0['CDELT1'] = h0['CDELT1']/sz_ratio[0]
                            h0['CDELT2'] = h0['CDELT2']/sz_ratio[1]
                            h0['CRPIX2'] = int(h0['CRPIX2']*sz_ratio[1])
                            h0['CRPIX1'] = int(h0['CRPIX1']*sz_ratio[1]) 
                            fits.writeto(ofile_fits, masked, h0, overwrite=True, output_verify='ignore')

                    if len(all_masks)>0:
                        for i in range(len(all_images)):
                            plot_to_png(opath+"/"+folder_name+"/"+file_names[i]+".png", [all_orig_img[i]], [all_masks[i]],[all_centers[i]],[all_centerpix[i]],mask_threshold=mask_threshold,scr_threshold=scr_threshold, title=[file_names[i]], labels=[all_lbl[i]], boxes=[all_boxes[i]], scores=[all_scores[i]])
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

            else:
                print("WARNING: COULD NOT PROCESS EVENT "+ files[0][49:-13] )

                            



