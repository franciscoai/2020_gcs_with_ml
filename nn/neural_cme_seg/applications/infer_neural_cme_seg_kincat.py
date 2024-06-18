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

def plot_to_png(ofile, orig_img, event, all_center, mask_threshold, scr_threshold, title=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
    
    color=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w']
    obj_labels = ['Back', 'Occ','CME','N/A']
    masks=event['MASK']
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(orig_img[0]), np.nan)
    fig, axs = plt.subplots(1, len(orig_img)*2, figsize=(20, 10))
    axs = axs.ravel()
    
    for i in range(len(orig_img)):
        axs[i].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')
        axs[i].axis('off')
        axs[i+1].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')        
        axs[i+1].axis('off') 
        for b in range(len(event['LABEL'])):
            scr = event['SCR'][b]
            if scr > scr_threshold:             
                masked = nans.copy()            
                masked[:, :][masks[b] > mask_threshold] = event['CME_ID'][b]           
                axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1) # add mask
                
                box =  mpl.patches.Rectangle(event['BOX'][b][0:2], event['BOX'][b][2]- event['BOX'][b][0], event['BOX'][b][3]- event['BOX'][b][1], linewidth=2, edgecolor=color[int(event['CME_ID'][b])] , facecolor='none') # add box
                axs[i+1].add_patch(box)
                axs[i+1].scatter(round(all_center[0][0]), round(all_center[0][1]), color='red', marker='x', s=100)
                axs[i+1].annotate(obj_labels[event['LABEL'][b]]+':'+'{:.2f}'.format(scr),xy=event['BOX'][b][0:2], fontsize=15, color=color[int(event['CME_ID'][b])])
     
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
occ_size = [50,55] # occulter radius in pixels. An artifitial occulter with constant value equal to the mean of the image is added before inference. Use 0 to avoid
occ_center=[[256,256],[260,247]]
sat="cor2_b"#"cor2_b"

if sat=="cor2_a":
    downloaded_files_list = repo_dir + '/nn_training/kincat/helcatslist_20160601_sta_downloaded.csv' # list of downloaded files
    occ_center=occ_center[0]
    occ_size= occ_size[0]
elif sat=="cor2_b":
    downloaded_files_list = repo_dir + '/nn_training/kincat/helcatslist_20160601_stb_downloaded.csv' # list of downloaded files
    occ_center=occ_center[1]
    occ_size= occ_size[1]
    avoid_dates=['2007-06-04','2007-06-07','2007-07-08','2007-08-21','2007-10-08','2007-11-04','2007-12-31','2008-02-23','2008-05-17','2008-06-01','2008-10-26','2008-12-27','2009-01-14','2009-02-11','2009-02-18','2009-04-23','2010-02-24','2010-03-30','2011-01-30','2011-03-03','2013-06-20','2013-07-18','2013-10-20','2013-10-25']


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
catalogue = catalogue[~catalogue['PRE_DATE'].isin(avoid_dates)]
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
        all_center=[]

        folder=files[0][40:-26]
        print("WORKING ON FOLDER "+folder)
        #if folder=="20080317":
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
                props_path=final_path+'mask_props'
                if not os.path.exists(final_path):
                    os.makedirs(final_path)
                if not os.path.exists(props_path):
                    os.makedirs(props_path)
                date= datetime.strptime(filename[0:-6],'%Y%m%d_%H%M%S')
                os.makedirs(os.path.join(opath, str(folder_name)),exist_ok=True)
                ofile = os.path.join(opath, str(folder_name), filename )
                
                if header['NAXIS1'] != imsize_nn[0]:
                    scale=(header['NAXIS1']/imsize_nn[0])
                    plt_scl = header['CDELT1'] * scale
                    # crpix1 = imsize_nn[0]/2+(header['NAXIS1']/2-header['CRPIX1']-(header['CRVAL1']/header['CDELT1']))/scale
                    # crpix2 = imsize_nn[0]/2+(header['NAXIS2']/2-header['CRPIX2']-(header['CRVAL2']/header['CDELT2']))/scale
                    # crpix=[crpix2,crpix1]

                else:
                    plt_scl = header['CDELT1']
                    # crpix1 = imsize_nn[0]-header['CRPIX1']
                    # crpix2 = imsize_nn[1]-header['CRPIX2']
                    # crpix=[crpix1,crpix2]

                all_center.append(occ_center)
                all_plate_scl.append(plt_scl)                    
                all_images.append(img)
                all_dates.append(date)
                all_occ_size.append(occ_size)
                file_names.append(filename)
                all_headers.append(header)
                
        if len(all_images)>=2:
            ok_orig_img,ok_dates, df =  nn_seg.infer_event2(all_images, all_dates, filter=filter, plate_scl=all_plate_scl, occulter_size=all_occ_size,centerpix=all_center,  plot_params=props_path)

            zeros = np.zeros(np.shape(ok_orig_img[0]))
            all_idx=[]
            for date in all_dates:
                if date not in ok_dates:
                    idx = all_dates.index(date)
                    all_idx.append(idx)
            file_names = [file_name for h, file_name in enumerate(file_names) if h not in all_idx]
            all_center =[all_center for h,all_center in enumerate(all_center) if h not in all_idx]
            all_plate_scl =[all_plate_scl for h,all_plate_scl in enumerate(all_plate_scl) if h not in all_idx]
            all_dates =[all_dates for h,all_dates in enumerate(all_dates) if h not in all_idx]
            all_occ_size =[all_occ_size for h,all_occ_size in enumerate(all_occ_size) if h not in all_idx]
            all_headers =[all_headers  for h,all_headers in enumerate(all_headers) if h not in all_idx]

            for m in range(len(ok_dates)):
                event = df[df['DATE_TIME'] == ok_dates[m]].reset_index(drop=True)
                image=ok_orig_img[m]
                for n in range(len(event['MASK'])):
                    if event['SCR'][n] > scr_threshold:             
                        masked = zeros.copy()
                        masked[:, :][(event['MASK'][n]) > mask_threshold] = 1
                        # safe fits
                        ofile_fits = os.path.join(os.path.dirname(ofile), file_names[m]+"_CME_ID_"+str(int(event['CME_ID'][n]))+'.fits')
                        h0 = all_headers[m]
                        # adapts hdr because we use smaller im size
                        sz_ratio = np.array(masked.shape)/np.array([h0['NAXIS1'], h0['NAXIS2']])
                        h0['NAXIS1'] = masked.shape[0]
                        h0['NAXIS2'] = masked.shape[1]
                        h0['CDELT1'] = h0['CDELT1']/sz_ratio[0]
                        h0['CDELT2'] = h0['CDELT2']/sz_ratio[1]
                        h0['CRPIX2'] = int(h0['CRPIX2']*sz_ratio[1])
                        h0['CRPIX1'] = int(h0['CRPIX1']*sz_ratio[1]) 
                        fits.writeto(ofile_fits, masked, h0, overwrite=True, output_verify='ignore')

                plot_to_png(opath+"/"+folder_name+"/"+file_names[m]+".png", [ok_orig_img[m]], event,[all_center[m]],mask_threshold=mask_threshold,scr_threshold=scr_threshold, title=[file_names[m]])  

            #Saves df for future comparison
            col_drop= ['MASK'] # its necessary to drop mask column (its a matrix) to succsessfully save the csv file
            df = df.drop(col_drop, axis=1)
            df.to_csv(final_path+folder_name+'_filtered_stats', index=False)

        else:
            print("WARNING: COULD NOT PROCESS EVENT "+ files[0][49:-13] )

                        



