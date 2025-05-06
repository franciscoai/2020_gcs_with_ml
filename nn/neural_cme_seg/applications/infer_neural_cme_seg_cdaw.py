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



def read_fits(file_path, header=False, imageSize=[512,512],smooth_kernel=[0,0]):
    try:       
        img = fits.open(file_path)[0].data
        img=img.astype("float32")
        img = gaussian_filter(img, sigma=smooth_kernel)
        if imageSize:
            img = rebin(img, imageSize, operation='mean')  
            #img = rebin_interp(img, new_size=[512,512])#si la imagen no es cuadrada usar esto
            if header:
                hdr = fits.open(file_path)[0].header
                naxis_original1 = hdr["naxis1"]
                naxis_original2 = hdr["naxis2"]
                hdr["naxis1"]=imageSize[0]
                hdr["naxis2"]=imageSize[1]
                hdr["crpix1"]=hdr["crpix1"]/(naxis_original1/imageSize[0])
                hdr["crpix2"]=hdr["crpix2"]/(naxis_original2/imageSize[1])
                hdr["CDELT1"]=hdr["CDELT1"]*(naxis_original1/imageSize[0])
                hdr["CDELT2"]=hdr["CDELT2"]*(naxis_original2/imageSize[1])
               
        # flips y axis to match the orientation of the images
        #img = np.flip(img, axis=0) 
        if header:
            return img, hdr
        else:
            return img 
    except:
        print(f'WARNING. I could not find file {file_path}')
        return None
                            


def get_cdaw(repo,path,opath):
    '''
    Gets cdaw full catalogue(1996-2023) and finds the lasco images corresponding to that dates. Returns two dataframes, "catalogue" who is the full cdaw_catalogue and  "found_events" contains the lasco events present in cdaw catalogue. 
    repo: all yht files of cdaw
    path:lasco level 0.5 image repository
    opath: output path for the found  lasco events present in cdaw catalogue
    '''
    repo_dir= repo+"cdaw/"
    if os.path.exists(opath+'found_events_cdaw.csv'):
        found_events = pd.read_csv(opath+'found_events_cdaw.csv')
        catalogue = pd.read_csv(repo+"full_cdaw_catalogue.csv", low_memory=False)

    else:
        if os.path.exists(repo+"full_cdaw_catalogue.csv"):
            catalogue = pd.read_csv(repo+"full_cdaw_catalogue.csv")
        else: 
            
            dfs = []
            repo_files=os.listdir(repo_dir)
            for h in tqdm(repo_files): 
                df = pd.read_csv(repo_dir+h, header=None, names=['COL'], delimiter='\t')
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
                df_fixed["FOLDER_NAME"]=h
                dfs.append(df_fixed)
            
            catalogue = pd.concat(dfs, ignore_index=True)
            catalogue['DATE_TIME'] = pd.to_datetime(catalogue['DATE'] + ' ' + catalogue['TIME'])
            catalogue.to_csv(repo+'full_cdaw_catalogue.csv', index=False)

        
        # Create an empty DataFrame to store the paths and dates
        found_events = pd.DataFrame(columns=['PATH',"DATE_TIME"])
        date= pd.to_datetime(catalogue['DATE'])
        date= date.dt.strftime('%Y%m%d')
        date=date.unique()
        
        for k in tqdm(range(len(catalogue['DATE'].unique()))):
            
            #if date[k] == "20220224":#"20120422"
               
            folder_path=path+catalogue['TEL'][k].lower()+'/'+date[k]+'/'
            if os.path.exists(folder_path):
                    
                try:
                    files = os.listdir(folder_path)
                except Exception as error:
                    print("Error " + str(error))
                    files=[]
                # Iterate over the paths and extract the date from the headers
                for l in tqdm(files, desc="Preparing lasco data"):
                    
                    if l.endswith(".fts"):
                        
                        file_path=folder_path+l
                        
                        basename = os.path.basename(file_path)
                        header = fits.getheader(file_path)
                        date_obs = header["DATE-OBS"]
                        time_obs = header["TIME-OBS"]
                        if len(time_obs)==0:
                            time_obs = date_obs.split("T")[1]
                            date_obs = date_obs.split("T")[0]
                            
                        try:    
                            datetime_obs = datetime.strptime(date_obs + ' ' + time_obs, '%Y/%m/%d %H:%M:%S.%f')
                            
                        except:
                            try:
                                datetime_obs = datetime.strptime(date_obs + ' ' + time_obs, '%Y-%m-%d %H:%M:%S.%f')
                                datetime_obs.strftime('%Y/%m/%d %H:%M:%S.%f')
                            except:
                                breakpoint()
                                print("Empty or Corrupted .fts file " +file_path)

                        found_events.loc[len(found_events.index)] = [file_path, datetime_obs]
    
        found_events= found_events.drop_duplicates()    
        found_events.to_csv(opath+'found_events_cdaw.csv', index=False)
    
    return found_events , catalogue

def plot_to_png(ofile, orig_img, event, control_df, all_center, mask_threshold, scr_threshold,  title=None):
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
    
    # for i in range(len(orig_img)):
    #     axs[i].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')
    #     axs[i].axis('off')
    #     axs[i+1].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')        
    #     axs[i+1].axis('off') 
    #     for b in range(len(event['LABEL'])):
    #         scr = event['SCR'][b]
    #         if scr > scr_threshold:             
    #             masked = nans.copy()            
    #             masked[:, :][masks[b] > mask_threshold] = event['CME_ID'][b]           
    #             axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1) # add mask
                
    #             box =  mpl.patches.Rectangle(event['BOX'][b][0:2], event['BOX'][b][2]- event['BOX'][b][0], event['BOX'][b][3]- event['BOX'][b][1], linewidth=2, edgecolor=color[int(event['CME_ID'][b])] , facecolor='none') # add box
    #             axs[i+1].add_patch(box)
    #             axs[i+1].scatter(round(all_center[0][0]), round(all_center[0][1]), color='red', marker='x', s=100)
    #             axs[i+1].annotate(obj_labels[event['LABEL'][b]]+':'+'{:.2f}'.format(scr),xy=event['BOX'][b][0:2], fontsize=15, color=color[int(event['CME_ID'][b])])
    
    for i in range(len(orig_img)):
        axs[i].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')
        axs[i].axis('off')
        axs[i+1].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')        
        axs[i+1].axis('off') 
        event_datetime = event['DATE_TIME'].iloc[0] 
        masks_for_date = control_df[control_df['DATE_TIME'] == event_datetime]

    for _, row in masks_for_date.iterrows():
        if row['SCR'] > scr_threshold:
            masked = nans.copy()
            masked[:, :][row['MASK'] > mask_threshold] = row['CME_ID']
            color_idx = int(row['CME_ID'])

            axs[i].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1)  # Agregar máscaras en axs[i]

    # Todo lo demás sigue en axs[i+1] sin modificaciones
    for b in range(len(event['LABEL'])):
        scr = event['SCR'][b]
        if scr > scr_threshold:             
            masked = nans.copy()            
            masked[:, :][masks[b] > mask_threshold] = event['CME_ID'][b]           
            axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1)  # add mask
            
            box = mpl.patches.Rectangle(event['BOX'][b][0:2], event['BOX'][b][2]- event['BOX'][b][0], 
                                        event['BOX'][b][3]- event['BOX'][b][1], linewidth=2, 
                                        edgecolor=color[int(event['CME_ID'][b])], facecolor='none')  # add box
            axs[i+1].add_patch(box)
            axs[i+1].scatter(round(all_center[0][0]), round(all_center[0][1]), color='red', marker='x', s=100)
            axs[i+1].annotate(obj_labels[event['LABEL'][b]]+':'+'{:.2f}'.format(scr),
                            xy=event['BOX'][b][0:2], fontsize=15, color=color[int(event['CME_ID'][b])])


        
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

#---------------------------------------------------------- MAIN ----------------------------------------------------------
repo_dir =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
cdaw_repo='/gehme/data/catalogues/soho/lasco/'
lasco_path='/gehme/data/soho/lasco/level_1/'
cdaw_catalogue='/gehme-gpu/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn/neural_cme_seg/applications/cdaw_catalogue/'

level = 'L1' # fits reduction level
sat='lasco'
filter_pol_images = True
exclude_mult=False
exclude_poor_events=True
quality_idx=2# if exclude_poor_events=True it filtrates every event with a quality index lower to quality_idx. Posible values: (0-5)
units= ["-","-","yyyymmdd","hhmm","yyyymmdd","hhmm","deg","lon","old","deg","deg","deg", "yyyymmdd","hhmm","km/s","g","km/s","LON","LAT","km/s","LON","LAT","km/s","LON","LAT"]

imsize=[0,0]# if 0,0 no rebin its applied
imsize_nn=[512,512] #for rebin befor the nn
smooth_kernel=[2,2] #changes the gaussian filter size
occ_size = [90] # occulter radius in pixels. An artifitial occulter with constant value equal to the mean of the image is added before inference. Use 0 to avoid
occ_center=[[256,256]]
occulter_size_ext = []

smooth_kernel=[2,2] #changes the gaussian filter size
MAX_TIME_DIFF= timedelta(minutes=15)
MAX_TIME_DIFF_IMAGE_L1=timedelta(minutes=60)
plot_png=True #Saves the png files with the selected mask vs all masks
w_metric=[0.4,0.4,0.1,0.1] #weights for the metric used to select the best mask [IOU,CPA,AW,APEX]

#nn model parameters
model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_A6_DS32"#"/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/"
model_version="A6"#"v4"
opath= model_path + "/infer_neural_cme_seg_kincat_"+level+"/"+sat +"_flor_test"
file_ext=".fits"
trained_model = "4.torch"#'9999.torch'
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

# Create the output directory
odir = opath + "/c2"
os.makedirs(odir, exist_ok=True)

#gets catalogues
found_events, cdaw_full  = get_cdaw(cdaw_repo,lasco_path, cdaw_catalogue)
if exclude_poor_events:
    cdaw_full['QUALITY_INDEX'] = cdaw_full['QUALITY_INDEX'].str.extract(r'(\d+)').astype(int)
    cdaw_full=cdaw_full.loc[cdaw_full['QUALITY_INDEX']>quality_idx]
    
if exclude_mult:
    unique_dates=cdaw_full["DATE"].unique()
    for j in range(len(unique_dates)):
        date_event=cdaw_full.loc[cdaw_full["DATE"]==unique_dates[j]]
        orig_files=date_event['FOLDER_NAME'].unique()
        for g in range(len(orig_files)-1):
            first_event=cdaw_full.loc[cdaw_full['FOLDER_NAME']==orig_files[g]]
            second_event=cdaw_full.loc[cdaw_full['FOLDER_NAME']==orig_files[g+1]]
            first_event['DATE_TIME'] = pd.to_datetime(first_event['DATE_TIME'])
            second_event['DATE_TIME'] = pd.to_datetime(second_event['DATE_TIME'])
            first_interval = pd.Interval(first_event["DATE_TIME"].min(),first_event["DATE_TIME"].max() , closed='both')
            second_interval = pd.Interval(second_event["DATE_TIME"].min(), second_event["DATE_TIME"].max(), closed='both')
            if first_interval.overlaps(second_interval):
                cdaw_full = cdaw_full.loc[~cdaw_full['FOLDER_NAME'].str.contains(orig_files[g], case=False, na=False)]
                cdaw_full = cdaw_full.loc[~cdaw_full['FOLDER_NAME'].str.contains(orig_files[g+1], case=False, na=False)]
                
            



    

#transforms to datetime objects and sorts the catalogues
found_events["DATE_TIME"] = pd.to_datetime(found_events["DATE_TIME"])
found_events["DATE_TIME"] = found_events["DATE_TIME"].dt.strftime('%Y-%m-%d %H:%M:%S')
found_events= found_events.drop_duplicates() 
found_events = found_events.sort_values(by='DATE_TIME')
cdaw_full["DATE_TIME"] =pd.to_datetime(cdaw_full["DATE_TIME"])
cdaw_full= cdaw_full.sort_values(by='DATE_TIME')
yht_files=cdaw_full["FOLDER_NAME"].unique()
#cdaw_unique_values = cdaw_full[cdaw_full["FOLDER_NAME"].isin(yht_files)].drop_duplicates(subset="FOLDER_NAME", keep="first")

# Process the lasco data
for i in tqdm(range(len(yht_files)), desc="Processing lasco data"):
    #if yht_files[i]=="20021110.083005.w068n.v0290.p097s.yht":
    all_images=[]
    all_dates=[]
    all_occ_size=[]
    all_plate_scl=[]
    file_names=[]
    all_headers=[]
    all_center=[]
    all_ofiles=[]
    #separates cdaw catalogue in cdaw_events (to be part of the cdaw_event the files must have same date and be from the same yht file)
    cdaw_event=cdaw_full.loc[cdaw_full["FOLDER_NAME"]==yht_files[i]]
    cdaw_event=cdaw_event.loc[cdaw_event["TEL"]=="C2"]
    cdaw_event=cdaw_event.reset_index(drop=True)

    if len(cdaw_event)>0:
        date=cdaw_event["DATE_TIME"].iloc[0]
            
        prev_date = date - MAX_TIME_DIFF

        #finds all files present in the folder for the corresponding cdaw_event
        event=found_events.loc[(found_events["DATE_TIME"]>=str(prev_date))&(found_events["DATE_TIME"]<=str(cdaw_event["DATE_TIME"].iloc[-1]))]
        event["DATE_TIME"]=pd.to_datetime(event["DATE_TIME"])
        event = event.reset_index(drop=True)
        
        #finds the cdaw_event files present in the event
        #specific_event=pd.to_datetime("2003-12-08")
        #if date.date()==specific_event.date():     
            
        if len(event)>2:

            
            #coincidences = event[event["DATE_TIME"].isin(cdaw_event["DATE_TIME"])]
            #coincidences = coincidences.reset_index(drop=True)
            for j in range(len(event)-1):#for j in range(len(coincidences)-1):
                file1 = [event["PATH"].iloc[j]]#file1 = [coincidences["PATH"].iloc[j]]
                prev_time=event["DATE_TIME"].iloc[j]-MAX_TIME_DIFF_IMAGE_L1#prev_time=coincidences["DATE_TIME"].iloc[j]-MAX_TIME_DIFF_IMAGE_L1
                prev_images=event.loc[(event["DATE_TIME"]<event["DATE_TIME"].iloc[j])&(event["DATE_TIME"]>prev_time)]
                #prev_images=event.loc[(event["DATE_TIME"]<coincidences["DATE_TIME"].iloc[j])&(event["DATE_TIME"]>prev_time)]
                prev_images = prev_images.reset_index(drop=True)
                
                if len(prev_images)>0:
                    file2 = [prev_images["PATH"].iloc[-1]]
                    
                    if len(file1)!=0 or len(file2)!=0:
                        image1, header1 = read_fits(file1[0],header=True)# final image
                        image2,header2 = read_fits(file2[0],header=True)# initial image
                        if (image1 is not None) and (image2 is not None):
                            # # Check shape
                            # if header1['NAXIS1'] != imsize_nn[0]:
                            #     scale=(header1['NAXIS1']/imsize_nn[0])
                            #     plt_scl = header1['CDELT1'] * scale
                            # else:
                            #     plt_scl = header1['CDELT1']
                            # # Resize images if necessary
                            # if (imsize[0]!=0 and imsize[1]!=0) or (image1.shape !=image2.shape):
                            #     image1 = rebin(image1,imsize_nn,operation='mean') 
                            #     image2 = rebin(image2,imsize_nn,operation='mean')
                            #     header1['NAXIS1'] = imsize_nn[0]   
                            #     header1['NAXIS2'] = imsize_nn[1]
                            
                            plt_scl = header1['CDELT1']
                            im_date=event["DATE_TIME"].iloc[j]#im_date=coincidences["DATE_TIME"].iloc[j]
                            # Gets diff image
                            img=image1-image2

                            filename=os.path.basename(file1[0])[:-4]
                            folder_name = date.strftime('%Y%m%d')
                            event_folder= (cdaw_full.loc[cdaw_full["DATE_TIME"]==date, "FOLDER_NAME"].values)[0][:-4]
                            ofile = opath+"/"+folder_name+"/"+event_folder+"/"
                            final_path=ofile+"filtered/"
                            props_path=final_path+'mask_props'
                            # if not os.path.exists(final_path):
                            #     os.makedirs(final_path)
                            if not os.path.exists(props_path):
                                os.makedirs(props_path)
                            
                            all_center.append(occ_center[0])
                            all_plate_scl.append(plt_scl)                    
                            all_images.append(img)
                            all_dates.append(im_date)
                            all_occ_size.append(occ_size[0])
                            file_names.append(filename)
                            all_headers.append(header1)
                            all_ofiles.append(ofile)
                            occulter_size_ext.append(300)
        
        if len(all_images)>=2:
            data_list =  nn_seg.infer_event2(all_images, all_dates, w_metric,filter=filter, plate_scl=all_plate_scl, occulter_size=all_occ_size,centerpix=all_center,  plot_params=props_path, occulter_size_ext=occulter_size_ext)
            
            if len(data_list)==4:
                ok_orig_img,ok_dates,df, control_df =data_list
            
                if len(ok_dates)>0:
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
                    all_ofiles=[all_ofiles  for h,all_ofiles in enumerate(all_ofiles) if h not in all_idx]
                    
                    for d in all_ofiles:
                        os.makedirs(d,exist_ok=True)    
                    for m in range(len(ok_dates)):
                        event = df[df['DATE_TIME'] == ok_dates[m]].reset_index(drop=True)
                        control_event=control_df[control_df['DATE_TIME'] == ok_dates[m]].reset_index(drop=True)
                        for n in range(len(event['MASK'])):
                            if event['SCR'][n] > scr_threshold:             
                                masked = zeros.copy()
                                masked[:, :][(event['MASK'][n]) > mask_threshold] = 1
                                # save fits
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
                        
                        if plot_png:
                            plot_to_png(opath+"/"+folder_name+"/"+event_folder+"/"+file_names[m]+".png", [ok_orig_img[m]], event, control_df,[all_center[m]],mask_threshold=mask_threshold,scr_threshold=scr_threshold, title=[file_names[m]])  
                        
                    #Saves df for future comparison
                    col_drop= ['MASK'] # its necessary to drop mask column (its a matrix) to succsessfully save the csv file
                    df = df.drop(col_drop, axis=1)
                    if not os.path.exists(final_path):
                        os.makedirs(final_path)
                    df.to_csv(final_path+folder_name+'_filtered_stats', index=False)

                else:
                    print("WARNING: COULD NOT PROCESS EVENT "+ str(date)+", LESS THAN TWO IMAGES IN THE EVENT")
            
directory= model_path+"/"+"infer_neural_cme_seg_kincat_L1/lasco"+"_flor_test"
elements = os.listdir(directory)
for element in elements:
    element_path = os.path.join(directory, element)
    # Verificar si es una carpeta
    if os.path.isdir(element_path):
        # Verificar si la carpeta está vacía
        if not os.listdir(element_path):
            # Eliminar la carpeta vacía
            os.rmdir(element_path)

