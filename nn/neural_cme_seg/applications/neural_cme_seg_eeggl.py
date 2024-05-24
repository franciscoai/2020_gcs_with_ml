import os
import sys
from astropy.io import fits
import numpy as np
import torch.utils.data
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import csv
asd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(asd)
asd2=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(asd2)
asd3=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(asd3)
from neural_cme_seg_diego import neural_cme_segmentation
from ext_libs.rebin import rebin
import pandas as pd
from datetime import datetime, timedelta
import glob
from scipy.ndimage.filters import gaussian_filter
import pickle
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
from btot_file_sorted import btot_file_sorted
from manage_variables import manage_variables
#wrapper_eeggl is a file create in the local computer to run pyGCS_raytrace_eeggl.py
#This file should be copied and be updated on gehme server.
from wrapper_eeggl import *

def read_fits(file_path, header=False, imageSize=[512,512],smooth_kernel=[0,0]):
    try:       
        img = fits.open(file_path)[0].data
        img=img.astype("float32")
        img = gaussian_filter(img, sigma=smooth_kernel)
        if imageSize:
            img = rebin(img, imageSize, operation='mean')  
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
        img = np.flip(img, axis=0) 
        if header:
            return img, hdr
        else:
            return img 
    except:
        print(f'WARNING. I could not find file {file_path}')
        return None
    
def plot_to_png(ofile, orig_img, masks, scr_threshold=0.15, mask_threshold=0.6 , title=None, labels=None, boxes=None, scores=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
    # only detections with score larger than this value are considered
    color=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w']
    obj_labels = ['Back', 'Occ','CME','N/A']
    #
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(orig_img[0]), np.nan)
    fig, axs = plt.subplots(1, len(orig_img)*2, figsize=(20, 10))
    axs = axs.ravel()
    for i in range(len(orig_img)): #1 iteracion por imagen?
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
                    masked[:, :][masks[i][nb] > mask_threshold] = nb              
                    axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1) # add mask
                    box =  mpl.patches.Rectangle(b[0:2],b[2]-b[0],b[3]-b[1], linewidth=2, edgecolor=color[nb], facecolor='none') # add box
                    axs[i+1].add_patch(box)
                    if labels is not None:
                        axs[i+1].annotate(obj_labels[labels[i][nb]]+':'+'{:.2f}'.format(scr),xy=b[0:2], fontsize=15, color=color[nb])
                print(nb)
                nb+=1
    # axs[0].set_title(f'Cor A: {len(boxes[0])} objects detected') 
    # axs[1].set_title(f'Cor B: {len(boxes[1])} objects detected')               
    # axs[2].set_title(f'Lasco: {len(boxes[2])} objects detected')     
    #if title is not None:
    #    fig.suptitle('\n'.join([title[i]+' ; '+title[i+1] for i in range(0,len(title),2)]) , fontsize=16)   
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

#--------------------------------------------------------------------------------------------------------------------

def plot_to_png2(ofile, orig_img, event, all_center, mask_threshold, scr_threshold, title=None, plate_scl=1):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
    #mpl.use('TkAgg')
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
        axs[i].annotate( str(event["DATE_TIME"][i]),xy=[10,500], fontsize=15, color='w')

        #Incluir event date time
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
                axs[i+1].scatter(round(all_center[0][0]), round( all_center[0][1]), color='red', marker='x', s=100)
                axs[i+1].annotate(obj_labels[event['LABEL'][b]]+':'+'{:.2f}'.format(scr),xy=event['BOX'][b][0:2], fontsize=15, color=color[int(event['CME_ID'][b])])
                
                # draw a line on top of orig_img[i] (np array) from the center of the image to the image border at an angle event['AW_MIN'] given in radians
                im_half_size = int(orig_img[i].shape[0]/2)*0.9
                pt1 = (round(all_center[0][0]), round(all_center[0][1]))
                pt2 = (round(all_center[0][0] + im_half_size*np.cos(event['AW_MIN'][i])), round(all_center[0][1] + im_half_size*np.sin(event['AW_MIN'][i])))
                axs[i+1].plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='b', linewidth=2)
                # same for  event['AW_MAX']
                pt2 = (round(all_center[0][0] + im_half_size*np.cos(event['AW_MAX'][i])), round(all_center[0][1] + im_half_size*np.sin(event['AW_MAX'][i])))
                axs[i+1].plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='b', linewidth=2)
                # same for  event['CPA'] (this is a weighted average of angles)
                pt2 = (round(all_center[0][0] + im_half_size*np.cos(event['CPA'][i])), round(all_center[0][1] + im_half_size*np.sin(event['CPA'][i])))
                axs[i+1].plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='r', linewidth=2)
                #same for event['CPA'] calculated as AW_MIN+AW_MAX/2
                pt2 = (round(all_center[0][0] + im_half_size*np.cos(event['AW_MIN'][i]+(event['AW_MAX'][i]-event['AW_MIN'][i])/2)), round(all_center[0][1] + im_half_size*np.sin(event['AW_MIN'][i]+(event['AW_MAX'][i]-event['AW_MIN'][i])/2)))
                axs[i+1].plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='b', linewidth=2)

                # same for  event['APEX_ANGL']
                #apex in pixels
                apex_pixel = event['APEX'][i]/plate_scl
                pt2 = (round(all_center[0][0] + apex_pixel*np.cos(event['APEX_ANGL'][i])), round(all_center[0][1] + apex_pixel*np.sin(event['APEX_ANGL'][i])))
                # draw pt2 as a cross
                axs[i+1].scatter(pt2[0], pt2[1], color='g', marker='x', s=300)
                #Use a for loop to draw a point for each element corresponding to distance=event['APEX_DIST_PER'] and angle=event['APEX_ANGL_PER']
                for j in range(len(event['APEX_DIST_PER'][i])):
                    apex_pixel = event['APEX_DIST_PER'][i][j]/plate_scl
                    pt2 = (round(all_center[0][0] + apex_pixel*np.cos(event['APEX_ANGL_PER'][i][j])), round(all_center[0][1] + apex_pixel*np.sin(event['APEX_ANGL_PER'][i][j])))
                    axs[i+1].scatter(pt2[0], pt2[1], color='r', marker='x', s=100)
                #axs[i+1].plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='g', linewidth=2)
        #breakpoint()
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

#specific for eeggl functions
def rebin_interp(img,new_size=[512,512]):
    #rebinea imagenes, por default a 512
    x1 = np.linspace(0, img.shape[0], img.shape[0])-img.shape[0]
    y1 = np.linspace(0, img.shape[1], img.shape[1])-img.shape[1]
    #fun = interp2d(x1, y1, img, kind='linear')
    #interp2d is going to be deprecated, and replaced by RectBivariateSpline.
    spline = RectBivariateSpline(x1, y1, img)
    x = np.linspace(0, img.shape[0], new_size[0])-img.shape[0]
    y = np.linspace(0, img.shape[1], new_size[1])-img.shape[1]
    #Z3 = fun(x, y)
    interpolated_image = spline(x, y)
    return interpolated_image

def vec_to_matrix(vec,dim_matrix):
    #dim_matrix 300 en el caso de eeggl sta/b
    matrix = [vec[i:i+dim_matrix] for i in range(0, len(vec), dim_matrix)]
    matrix = np.array(matrix)
    matrix = matrix.T #traspongo xq la imagen se lee fila por fila desde 0,0 esquina izq arriba hacia abajo y luego fila por fila hacia la derecha.
    #matrix = matrix.astype("float32")   
    return matrix

def get_eeggl_header_correction(event,imsize=[512,512]):
    #This code will read the wrapper of a specific event (wrapper_YYYYMMDD), considering all the frames. It read each triplet of images and
    #calculate the average of hdr.CDELT1, hdr.CDELT2, and hdr.Rsun for each instrument of the triplet. 
    #This average is used to correct the header of the eeggl synth images. This is needed to correctly calculate the apex distance in Rsun units.
    #For each instrument it returns a list of mean values of [instr_cdel1, instr_cdelt2, instr_rsuns].
    #Esto deberia llamarse 1 sola vez!!! TODO, Cambiar.
    base_images    = ["","",""]
    cme_images     = ["","",""]
    cor2a_cdelt1    = []
    cor2a_cdelt2   = []
    cor2a_rsuns    = []
    cor2b_cdelt1   = []
    cor2b_cdelt2   = []
    cor2b_rsuns    = []
    lascoc2_cdelt1 = []
    lascoc2_cdelt2 = []
    lascoc2_rsuns  = []
    if event == '20110215':
        #breakpoint()
        for frame in range(10):
            base,cme = wrapper_20110215(frame)
            #continue if the frame exists
            if cme != 0:
                for index, img_dir in enumerate(cme):
                    #breakpoint()
                    img, hdr = read_fits(img_dir,header=True)
                    if index+1 ==1:
                        cor2a_rsuns.append(hdr['rsun'])
                        cor2a_cdelt1.append(hdr['cdelt1']) #suele ser 58.7999992372
                        cor2a_cdelt2.append(hdr['cdelt2']) #suele ser 58.7999992372
                    if index+1 ==2:
                        cor2b_rsuns.append(hdr['rsun'])
                        cor2b_cdelt1.append(hdr['cdelt1']) #suele ser 58.7999992372
                        cor2b_cdelt2.append(hdr['cdelt2']) #suele ser 58.7999992372
                    if index+1 ==3:
                        lascoc2_rsuns.append(hdr['rsun'])
                        lascoc2_cdelt1.append(hdr['cdelt1']) #suele ser 23.8
                        lascoc2_cdelt2.append(hdr['cdelt2']) #suele ser 23.8
    
    return [np.mean(cor2a_cdelt1),np.mean(cor2a_cdelt2),np.mean(cor2a_rsuns)],[np.mean(cor2b_cdelt1),np.mean(cor2b_cdelt2),np.mean(cor2b_rsuns)],[np.mean(lascoc2_cdelt1),np.mean(lascoc2_cdelt2),np.mean(lascoc2_rsuns)]

def create_header(matrix,time,instrument="",imsize=[512,512]):
    header = fits.Header()
    header['NAXIS1'] = imsize[0]
    header['NAXIS2'] = imsize[1]
    header['CRPIX1'] = imsize[0]/2
    header['CRPIX2'] = imsize[1]/2
    tiempo = time.split('"')
    header['time'] = tiempo[1]
    #c2a, c2b, c2 = get_eeggl_header_correction(20110215)
    #if instrument == 'cor2_a':
    #    header['CDELT1']=c2a[0]
    #    header['CDELT2']=c2a[1]
    #    header['RSUN']  =c2a[2]
    #elif instrument == 'cor2_b':
    #    header['CDELT1']=c2b[0]
    #    header['CDELT2']=c2b[1]
    #    header['RSUN']  =c2b[2]
    #elif instrument == 'lascoC2':
    #    header['CDELT1']=c2[0]
    #    header['CDELT2']=c2[1]
    #    header['RSUN']  =c2[2]
    #else:
    #    print("Instrument not recognized")
    return header

def read_dat(file,path="",dim_matrix=300,instrument=""):
    #instrument variable is required to read real hedears and create cdelt1/2 and rsun.
    pos_x=[]
    pos_y=[]
    pos_wl=[]
    pos_pb=[]
    #print(path+file)
    
    with open(path+file, 'r') as file:
    # Read all lines into a list
        line0, line1, line2, line3, line4, line5, line6, line7 = [file.readline() for _ in range(8)]
        time_event = line3
        time_event_start = line4
        time_seconds = line5

    # Process each line
        for line in file:            
            x, y, wl, pb = map(float, line.split())
            pos_x.append(x)
            pos_y.append(y)
            pos_wl.append(wl)
            pos_pb.append(pb)
    
    x_pos  = vec_to_matrix(pos_x,dim_matrix)
    y_pos  = vec_to_matrix(pos_y,dim_matrix)
    wl_pos = vec_to_matrix(pos_wl,dim_matrix)
    pb_pos = vec_to_matrix(pos_pb,dim_matrix)
    hdr = create_header(matrix=wl_pos,time=time_event,instrument=instrument)
    return x_pos, y_pos, wl_pos, pb_pos,hdr


#main
#------------------------------------------------------------------Testing the CNN--------------------------------------------------------------------------
aux_in = "/gehme-gpu"
#aux_in = "/gehme-gpu"
#model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4" #no contiene oculter externo
model_path= "/gehme/projects/2020_gcs_with_ml/output/neural_cme_seg_v4"
#model_version="v4"
trained_model = '6000.torch'
#model_path= "/gehme/projects/2020_gcs_with_ml/output/neural_cme_seg_v5" #contiene oculter externo
model_version="v4"
#trained_model = '66666.torch'

#--------------------------
#Select only one of the folllowing image types
#run = 'run016' #'run005'
run = 'run001_AWSoM_restart_run038_AWSoM'
eeggl = True        #Synthetic images created using eeggl.
btot  = False       #Synthetic images created using pyGCS.
real_img = False    #Real images from cor2a, cor2b, lascoC2, level1.
modified_masks = None #If True, it will use the modified masks for the real images. If None, it will use the original masks.
#---------------------------
#increase contrast radialy, usefull for siimulations.
increase_contrast = True #None or True. Will apply to infer2. IMPORTANT for eeggl.
#select date of event
cme_date_event = '2011-02-15' 

#select aproach
base_difference = False
running_difference = True

#select instrument of event
#instr='cor2_a'
instr='cor2_b'
#instr='lascoC2'

#select infer event
infer_event2=True
infer_event1=True

#manage input and output paths
ipath,opath,dir_modified_masks,list_name = manage_variables(cme_date_event,eeggl=eeggl,btot=btot,real_img=real_img,instr=instr,simulation_run=run,infer_event2=infer_event2,modified_masks=modified_masks)
#breakpoint()
mask_threshold = 0.6 # value to consider a pixel belongs to the object
scr_threshold  = 0.56 # only detections with score larger than this value are considered



#----------------------------

#main
gpu=0 # GPU to use
#If gpu 1 is out of ram, use gpu=0 or cpu. Check gpu status using nvidia-smi command on terminal.
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unless its not available
print(f'Using device:  {device}')

#loads nn model
nn_seg = neural_cme_segmentation(device, pre_trained_model = model_path + "/"+ trained_model, version=model_version)

os.makedirs(opath, exist_ok=True)
#inference on all images

#Select masks for different instruments
#if instrument == 'cor2':
#    mask = "/gehme-gpu/projects/2023_eeggl_validation/repo_diego/2020_gcs_with_ml/instrument_mask/mask_cor2_512.fts"
#    img_mask = read_fits(mask)


#list of all images in temporal order.
if real_img:
    init_range = 1
    #list_name = 'list.txt'
    with open(ipath+list_name, 'r') as file:
        lines = file.readlines()
    image_names = [line.strip() for line in lines]

if eeggl:
    init_range = 1
    #list_name = 'lista_sta_cor2_ordenada.txt'
    with open(ipath+list_name, 'r') as file:
        lines = file.readlines()
    image_names = [line.strip() for line in lines]
    #promedios de cdelt1, cdelt2, rsun para cor2a, cor2b, c2 obtenido leyendo las imagenes reales.
    c2a, c2b, c2 = get_eeggl_header_correction(cme_date_event.replace("-", ""))
#breakpoint()
if btot:
    init_range = 0
    image_names = btot_file_sorted(ipath,instr)

if infer_event2:
    all_images=[]
    all_dates=[]
    all_occ_size=[]
    all_plate_scl=[]
    file_names=[]
    all_headers=[]
    all_center=[]
    all_occulter_size_ext=[]

if instr != 'lascoC2':
    occ_center=[[256,256],[256,256]]#[255,265]] #[260,247] para cor2b
occ_size = [50,60] #70 para cor2b

if instr=="cor2_a":
    occ_center=occ_center[0]
    if occ_center:
        occ_size= occ_size[0]
    occulter_size_ext = 250
elif instr=="cor2_b":
    occ_size= occ_size[1]
    if occ_center:
        occ_center=occ_center[1]
    occulter_size_ext = 250
    #else:
    #    occ_center=[hdr1["crpix1"],hdr1["crpix2"]]
if instr=="lascoC2":
    occ_size = 90
    occulter_size_ext = 300
if infer_event2:
    os.makedirs(opath+'mask_props/', exist_ok=True)

for j in range(init_range,len(image_names)):

    if real_img == True:        
        if base_difference:
            #First image set to background
            img0, hdr0 = read_fits(ipath+image_names[0],header=True)
            img1, hdr1 = read_fits(ipath+image_names[j],header=True)
        #    img_diff[img_mask == 0] = 0
            
        if running_difference:
            img0, hdr0 = read_fits(ipath+image_names[j-1],header=True)
            img1, hdr1 = read_fits(ipath+image_names[j  ],header=True)

        #if 'occ_center' != locals():
        if instr =="lascoC2":    
            occ_center=[hdr1["crpix1"],hdr1["crpix2"]]
            #En esta resta asumimos que ambas imagenes tienen igual centerpix1/2, y ambas son north up.

    if btot == True:
        #sin running difference.
        img0       = np.zeros((512, 512))
        img1, hdr1 = read_fits(ipath+image_names[j],header=True)
        #opath = '/gehme/projects/2023_eeggl_validation/output/2011-02-15/gcs/cor2a/'

    if eeggl== True:
        if running_difference:
            x_pos, y_pos, wl_mat0, pb_mat0,hdr0 = read_dat(image_names[j-1],path=ipath,dim_matrix=300)
            wl_mat0 = np.flip(wl_mat0, axis=0)
            img0 = rebin_interp(wl_mat0,new_size=[512,512])
            x_pos, y_pos, wl_mat1, pb_mat1,hdr1 = read_dat(image_names[j  ],path=ipath,dim_matrix=300)
            wl_mat1 = np.flip(wl_mat1, axis=0)
            img1 = rebin_interp(wl_mat1,new_size=[512,512])
            if instr == 'cor2_a':
                hdr1['CDELT1']=c2a[0]
                hdr1['CDELT2']=c2a[1]
                hdr1['RSUN']  =c2a[2]
            elif instr == 'cor2_b':
                hdr1['CDELT1']=c2b[0]
                hdr1['CDELT2']=c2b[1]
                hdr1['RSUN']  =c2b[2]
            elif instr == 'lascoC2':
                hdr1['CDELT1']=c2[0]
                hdr1['CDELT2']=c2[1]
                hdr1['RSUN']  =c2[2]

        img_ratio = img1 / img0 #only used to calculate the occulter


    img_diff = img1 - img0
    
    if eeggl== True:
        min_value = np.min(img_diff)
        img_diff[np.isnan(img_ratio)] = min_value
    #openCV need astype matrix float32 or lower, does not accept float64
    img_diff = img_diff.astype("float32")
    
    if infer_event1:
    #Infer1
        #orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, occulter_size=60,centerpix=[255,255+10],occulter_size_ext=240) 
    
    #Cor2 con mascara original
        #orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, getmask=True,hdr=hdr1) 
        #breakpoint()

        if instr=='cor2_b':
            #Cor2 con mascara centrada ---> Cor2B me funciona mejor con esto, usando crpix1/2
            orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, occulter_size=occ_size,
                                                                    centerpix=occ_center,occulter_size_ext=occulter_size_ext,path=opath,histogram_names=image_names[j])
        
        if instr=='cor2_a':
            #Cor2 con mascara centrada (a medida) ---> Cor2A me funciona mejor con esto
            if eeggl==False:
                orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, occulter_size=occ_size,
                                                                        centerpix=occ_center,occulter_size_ext=occulter_size_ext)
            if eeggl==True:
                orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, occulter_size=occ_size,
                                                                        centerpix=occ_center,occulter_size_ext=occulter_size_ext)#,repleace_value=min_value)
        if instr=='lascoC2':    
            #C2
            orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, occulter_size=occ_size,
                                                                    centerpix=occ_center,occulter_size_ext=occulter_size_ext)
            
    # plot the predicted mask
        ofile = opath+"/"+os.path.basename(image_names[j])+'infer1.png'
        if not infer_event2:
            plot_to_png(ofile, [orig_img], [masks], scores=[scores], labels=[labels], boxes=[boxes])
        
        
        #contour_plot = plt.contour(asd)
        #plt.colorbar(contour_plot, label='Matrix Values')
        #plt.savefig('contour'+ofile)
        #plt.close()
    #----------------------------------------------------------------------------------------
    #infer2
    if infer_event2:
        
        if eeggl==True:
            date_format = '%Y/%m/%dT%H:%M:%S'
            time_string=hdr1['time']
            date = datetime.strptime(time_string[:-4], date_format)
            percentiles = [2,98]

        if real_img==True:    
            if instr != 'lascoC2':    
                date = datetime.strptime(image_names[j][0:-10],'%Y%m%d_%H%M%S')
                percentiles = [2,98]
            if instr == 'lascoC2':
                date_string = hdr1['fileorig'][:-4]
                if int(date_string[:2]) < 94: #soho was launched in 1995
                    aux = '20'
                else:
                    aux = '19'
                date = datetime.strptime(date_string,'%Y%m%d_%H%M%S')

        if btot ==True:
            if instr != 'lascoC2':
                name_file = hdr1["filename"]
                date = datetime.strptime(name_file[0:-10],'%Y%m%d_%H%M%S')
                percentiles = [0.1,99.9]
            if instr == 'lascoC2': 
                breakpoint()
        
        #Calculate pixel size in Rsun units.
        #cdelt = pixel size in arsec
        #Rsun = Rsun in arcsec
        plt_scl = hdr1['CDELT1']/hdr1['Rsun']


        all_center.append(occ_center)
        all_plate_scl.append(plt_scl)                    
        all_images.append(img_diff)
        all_dates.append(date)
        all_occ_size.append(occ_size)
        file_names.append(image_names[j])
        all_headers.append(hdr1)
        all_occulter_size_ext.append(occulter_size_ext)

if infer_event2:
    #dir_modified_masks = '/gehme/projects/2023_eeggl_validation/repo_diego/2020_gcs_with_ml/nn/neural_cme_seg/applications/EEGGL_project/new_masks20110215_cor2a_v10.pkl'
    ok_orig_img,ok_dates, df =  nn_seg.infer_event2(all_images, all_dates, filter=filter, plate_scl=all_plate_scl,resize=False,
                                                    occulter_size=all_occ_size,occulter_size_ext=all_occulter_size_ext,
                                                    centerpix=all_center,plot_params=opath+'mask_props',filter_halos=False,percentiles=percentiles,
                                                    modified_masks=dir_modified_masks,increase_contrast=increase_contrast)
    
    zeros = np.zeros(np.shape(ok_orig_img[0]))
    all_idx=[]
    #new_ok_dates=[datetime.utcfromtimestamp(dt.astype(int) * 1e-9) for dt in ok_dates]

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

    save_full_pickle = True
    if save_full_pickle:
        dict={}
        dict["df"]=df
        dict["ok_orig_img"]=ok_orig_img
        dict["ok_dates"]=ok_dates
        dict["file_names"]=file_names
        dict["all_center"]=all_center
        dict["all_plate_scl"]=all_plate_scl
        dict["all_dates"]=all_dates
        dict["all_occ_size"]=all_occ_size
        dict["all_headers"]=all_headers
        dict['scr_threshold']=scr_threshold
        dict['mask_threshold']=mask_threshold
        #breakpoint()
        with open(opath+'/full_parametros_pre_plot2.pkl', 'wb') as write_file:
            pickle.dump(dict, write_file)
            print("Archivo pickle guardado en: ", opath+'/full_parametros_pre_plot2.pkl') 

    for m in range(len(ok_dates)):
        event = df[df['DATE_TIME'] == ok_dates[m]].reset_index(drop=True)
        image=ok_orig_img[m]
        for n in range(len(event['MASK'])):
            if event['SCR'][n] > scr_threshold:             
                masked = zeros.copy()
                masked[:, :][(event['MASK'][n]) > mask_threshold] = 1
                # safe fits
                
                ofile_fits = os.path.join(os.path.dirname(opath), file_names[m]+"_CME_ID_"+str(int(event['CME_ID'][n]))+'.fits')
                h0 = all_headers[m]
                fits.writeto(ofile_fits, masked, h0, overwrite=True, output_verify='ignore')
                print("Saving fits: "+ofile_fits)
        #breakpoint()
        plot_to_png2(opath+file_names[m]+"infer2.png", [ok_orig_img[m]], event,[all_center[m]],mask_threshold=mask_threshold,
                    scr_threshold=scr_threshold, title=[file_names[m]], plate_scl=all_plate_scl[m])
        print("Imagenes creadas en:" + opath)
    #breakpoint()
    #with open(opath+'mask_props/save_all_data.pkl', 'wb') as write_file:
    #    data_dict={}
    #    data_dict["df"]=df
    #    data_dict["ok_orig_img"]=ok_orig_img
    #    data_dict["ok_dates"]=ok_dates
    #    pickle.dump(data_dict,write_file)    

print("Program finished without errors")







"""
for f in files:
    print(f'Processing file {f}')
    if file_ext == ".fits" or file_ext == ".fts":
        img = read_fits(f)
    else:
        img = cv2.imread(f)
    breakpoint()
    orig_img, masks, scores, labels, boxes  = nn_seg.infer(img, model_param=None, resize=False, occulter_size=0)
    # plot the predicted mask
    ofile = opath+"/"+os.path.basename(f)+'.png'
    plot_to_png(ofile, [img], [masks], scores=[scores], labels=[labels], boxes=[boxes])
"""

"""    
    #-----------------------------------------------------------------------
    valores = img_diff[~np.isnan(img_diff)]
    hist, bins, _ = plt.hist(valores, bins=30, range=((np.min(valores)+np.std(valores)), (np.max(valores)-np.std(valores))), color='blue', alpha=0.7, density=True)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img_diff, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.plot(hist, color='gray')
    type_info = img_diff.dtype
    min_value = round(np.nanmin(img_diff),2)
    max_value = round(np.nanmax(img_diff),2)
    mean_value = round(np.nanmean(img_diff),2)
    plt.title(f'Type: {type_info}, Min: {min_value}')
    plt.xlabel(f'Max: {max_value}, Mean: {mean_value:.2f}')
    plt.show()
    plt.savefig("/gehme-gpu/projects/2023_eeggl_validation/output/histo_real.png")
    #-----------------------------------------------------------------------
    """
