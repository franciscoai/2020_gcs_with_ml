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
from scipy.ndimage.filters import gaussian_filter
import csv
from neural_cme_seg import neural_cme_segmentation
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
asd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(asd)
asd2=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(asd2)
asd3=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(asd3)
from ext_libs.rebin import rebin
import pandas as pd
from datetime import datetime, timedelta
import glob
from scipy.ndimage.filters import gaussian_filter
import pickle
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline


def mask_occulter(img, occulter_size,centerpix,repleace_value=None):
    '''
    Replace a circular area of radius occulter_size in input image[h,w,3] with a constant value
    repleace_value: if None, the area is replaced with the image mean. If set to scalar float that value is used
    '''
    if centerpix is not None:
        w=int(round(centerpix[0]))
        h=int(round(centerpix[1]))
    else:
        h,w = img.shape[:2]
        h=int(h/2)
        w=int(w/2)

    mask = np.zeros((img.shape[0],img.shape[0]), dtype=np.uint8)
    cv2.circle(mask, (w,h), occulter_size, 1, -1)
    
    if repleace_value is None:
        img[mask==1] = 0#np.nan #np.mean(img)
    else:
        img[mask==1] = repleace_value
    return img


def read_fits(file_path, header=False, imageSize=[512,512]):
    try:       
        img = fits.open(file_path)[0].data
        img=img.astype("float32")
        img = gaussian_filter(img, sigma=[0,0])
        #img = rebin(img, imageSize, operation='mean')
        img = rebin_interp(img, new_size=[512,512])
        img = np.flip(img, axis=0)    
        if header:
            return img, fits.open(file_path)[0].header
        else:
            return img 
    except:
        print(f'WARNING. Error while processing {file_path}')
        return None
    
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
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

#--------------------------------------------------------------------------------------------------------------------

def plot_to_png2(ofile, orig_img, event, all_center, mask_threshold, scr_threshold, title=None, plate_scl=1):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
    # only detections with score larger than this value are considered
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
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()
    
#main
#------------------------------------------------------------------Testing the CNN--------------------------------------------------------------------------
model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4"
model_version="v4"
repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ipath = '/gehme/data/solo/fsi/20220325/'
opath= repo_path + '/output/fsi/20220325/'
file_ext=".fits" # files to use
trained_model = '6000.torch'
occ_size = 120 # Artifitial occulter radius in pixels. Use 0 to avoid. [Stereo a/b C1, Stereo a/b C2, Lasco C2]
occ_center = [258.8,256.1] # Center of the occulter. Use None to avoid. [Stereo a/b C1, Stereo a/b C2, Lasco C2]
occulter_size_ext = 500 # Size of the occulter in pixels. Use 0 to avoid. [Stereo a/b C1, Stereo a/b C2, Lasco C2]
run_diff=True

#main
gpu=0 # GPU to use
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unless its not available
print(f'Using device:  {device}')
#loads images
with open(ipath+'list.txt', 'r') as file:
    lines = file.readlines()
files = [line.strip() for line in lines]
files = [os.path.join(ipath, e) for e in files]

#loads nn model
nn_seg = neural_cme_segmentation(device, pre_trained_model = model_path + "/"+ trained_model, version=model_version)
os.makedirs(opath, exist_ok=True)
#inference on all images
for j in range(1,len(files)):
    print(f'Processing file {files[j]}')
    if run_diff:
        img0 = read_fits(files[j-1])
        img1 = read_fits(files[j  ])
        img = img1 - img0
        img = img.astype("float32")
    orig_img, masks, scores, labels, boxes  = nn_seg.infer(img, model_param=None, resize=False, occulter_size=occ_size,centerpix=occ_center,occulter_size_ext=occulter_size_ext)
    # plot the predicted mask
    ofile = opath+"/"+os.path.basename(files[j])+'_3.png'
    plot_to_png(ofile, [orig_img], [masks], scores=[scores], labels=[labels], boxes=[boxes])

