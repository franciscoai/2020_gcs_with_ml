def read_fits(file_path,smooth_kernel=[0,0]):
    imageSize=[512,512]
    
    try:       
        
        file=glob.glob((file_path)[0:-5]+"*")
        file = [r for r in file if r.endswith((".fits", ".fts"))]
        img = fits.open(file[0])
        img=(img[0].data).astype("float32")
        #img = gaussian_filter(img, sigma=smooth_kernel)

        # if smooth_kernel[0]!=0 and smooth_kernel[1]!=0: 
        img = rebin(img, imageSize,operation='mean')
        
        return img  
    except:
        print(f'WARNING. I could not find file {file_path}')
        return None


def rec2pol(mask,scores,labels,boxes,imsize):
    
    mask_threshold = 0.6 # value to consider a pixel belongs to the object
    scr_threshold = 0.6 # only detections with score larger than this value are considered
    nans = np.full(imsize, np.nan)
    pol_mask=[]
    pts=[]
    try:
        if boxes is not None:
            nb=0
            for b in boxes:
                if scores is not None:
                    scr = scores[0]
                else:
                    scr = 0   
                if scr > scr_threshold:
                    #creates an array with zero value inside the mask and Nan value outside             
                    masked = nans.copy()
                    masked[:, :][mask[nb] > mask_threshold] = nb   
                    
                    #calculates geometric center of the image
                    height, width = masked.shape
                    center_x = width // 2
                    center_y = height // 2
                    center=[center_x,center_y]
                    #calculates distance to the point and the angle for the positive x axis
                    for x in range(width):
                        for y in range(height):
                            value=masked[x,y]
                            if not np.isnan(value):
                                pts.append([y,x])#the coordinates are written in the format (y,x)
                                x_dist = (x-center_x)
                                y_dist = (y-center_y)
                                distance= math.sqrt(x_dist**2 + y_dist**2)
                                angle=np.arctan2(y_dist,x_dist)+np.radians(270)
                                pol_mask.append([distance,angle])

        return pol_mask,center,masked,pts
    except:
        print("A parameter its None")



def plot_to_png(ofile,image_path,orig_img, masks, pol_mask, center,imsize, title=None, labels=None, boxes=None, scores=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
    mask_threshold = 0.6 # value to consider a pixel belongs to the object
    scr_threshold = 0.6 # only detections with score larger than this value are considered
    color=['r','b','g','k','y']
    obj_labels = ['Occ', 'CME','N/A','N/A']
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(orig_img[0]), np.nan)
    fig, axs = plt.subplots(1, len(orig_img)*2, figsize=(20, 10))
    axs = axs.ravel()

    #takes the min and max angles and calculates cpa and wide angles
    angles = [s[1] for s in pol_mask]
    ang=[]
    points=[]
    min_ang = min(angles)
    max_ang = max(angles)
    cpa_ang=(max_ang+min_ang)/2
    wide_ang=(max_ang-min_ang)
    ang.append([min_ang,max_ang,cpa_ang])

    #transforms angles to degrees for display purposes
    angulo_max=np.degrees(max_ang)
    angulo_min=np.degrees(min_ang)
    angulo_cpa=np.degrees(cpa_ang)
    title=str(angulo_max)+"/"+str(angulo_cpa)+"/"+str(angulo_min)

    #obtains end points and direction for the lines
    for i in ang[0]:
        vector=np.array([np.cos(i), np.sin(i)])
        start_point = center
        x_end = start_point[0] + (imsize[0]/2) * vector[0]
        y_end = start_point[1] + (imsize[1]/2) * vector[1]
        x_points = np.array([start_point[0], x_end])
        y_points = np.array([start_point[1], y_end])
        points.append([x_points,y_points])

    #saves the image
    for i in range(len(orig_img)):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(orig_img[0], vmin=0, vmax=1, cmap='gray')
        axs[0].axis('off')
        axs[1].imshow(orig_img[0], vmin=0, vmax=1, cmap='gray')      
        axs[1].axis('off')        
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
                    axs[i+1].plot(points[0][0], 512-points[0][1], color='blue', label='Recta min')
                    axs[i+1].plot(points[1][0],512-points[1][1] , color='orange', label='Recta max')
                    axs[i+1].plot(points[2][0], 512-points[2][1], color='green', label='Recta cpa')
                    axs[i+1].scatter(256,256, color='purple', marker='o')
                    if labels is not None:
                        axs[i+1].annotate(obj_labels[labels[i][nb]]+':'+'{:.2f}'.format(scr),xy=b[0:2], fontsize=15, color=color[nb])
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

    #calculates diferent angles an parameters
    ang_list=[]
    date_time=datetime.strptime(f[:-6],"%Y%m%d_%H%M%S")
    masked[masked == 0] = 1
    cm = center_of_mass(np.nan_to_num(masked))#returns center of mass cordinates in the form (y,x)
    center_x=imsize[0]
    center_y=imsize[1]
    x_dist = (cm[1]-center_x)
    y_dist = (cm[0]-center_y) 
    cm_ang=np.arctan2(y_dist,x_dist)+np.radians(270)
    distance = [s[1] for s in pol_mask]
    distance_abs= max(distance, key=abs)
    idx_dist = distance.index(distance_abs)
    apex_dist= distance[idx_dist] 
    apex_ang=angles[idx_dist]
    ang_list.extend([image_path, date_time, min_ang, max_ang, cpa_ang, wide_ang, cm[1], cm[0], cm_ang, apex_dist, apex_ang])                      
    return ang_list



import pickle
import os
import sys
import math
import glob
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.ndimage import center_of_mass
from astropy.io import fits

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from ext_libs.rebin import rebin


data_dir="/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v3/infer_neural_cme_seg_kincat/cor2_a/"
image_dir="/gehme/data/stereo/secchi/L0/a/img/cor2/"
imsize=[512,512] #shape of the mask image



folders = os.listdir(data_dir)
for folder in folders:
    files=os.listdir(data_dir+folder)
    ang_data=[]
    for file in files:
        if file.endswith(".pkl"):
            print("processing file "+str(file))
            img_path=data_dir+folder+"/"+str(file[:-11]+".fts")
            pkl_path=data_dir+folder+"/"+file
            with open(pkl_path, 'rb') as archivo_pickle:
                mask = pickle.load(archivo_pickle)
                scores = pickle.load(archivo_pickle)
                labels= pickle.load(archivo_pickle)
                boxes= pickle.load(archivo_pickle)

            pol=rec2pol(mask,scores,labels,boxes,imsize=imsize)
            if (pol is not None) :
                pol_mask=pol[0]
                center= pol[1]
                masked=pol[2]
                pts=pol[3]
                f=file[:-11]
                img=read_fits(img_path)
                folder_path = data_dir+folder
                final_path = os.path.join(folder_path, "stats")
                if not os.path.exists(final_path):
                    os.makedirs(final_path)
                opath=final_path+"/"+str(f)+"_stats.png"
                ang_list=plot_to_png(opath,img_path,[img],[mask], title=[f],labels=[labels], boxes=[boxes], scores=[scores],pol_mask=pol_mask, center=center,imsize=imsize)
                ang_data.append(ang_list)

    columnas = ['PATH', 'DATE_TIME', 'MIN_ANG','MAX_ANG','CPA_ANG','WIDE_ANG','MASS_CENTER_X','MASS_CENTER_Y','MASS_CENTER_ANG','APEX_RADIUS','APEX_ANG'] 
    df = pd.DataFrame(ang_data, columns=columnas)
    df.to_csv(final_path+'/'+folder+'_stats', index=False)
