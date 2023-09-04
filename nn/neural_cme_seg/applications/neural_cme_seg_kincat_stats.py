
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
from nn.neural_cme_seg.neural_cme_seg import neural_cme_segmentation



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


def _rec2pol(imsize,mask_threshold,mask):
        '''
        Converts the x,y mask to polar coordinates
        Only pixels above the mask_threshold are considered
        TODO: Consider arbitrary image center
        '''
        nans = np.full(imsize, np.nan)
        pol_mask=[]
        #creates an array with zero value inside the mask and Nan value outside             
        masked = nans.copy()
        masked[:, :][mask > mask_threshold] = 0   
        #calculates geometric center of the image
        height, width = masked.shape
        center_x = width // 2
        center_y = height // 2
        #calculates distance to the point and the angle for the positive x axis
        for x in range(width):
            for y in range(height):
                value=masked[x,y]
                if not np.isnan(value):
                    x_dist = (x-center_x)
                    y_dist = (y-center_y)
                    distance= np.sqrt(x_dist**2 + y_dist**2)
                    angle = np.arctan2(x_dist,y_dist)
                    if angle<0:
                        angle+=2*np.pi
                    pol_mask.append([distance,angle])
        return pol_mask


# def rec2pol(mask,scores,labels,boxes,imsize):
    
#     mask_threshold = 0.5 # value to consider a pixel belongs to the object
#     scr_threshold = 0.3 # only detections with score larger than this value are considered
#     nans = np.full(imsize, np.nan)
#     pol_mask=[]
#     pts=[]
#     try:
#         if boxes is not None:
#             for b in boxes:
#                 if scores is not None:
#                     scr = scores
#                 else:
#                     scr = 0   
#                 if scr > scr_threshold:
#                     #creates an array with zero value inside the mask and Nan value outside             
#                     masked = nans.copy()
#                     masked[:, :][mask > mask_threshold] = 0   
                    
#                     #calculates geometric center of the image
#                     height, width = masked.shape
#                     center_x = width // 2
#                     center_y = height // 2
#                     center=[center_x,center_y]
#                     #calculates distance to the point and the angle for the positive x axis
#                     for x in range(width):
#                         for y in range(height):
#                             value=masked[x,y]
#                             if not np.isnan(value):
#                                 pts.append([y,x])#the coordinates are written in the format (y,x)
#                                 x_dist = (x-center_x)
#                                 y_dist = (y-center_y)
#                                 distance= math.sqrt(x_dist**2 + y_dist**2)
#                                 angle=np.arctan2(y_dist,x_dist)+np.radians(270)
#                                 pol_mask.append([distance,angle])

#         return pol_mask,center,masked,pts
#     except:
#         print("A parameter is None")

def _compute_mask_prop(imsize,image_path,orig_img,file_title,scr_threshold,mask_threshold, masks, scores, labels, boxes, plate_scl=1, filter_halos=True, occulter_size=0):    
        '''
        Computes the CPA, AW and apex radius for the given 'CME' masks.
        If the mask label is not "CME" and if is not a Halo and filter_halos=True, it returns None
        plate_scl: if defined the apex radius is scaled based on the plate scale of the image. If 0, no filter is applied
        filter_halos: if True, filters the masks with boxes center within the occulter size, and wide_ angle > MAX_WIDE_ANG
        TODO: Compute apex in Rs; Use generic image center
        '''
        
        max_wide_ang = np.radians(270.) # maximum angular width [deg]
        obj_labels=['Background','Occ','CME']
        nans = np.full(np.shape(orig_img[0]), np.nan)
        prop_list=[]
        
        for i in range(len(masks)):
            if filter_halos:
                box_center = np.array([boxes[i][0]+(boxes[i][2]-boxes[i][0])/2, boxes[i][1]+(boxes[i][3]-boxes[i][1])/2])
                distance_to_box_center = np.sqrt((imsize[0]/2-box_center[0])**2 + (imsize[1]/2-box_center[1])**2)
                if distance_to_box_center < 0.8 * occulter_size:
                    halo_flag = False
                else:
                    halo_flag = True
            else:
                halo_flag = True
            
            if obj_labels[labels[i]]=='CME' and scores[i]>scr_threshold and halo_flag: 

                pol_mask=_rec2pol(imsize,mask_threshold,masks[i])
                if (pol_mask is not None):            
                    #takes the min and max angles and calculates cpa and wide angles
                    angles = [s[1] for s in pol_mask]
                    if len(angles)>0:
                        # checks for the case where the cpa is close to 0 or 2pi
                        min_ang = np.min(angles)
                        max_ang = np.max(angles)
                                         

                        cm_mask = nans.copy()
                        cm_mask[:, :][masks[0] > mask_threshold] = 1 
                        cm = center_of_mass(np.nan_to_num(cm_mask))
                        center_x=imsize[0]/2
                        center_y=imsize[1]/2
                        x_dist = (cm[1]-center_x)
                        y_dist = (cm[0]-center_y)
                        cm_dist= math.sqrt(x_dist**2 + y_dist**2)
                        cm_ang=np.arctan2(-y_dist,x_dist)
                         
                        
                        if max_ang-min_ang >= 0.9*2*np.pi:
                            angles = [s-2*np.pi if s>np.pi else s for s in angles]
                        cpa_ang= np.median(angles)
                        if cpa_ang < 0:
                            cpa_ang += 2*np.pi
                        wide_ang=np.abs(np.percentile(angles, 95)-np.percentile(angles, 5))
                        #calculates diferent angles an parameters
                        distance = [s[0] for s in pol_mask]
                        distance_abs= max(distance, key=abs)
                        idx_dist = distance.index(distance_abs)
                        apex_dist= distance[idx_dist] * plate_scl
                        apex_ang=angles[idx_dist]
                        if apex_ang < 0:
                            apex_ang += 2*np.pi
                        # if self.debug_flag == 5:
                        #     plt.hist(angles, bins=50)
                        #     plt.savefig(self.plot_params + '/angles_hist.png')
                        #     plt.close()
                        #     breakpoint()

                        date_time=datetime.strptime(f[:-6],"%Y%m%d_%H%M%S")
                        
                        if filter_halos:
                            if wide_ang < max_wide_ang:
                                prop_list.append([image_path,i,date_time,float(scores[i]),min_ang, max_ang,cpa_ang, wide_ang, cm[1], cm[0],cm_dist, cm_ang, apex_dist, apex_ang]) 
                        else:
                            prop_list.append([image_path,i,date_time,float(scores[i]),min_ang, max_ang,cpa_ang, wide_ang,cm[1], cm[0],cm_dist, cm_ang, apex_dist, apex_ang])
        
        return prop_list         


def plot_to_png(ofile,data_list,scr_threshold,masks,labels=None,boxes=None,scores=None,title=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """ 
    color=['r','b','g','k','y']
    obj_labels = ['Occ', 'CME','N/A','N/A']
    cmap = mpl.colors.ListedColormap(color) 
    nans = np.full(np.shape(orig_img[0]), np.nan)
    for i in range(len(data_list)):
        ang=[]
        points=[]
        img_path=data_list[i][0]
        orig_img=read_fits(img_path)
        min_ang=data_list[i][4]
        max_ang=data_list[i][5]
        cpa_ang=data_list[i][6]
        cm_ang=data_list[i][11]
        ang.append([min_ang,max_ang,cpa_ang,cm_ang])
        center=[imsize[0]/2,imsize[1]/2]
        for j in ang[0]:
            vector=np.array([np.cos(j), np.sin(j)])
            start_point = center
            x_end = start_point[0] + (imsize[0]/2) * vector[0]
            y_end = start_point[1] + (imsize[1]/2) * vector[1]
            x_points = np.array([start_point[0], x_end])
            y_points = np.array([start_point[1], y_end])
            points.append([x_points,y_points])


        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(orig_img, vmin=0, vmax=1, cmap='gray')
        axs[0].axis('off')
        axs[1].imshow(orig_img, vmin=0, vmax=1, cmap='gray')      
        axs[1].axis('off')        
        if boxes is not None:
            #print("boxes ok")
            if scores is not None:
                scr = scores[0][i]
            else:
                scr = 0   
                if scr > scr_threshold:
                        #print("score is ok")             
                        masked = nans.copy()
                        masked[:, :][masks[0][i] > mask_threshold] = 0              
                        axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1) # add mask
                        axs[i+1].plot(points[0][0], 512-points[0][1], color='blue', label='Recta min')
                        axs[i+1].plot(points[1][0],512-points[1][1] , color='orange', label='Recta max')
                        axs[i+1].plot(points[2][0], 512-points[2][1], color='green', label='Recta cpa')
                        axs[i+1].plot(points[3][0], 512-points[3][1], color='purple', label='Recta centro de masa')
                        if labels is not None:
                            axs[i+1].annotate(obj_labels[labels[0][i]]+':'+'{:.2f}'.format(scr),xy=b[0:2], fontsize=15, color=color[0])
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(ofile)
        plt.close()


def plot_stats(df,folder):
    parameters=['WIDE_ANG','MASS_CENTER_RADIUS','MASS_CENTER_ANG','APEX_RADIUS','APEX_ANG']
    fa=0
    df["DATE_TIME"]= pd.to_datetime(df["DATE_TIME"])
    #max,cpa and min angles graphic
    x = []  
    max_ang_y = []  
    min_ang_y = []
    cpa_ang_y = []  

    for idx, row in df.iterrows():
        date_time = row['DATE_TIME']
        max_ang = np.degrees(row['MAX_ANG'])
        min_ang = np.degrees(row['MIN_ANG'])
        cpa_ang = np.degrees(row['CPA_ANG'])
        x.append(date_time)
        max_ang_y.append(max_ang)
        min_ang_y.append(min_ang)
        cpa_ang_y.append(cpa_ang)

    fig, ax = plt.subplots()
    ax.scatter(x, max_ang_y, color='red', label='max_ang')
    ax.scatter(x, min_ang_y, color='blue', label='min_ang')
    ax.scatter(x, cpa_ang_y, color='green', label='cpa_ang')
    ax.set_xlabel('Date and hour')
    ax.set_ylabel('Angles')
    ax.set_title('Dispersion graphic of max, cpa and min angles')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(data_dir+folder+"/stats/"+"max_cpa_min_angles.png") 

    #mass center coordinates graphic
    cm_y_list = []
    cm_x_list = []  

    for idx, row in df.iterrows():
        
        cm_x = row['MASS_CENTER_X']
        cm_y = row['MASS_CENTER_Y']
        cm_x_list.append(cm_x)
        cm_y_list.append(cm_y)

    fig1, ax1 = plt.subplots()
    ax1.scatter(cm_x_list, cm_y_list, color='red', label='max_ang')
    ax1.set_xlabel('x coordinates')
    ax1.set_ylabel('y coordinates')
    ax1.set_title('Dispersion graphic of mass center coordinates')
    ax1.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig1.savefig(data_dir+"/"+folder+"/stats/"+"mass_center_coordinates.png") 
    
    #generates graphics for the parameters
    for i in parameters:
        a=[]
        for idx, row in df.iterrows():
            if i.endswith("ANG"):        
                b = np.degrees(row[i])
            else:
                b = row[i]
            a.append(b)
        fig2, ax2 = plt.subplots()
        ax2.scatter(x, a, color='red', label=str(i.lower()))
        ax2.set_xlabel('Date and hour')
        ax2.set_title('Dispersion graphic of '+str(i.lower()))
        ax2.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig2.savefig(data_dir+"/"+folder+"/stats/"+str(i.lower())+".png") 
    fa=fa+1




# def plot_to_png(ofile,image_path,orig_img, masks,imsize, title=None, labels=None, boxes=None, scores=None):
#     """
#     Plot the input images (orig_img) along with the infered masks, labels and scores
#     in a single image saved to ofile
#     """    
#     mask_threshold = 0.5 # value to consider a pixel belongs to the object
#     #scr_threshold = 0.3 # only detections with score larger than this value are considered
#     color=['r','b','g','k','y']
#     obj_labels = ['Occ', 'CME','N/A','N/A']
#     cmap = mpl.colors.ListedColormap(color)  
#     nans = np.full(np.shape(orig_img[0]), np.nan)
    
#     ang_list=[]
#     for i in range(len(masks[0])):
#         print("Mask "+str(i)+" of "+str(len(masks[0])-1))
#         if labels[0][i]==2:
#             pol=rec2pol(masks[0][i],scores[0][i],labels[0][i],boxes[0][i],imsize=imsize)
#             if (pol is not None):
#                 pol_mask=pol[0]
#                 center= pol[1]
#                 masked=pol[2]
#                 pts=pol[3]
#                 #takes the min and max angles and calculates cpa and wide angles
#                 angles = [s[1] for s in pol_mask]
#                 ang=[]
#                 points=[]
#                 if len(angles)>0:
#                     min_ang = min(angles)
#                     max_ang = max(angles)
#                     cpa_ang=(max_ang+min_ang)/2
#                     wide_ang=(max_ang-min_ang)
                    

#                     #calculates diferent angles an parameters
                    
#                     date_time=datetime.strptime(f[:-6],"%Y%m%d_%H%M%S")
#                     cm_mask = nans.copy()
#                     cm_mask[:, :][masks[0][i] > mask_threshold] = 1  
#                     #cm_mask = np.where(masked == 0, 1, masked)#accomodates values of the mask to be able to calculate the center of mass with the following function
#                     cm = center_of_mass(np.nan_to_num(cm_mask))#returns center of mass cordinates in the form (y,x)
#                     center_x=imsize[0]/2
#                     center_y=imsize[1]/2
#                     x_dist = (cm[1]-center_x)
#                     y_dist = (cm[0]-center_y)
#                     cm_dist= math.sqrt(x_dist**2 + y_dist**2)
#                     cm_ang=np.arctan2(-y_dist,x_dist)#+np.radians(270)
#                     distance = [s[0] for s in pol_mask]
#                     distance_abs= max(distance, key=abs)
#                     idx_dist = distance.index(distance_abs)
#                     apex_dist= distance[idx_dist] 
#                     apex_ang=angles[idx_dist]
#                     apex_coord=pts[idx_dist]
#                     ang.append([min_ang,max_ang,cpa_ang,cm_ang])
                    
#                     #transforms angles to degrees for display purposes
#                     angulo_max=np.degrees(max_ang)
#                     angulo_min=np.degrees(min_ang)
#                     angulo_cpa=np.degrees(cpa_ang)
#                     title=str(angulo_max)+"/"+str(angulo_cpa)+"/"+str(angulo_min)

#                     #obtains end points and direction for the lines
#                     for j in ang[0]:
#                         vector=np.array([np.cos(j), np.sin(j)])
#                         start_point = center
#                         x_end = start_point[0] + (imsize[0]/2) * vector[0]
#                         y_end = start_point[1] + (imsize[1]/2) * vector[1]
#                         x_points = np.array([start_point[0], x_end])
#                         y_points = np.array([start_point[1], y_end])
#                         points.append([x_points,y_points])

#                     #saves the image
#                     # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#                     # axs[0].imshow(orig_img[0], vmin=0, vmax=1, cmap='gray')
#                     # axs[0].axis('off')
#                     # axs[1].imshow(orig_img[0], vmin=0, vmax=1, cmap='gray')      
#                     # axs[1].axis('off')        
#                     # if boxes is not None:
#                     #     #print("boxes ok")
#                     #     if scores is not None:
#                     #         scr = scores[0][i]
#                     #     else:
#                     #         scr = 0   
#                     #         if scr > scr_threshold:
#                     #                 #print("score is ok")             
#                     #                 masked = nans.copy()
#                     #                 masked[:, :][masks[0][i] > mask_threshold] = 0              
#                     #                 axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1) # add mask
#                     #                 axs[i+1].plot(points[0][0], 512-points[0][1], color='blue', label='Recta min')
#                     #                 axs[i+1].plot(points[1][0],512-points[1][1] , color='orange', label='Recta max')
#                     #                 axs[i+1].plot(points[2][0], 512-points[2][1], color='green', label='Recta cpa')
#                     #                 axs[i+1].plot(points[3][0], 512-points[3][1], color='purple', label='Recta centro de masa')
#                     #                 axs[i+1].scatter(cm[1],cm[0], color='purple', marker='o')
#                     #                 axs[i+1].scatter(apex_coord[0],apex_coord[1], color='cyan', marker='o')
#                     #                 if labels is not None:
#                     #                     axs[i+1].annotate(obj_labels[labels[0][i]]+':'+'{:.2f}'.format(scr),xy=b[0:2], fontsize=15, color=color[0])
#                     #         else:
#                     #              print("scr pequeño")
#                     # else:
#                     #      print("boxes is none")
#                     # fig.suptitle(title)
#                     # plt.tight_layout()
#                     # #fig.show()
#                     # plt.savefig(ofile)
#                     # plt.close()
#                     breakpoint()
#                     ang_list.append([image_path,i,date_time,scores[0][i], min_ang, max_ang, cpa_ang, wide_ang, cm[1], cm[0],cm_dist, cm_ang, apex_dist, apex_ang])          
            
#         else:
#             print("Label different than 1")

#     return ang_list         
        
######################################################################## MAIN #################################################################################################

data_dir="/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat_L1/cor2_a/"
image_dir="/gehme/data/stereo/secchi/L0/a/img/cor2/"
imsize=[512,512] #shape of the mask image
scr_threshold = 0.25 
mask_threshold = 0.5 # value to consider a pixel belongs to the object
fa=0

folders = np.sort(os.listdir(data_dir))

for folder in folders:
    print("Working on folder "+folder+", folder "+str(fa)+" of "+str(len(folders)-1))
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
                
            f=file[:-11]
            img=read_fits(img_path)
            folder_path = data_dir+folder
            final_path = os.path.join(folder_path, "stats")
            if not os.path.exists(final_path):
                os.makedirs(final_path)
            opath=final_path+"/"+str(f)+"_stats.png"
            ang_list=_compute_mask_prop(imsize,img_path,[img],f,scr_threshold,mask_threshold,mask,scores,labels,boxes)

            #plot_to_png(opath,ang_list,scr_threshold,[mask],[labels],[boxes],[scores],f)
                           
            if ang_list!= None:
                ang_data.extend(ang_list)
    fa=fa+1
    columnas = ['PATH',"MASK_LABEL",'DATE_TIME', "SCORE",'MIN_ANG','MAX_ANG','CPA_ANG','WIDE_ANG','MASS_CENTER_X','MASS_CENTER_Y','MASS_CENTER_RADIUS','MASS_CENTER_ANG','APEX_RADIUS','APEX_ANG'] 
    
    df = pd.DataFrame(ang_data, columns=columnas)
    df.to_csv(final_path+'/'+folder+'_stats', index=False)
    plot_stats(df,folder)
