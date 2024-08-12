import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import matplotlib as mpl
import math
import sunpy.sun.constants as sun_const
from datetime import datetime
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.dates as mdates
import pickle
from scipy import stats
mpl.use('Agg')
from astropy.io import fits
import logging

__author__ = "Francisco Iglesias"
__copyright__ = "Grupo de Estudios en Heliofisica de Mendoza - https://sites.google.com/um.edu.ar/gehme"
__version__ = "0.2"
__maintainer__ = "Francisco Iglesias | Diego Lloveras"
__email__ = "franciscoaiglesias@gmail.com | lloverasdiego@gmail.com"

def quadratic(t,a,b,c):
    return a*t**2. + b*t + c

def quadratic_error(p, x, y, w):
    return w*(quadratic(x, *p) - y)

def quadratic_velocity(t, a, b):
    return 2 * a * t + b

def linear(t,a,b):
    return a*t + b

def linear_error(p, x, y, w):
    return w*(linear(x, *p) - y)

class neural_cme_segmentation():
    '''
    Class to perform CME segmentation using Mask R-CNN

    Based on Pytorch vision model MASKRCNN_RESNET50_FPN from the paper https://arxiv.org/pdf/1703.06870.pdf

    see also:
    https://pytorch.org/vision/0.12/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    https://towardsdatascience.com/train-mask-rcnn-net-for-object-detection-in-60-lines-of-code-9b6bbff292c3
    '''
    def __init__(self, device, pre_trained_model=None, version='v4', imsize=[512,512], logger=None):
        '''
        Initializes the model
        device: device to use for training and inference
        pre_trained_model: path to model parameters to use for inference
        version: version of the model to use. Options are 'v3' and 'v4'
        '''
        self.device = device
        self.pre_trained_model = pre_trained_model    
        self.imsize = imsize   
        self.version = version 
        # model param
        if  self.version  == 'v3':
            # First reasonably performing version of the model
            self.num_classes = 3 # background, CME, occulter
            # number of trainable layers in the backbone resnet, int in [0,5] range. Specifies the stage number.
            # Stages 2-5 are composed of 6 convolutional layers each. Stage 1 is more complex
            # See https://arxiv.org/pdf/1512.03385.pdf
            self.trainable_backbone_layers = 3
        if  self.version  == 'v4':
            # Second version of the model. Trainning more layers in the backbone
            self.labels=['Background','Occ','CME'] # labels for the different classes
            self.num_classes = 3 # background, CME, occulter
            self.trainable_backbone_layers = 4
            self.mask_threshold = 0.60 # value to consider a pixel belongs to the object
            self.scr_threshold = 0.51 # only detections with score larger than this value are considered
        if  self.version  == 'v5':
            # Third version of the model. Trainning with only 2 classes, CME and background, and all the backbone layers
            # Also updated the normalization routine
            self.labels=['Background','CME'] # labels for the different classes
            self.num_classes = 2 # background, CME
            self.trainable_backbone_layers = 5
            self.mask_threshold = 0.60 # value to consider a pixel belongs to the object
            self.scr_threshold = 0.51 # only detections with score larger than this value are considered            
        # innitializes the model
        self.model=torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT', trainable_backbone_layers=self.trainable_backbone_layers) 
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features 
        self.model.roi_heads.box_predictor=FastRCNNPredictor(self.in_features,num_classes=self.num_classes)
        if self.pre_trained_model is not None:
            model_param = torch.load(self.pre_trained_model, map_location=device)
            self.model.load_state_dict(model_param)      
        self.model.to(self.device)
        # uses the input logger
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())
        return

    def _apply_linear_multiplier(self, image):
        """
        Applies a linear multiplier to an image, increasing outwards from the center.
        Args:
            image: A NumPy array representing the image (grayscale or color).
        Returns:
            A NumPy array representing the modified image.
        """
        # Get image dimensions
        height, width = image.shape[:2]
        # Create a meshgrid representing coordinates from center outwards
        y, x = np.ogrid[0:height, 0:width]
        center_x = width // 2
        center_y = height // 2
        radius = max(center_x, center_y)  # Consider the larger dimension for radius
        # Calculate normalized distance from center (0 at center, 1 at edges)
        distance = np.sqrt(((x - center_x) ** 2) + ((y - center_y) ** 2)) #/ radius
        # Define linear multiplier function (adjust slope and offset as needed)
        multiplier = 1. + (0.1/np.max(distance)) * distance #0.15
        #aumenta 10% desde el centro hasta la esquina. entonces en 256 pixels seria 10% * cos(45)
        # Apply element-wise multiplication with broadcasting
        modified_image = np.multiply(image, multiplier)
        
        return modified_image,radius,distance, multiplier

    def normalize(self, image, excl_occulter_level='auto', sd_range=2, norm_limits=[None, None], increase_contrast=False, 
                  median_kernel=3, plot_histograms=False, histogram_names='', path=''):
        '''
        Normalizes the input image to the 0-1 range. 
        Note that only the first channel of the image is used!
        
        sd_range: number of standard deviations around the mean to use for normalization 
        norm_limits: if not None, the image is first truncated to the given limits
        increase_contrast: if True, increases the contrast of the normalized image radially.
        plot_histograms: if True, plots the histograms of the original and normalized images
        histogram_names: name of the histogram to save 
        path: dir path to save the histograms
        excl_occulter_level: if not None, this level is excluded from the mean and sd computation used to normalize
                             Use 'auto' to exclude the most frequent value in the image
        median_kernel: if not None, the image is smoothed using a median kernel of the given size. Min is 3
        '''
        # pre clipping
        if norm_limits[0] is not None:
            image[image < norm_limits[0]] = 0
        if norm_limits[1] is not None:
            image[image > norm_limits[1]] = 1
        
        #using only first channel
        oimage = image[:,:,0].copy()

        #median kernel
        if median_kernel is not None:
            oimage = cv2.medianBlur(oimage, median_kernel)

        #exclude occulter values at excl_occulter_level
        if excl_occulter_level is not None:
            if excl_occulter_level == 'auto':
                # finds the most frequent integer value in the image
                occ_indx = (oimage != stats.mode(oimage.flatten(),keepdims=True)[0][0])
            else:
                occ_indx = (oimage != excl_occulter_level) & (oimage != excl_occulter_level-1)
            m = np.nanmean(oimage[occ_indx])
            sd = np.nanstd(oimage[occ_indx])
        else:
            m = np.nanmean(oimage)
            sd = np.nanstd(oimage)
        
        if plot_histograms:
            plt.hist(oimage, bins=50, range=(np.percentile(oimage,1),np.percentile(oimage,99)), color='blue', alpha=0.7)
            plt.title('Histograma de valores de la imagen')
            plt.xlabel('Valor')
            plt.ylabel('Frecuencia')
            plt.savefig(path+"histograms/orig/"+histogram_names+".png")
            plt.close()

        #normalizing
        oimage = (oimage - m + sd_range * sd) / (2 * sd_range * sd)

        #If True, increase contrast of the Normalized image radialy, above a specific radius.
        if increase_contrast:
            modified_image,radius,distance, multiplier = self._apply_linear_multiplier(oimage[:,:,0])
            m1  = np.mean(oimage)
            sd1 = np.std(oimage)
            binary_image = np.where(oimage[:,:,0] > m1+sd1/15, 1, 0) #0.51
            aaa=np.where(distance > 100., 1, 0)
            oimage[np.logical_and(binary_image == 1, aaa == 1)] = np.multiply(oimage, multiplier)[np.logical_and(binary_image == 1, aaa == 1)]
        
        #clipping to 0-1 range
        oimage[oimage > 1]=1
        oimage[oimage < 0]=0

        #checking for nan values and replace for the mean value
        mean = np.mean(oimage)
        if np.isnan(mean):
            self.logger.warning('Found nan values in the normalized image. Replacing with the mean value')
            non_nan_mean = np.nanmean(oimage) # mean of the non-nan values        
            if str(non_nan_mean).isdigit() == False:
                self.logger.warning('Full nan, se pudre la momia')
                
            oimage = np.nan_to_num(oimage, nan=non_nan_mean)

        if plot_histograms:
            normalized_asd = oimage.flatten()
            plt.hist(normalized_asd, bins=50, range=(np.percentile(normalized_asd, 5),  np.percentile(normalized_asd, 95)), color='blue', alpha=0.7)
            plt.hist(normalized_asd, bins=50, range=(0,1), color='blue', alpha=0.7)
            plt.title('Histograma de valores de la imagen Normalizados')
            plt.xlabel('Valor')
            plt.ylabel('Frecuencia')
            plt.savefig(path+"histograms/resize/"+histogram_names+".png")
            plt.close()
        
        return  np.dstack([oimage,oimage,oimage])
 
    def train(self, opt_type='adam' , lr=1e-5):
        '''
        Sets optimizer type and train mode
        '''
        if opt_type == 'adam':
            self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr) # optimization technique that comes under gradient decent algorithm    
        else:
            self.logger.error('Optimizer type not recognized.')
            return None
        self.model.train()        
        return 
    
    def resize(self, img):
        '''
        Resizes the input image to the given size
        '''
        return cv2.resize(img, self.imsize, cv2.INTER_LINEAR)
    
    def get_mask(self, img, hdr,replace_value=None):
        '''
        devuelve una imagen con el oculter!
        Based on get_smask.pro and sccrorigin.pro
        OBS: lo mismo puede utilizarse para euvi, cor1 y hi2.
        '''
        if hdr["detector"] == "COR2":
            if hdr["obsrvtry"] == "STEREO_A":
                #file = "/usr/local/ssw/stereo/secchi/calibration/cor2A_mask.fts" #path original de ssw
                file = "/gehme-gpu/projects/2023_eeggl_validation/repo_diego/2020_gcs_with_ml/instrument_mask/cor2A_mask.fts"
                r1col = 129
                r1row = 1

            if hdr["obsrvtry"] == "STEREO_B":
                #file = "/usr/local/ssw/stereo/secchi/calibration/cor2B_mask.fts"
                file = "/gehme-gpu/projects/2023_eeggl_validation/repo_diego/2020_gcs_with_ml/instrument_mask/cor2B_mask.fts"
                r1col = 1
                r1row = 79

            if hdr["rectify"] == "STR":
                #check sccrorigin.pro
                r1col = 51
                r1row = 1
                self.logger.warning("Please take a look at this image!!")

        smask = fits.open(file)[0].data
        smask = np.flip(smask, axis=0)
        fullm = np.zeros((2176,2176))
        xy = [r1col,r1row]
        
        fullm[xy[0]-1:xy[0]-1+smask.shape[0],xy[1]-1:xy[1]-1+smask.shape[1]] = smask

        #Chequear esta rotacion si la imagen es posterior al 2015-05-19!!!
        #De hecho, en IDL funciona hacer gt "string"???

        #if (hdr.date_obs ge '2015-05-19') and (hdr.detector ne 'EUVI'):
        #    fullm = rotate(fullm, 2)

        #mask = rebin(fullm[hdr.r1col-1:hdr.r2col-1,hdr.r1row-1:hdr.r2row-1], hdr.naxis1,hdr.naxis2)
        mask_full = fullm[hdr["r1col"]-1:hdr["r2col"]-1,hdr["r1row"]-1:hdr["r2row"]-1]
        dim = (hdr["naxis1"],hdr["naxis2"]) #si naxis no corresponde al shape de la imagen?
        dim = (img.shape[0],img.shape[1])
        mask = cv2.resize(mask_full, dim, interpolation=cv2.INTER_LINEAR)
        
        #Los valores deben ser 0 y 1
        if replace_value is None:
            img[mask!=1] = 0#np.nan #np.mean(img)
        else:
            img[mask!=1] = replace_value
        
        return img

    
    def mask_occulter(self, img, occulter_size,centerpix,repleace_value=None):
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

    def mask_occulter_external(self, img, occulter_size,centerpix,repleace_value=None):
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
            img[mask!=1] = 0#p.nan #np.mean(img)
        else:
            img[mask!=1] = repleace_value
        return img

    def infer(self, img, model_param=None, resize=True, occulter_size=None,occulter_size_ext=None,centerpix=None,getmask=None,hdr=None,repleace_value=None,histogram_names='',path='',increase_contrast=None):
        '''        
        Infers a cme segmentation mask in the input coronograph image (img) using the trained R-CNN
        model_param: model parameters to use for inference. If None, the model parameters given at initialization are used
        resize: if True, the input image is resized to the size of the training images
        occulter_size: size of an artifitial occulter added to the image. If 0, the occulter is not masked out. Mask pix within the occulter are eliminated
        '''    
        #loads model
        if model_param is not None:
            self.model.load_state_dict(model_param) 
        elif self.pre_trained_model is None:
            os.error('No model parameters given')
        self.model.eval() #set the model to evaluation state

        #inference
        if img.ndim == 2:
            images = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            images = img.copy()
        if resize and (images.shape[0] != self.imsize[0] or images.shape[1] != self.imsize[1]):
            images = self.resize(images)   
        
        #cambiar las 2 de abajo por una flag que se llame get_mask
        if occulter_size_ext != None:
            images = self.mask_occulter_external(images,occulter_size_ext,centerpix,repleace_value)
        
        if occulter_size != None:
            images = self.mask_occulter(images, occulter_size,centerpix,repleace_value)    

        if getmask:
            images = self.get_mask(images, hdr)

        #if np.mean(images[:,:,0]) == 0:
        #    breakpoint()
        images = self.normalize(images,histogram_names=histogram_names,path=path,increase_contrast=increase_contrast) # normalize to 0,1
        
        oimages = images.copy()                                   
        images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
        images=images.swapaxes(1, 3).swapaxes(2, 3)
        images = list(image.to(self.device) for image in images)
        
        with torch.no_grad(): #runs the image through the net and gets a prediction for the object in the image.
            pred = self.model(images)

        # returns the network input image along with the infered masks, labels and scores
        orig_img =  np.array(oimages)[:,:,0]
        nmasks = len(pred[0]['masks'])
        all_lbl = []
        all_boxes = []
        all_scores = []    
        all_masks = [] 
        if nmasks > 0:
            for i in range(nmasks):
                scr=pred[0]['scores'][i].detach().cpu().numpy()
                all_scores.append(scr)
                lbl = pred[0]['labels'][i].detach().cpu().numpy()
                all_lbl.append(lbl)
                msk=pred[0]['masks'][i,0].detach().cpu().numpy()
                # eliminates px within the occulter for non -OCC masks
                if self.version == 'v4':
                    if occulter_size != None and lbl == self.labels.index('Occ'):
                        msk = self.mask_occulter(msk, occulter_size,centerpix, repleace_value=0)
                all_masks.append(msk)                
                box = pred[0]['boxes'][i].detach().cpu().numpy()
                all_boxes.append(box)              
        else:
            os.error('Found no maks in the current image')
        return orig_img, all_masks, all_scores, all_lbl, all_boxes
    
    def test_mask(self, img, target, mask_threshold=None, model_param=None, resize=True, occulter_size=0):
        """
        Makes an inference on img and return the mask that has the smallest loss wrt target binary mask
        mask_threshold: threshold for the predicted box pixels to be considered as part of the mask
        """
        if mask_threshold is not None:
            self.mask_threshold = mask_threshold
        orig_img, all_masks, all_scores, all_lbl, all_boxes = self.infer(img, model_param=model_param, resize=resize, occulter_size=occulter_size)
        if len(all_masks) == 0:
            self.logger.warning('Warning, no masks found in the image')
            return None
        # compute loss for all masks
        all_loss = []
        for i in range(len(all_masks)):
            msk = all_masks[i]
            msk[msk > self.mask_threshold] = 1
            msk[msk <= self.mask_threshold] = 0
            all_loss.append(np.sum(np.abs(msk - target))/np.sum(target))
        # return the mask with the smallest loss
        imin = np.argmin(all_loss)
        return orig_img, all_masks[imin], all_scores[imin], all_lbl[imin], all_boxes[imin], all_loss[imin]
        

    def _rec2pol(self, mask, center=None):
        '''
        Converts the x,y mask to polar coordinates
        Only pixels above the mask_threshold are considered
        TODO: Consider arbitrary image center, usefull in case of Cor-2B images.
        NO FUNCIONA BIEN EN EL CASO DE COR2B y en relacion al plot2. CHEQUEAR
        '''
        nans = np.full(self.imsize, np.nan)
        pol_mask=[]
        #creates an array with zero value inside the mask and Nan value outside             
        masked = nans.copy()
        masked[:, :][mask > self.mask_threshold] = 0   
        #calculates geometric center of the image
        height, width = masked.shape
        if center is None:
            #case center is not defined. Calculates geometric center of the image
            center_x = width / 2
            center_y = height / 2
        else:
            #case center is defined as input. 
            center_x = center[0]
            #since occ_center is given using images coordinates, we need to invert the y axis.
            center_y = height-center[1]
            #center_y = center[1]
        #calculates distance to the point and the angle for the positive y axis
        for x in range(width):
            for y in range(height):
                value=masked[x,y]
                if not np.isnan(value):
                    x_dist = (x-center_x)
                    y_dist = (y-center_y)
                    distance= np.sqrt(x_dist**2 + y_dist**2)
                    #Si y es vertical y x es horizontal, entonces angle positivo calculado con respecto a y positivo, en forma antihoraria.
                    angle = np.arctan2(x_dist,y_dist)
                    if angle<0:
                        angle+=2*np.pi
                    pol_mask.append([distance,angle])
        return pol_mask

    def _area_score(self, mask):
        '''
        Similar to rec2pol, but returns the area of the mask related to the total images size.
        '''
        area_score=0.
        nans = np.full(self.imsize, np.nan)
        #creates an array with zero value inside the mask and Nan value outside             
        masked = nans.copy()
        masked[:, :][mask > self.mask_threshold] = 0   
        #calculates geometric center of the image
        height, width = masked.shape
        #calculates the amount of pixels corresponding to the mask
        for x in range(width):
            for y in range(height):
                value=masked[x,y]
                if not np.isnan(value):
                    area_score = area_score + 1.
        area_score = area_score/(height*width)
        return area_score

    def _compute_mask_prop(self, masks, scores, labels, boxes, plate_scl=1, filter_halos=False, occulter_size=0,centerpix=None, percentiles=[5,95]):    
        '''
        Computes the CPA, AW and apex radius for the given 'CME' masks.
        If the mask label is not "CME" and if is not a Halo and filter_halos=True, it returns None
        plate_scl: if defined the apex radius is scaled based on the plate scale units of the image. If 0, no filter is applied
                    if plate scale = hdr['cdelt1']/hdr['rsun'] then apex is in units of rsun.
        filter_halos: if True, filters the masks with boxes center within the occulter size, and wide_ angle > MAX_WIDE_ANG
        '''
        self.max_wide_ang = np.radians(270.) # maximum angular width [deg]

        prop_list=[]
        for i in range(len(masks)):
            if filter_halos:
                box_center = np.array([boxes[i][0]+(boxes[i][2]-boxes[i][0])/2, boxes[i][1]+(boxes[i][3]-boxes[i][1])/2])
                if centerpix is not None:
                    distance_to_box_center = np.sqrt((centerpix[0]/2-box_center[0])**2 + (centerpix[1]/2-box_center[1])**2)
                else:
                    distance_to_box_center = np.sqrt((self.imsize[0]/2-box_center[0])**2 + (self.imsize[1]/2-box_center[1])**2)
                if distance_to_box_center < 0.8 * occulter_size:
                    halo_flag = False
                else:
                    halo_flag = True
            else:
                halo_flag = True
            
            if self.labels[labels[i]]=='CME' and scores[i]>self.scr_threshold and halo_flag: 
                pol_mask=self._rec2pol(masks[i],center=centerpix)
                
                if (pol_mask is not None):            
                    #takes the min and max angles and calculates cpa and wide angles
                    angles = [s[1] for s in pol_mask]
                    if len(angles)>0:
                        # checks for the case where the cpa is close to 0 or 2pi
                        if np.max(angles)-np.min(angles) >= 0.9*2*np.pi:
                            angles = [s-2*np.pi if s>np.pi else s for s in angles]
                        aw_min = np.percentile(angles, percentiles[0])
                        aw_max = np.percentile(angles, percentiles[1])

                        #calculate angular width
                        #cut list of values (angles) between two values, aw_min and aw_max
                        angles_between_percentiles = [s for s in angles if s >= aw_min and s <= aw_max]
                        cpa_ang= np.median(angles_between_percentiles)
                        if cpa_ang < 0:
                            cpa_ang += 2*np.pi
                        wide_ang=np.abs(aw_max-aw_min)
                        #calculates the distance to the apex
                        distance = [s[0] for s in pol_mask]
                        distance_abs= max(distance, key=abs)
                        idx_dist = distance.index(distance_abs)

                        #angle corresponding to the apex_dist position
                        angulos = [s[1] for s in pol_mask]
                        apex_angl = angulos[idx_dist] 
                        apex_dist = distance[idx_dist] * plate_scl
                        #caclulates the area of the mask in % of the total image area
                        area_score = self._area_score(masks[i])
                        #calculate apex distances as a percentil 98, and the corresponding angles
                        apex_dist_percentile = np.percentile(distance, 98) 
                        apex_dist_per = [d * plate_scl for d,a in zip(distance,angulos) if d >= apex_dist_percentile and d<=apex_dist_percentile+0.5]
                        apex_angl_per = [a for d,a in zip(distance,angulos) if d >= apex_dist_percentile and d<=apex_dist_percentile+0.5]
                        #breakpoint()
                        if filter_halos:
                            if wide_ang < self.max_wide_ang:
                                prop_list.append([i,float(scores[i]),cpa_ang, wide_ang, apex_dist, apex_angl, aw_min, aw_max, area_score,apex_dist_per,apex_angl_per])  
                            else:
                                prop_list.append([i,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, [np.nan], [np.nan]])
                        else:
                            prop_list.append([i,float(scores[i]),cpa_ang, wide_ang, apex_dist, apex_angl, aw_min, aw_max, area_score,apex_dist_per,apex_angl_per])
                else:
                    prop_list.append([i,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, [np.nan], [np.nan]])
            else:
                prop_list.append([i,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, [np.nan], [np.nan]])
        
        if len(masks) == 0:
            self.logger.warning('No masks found in the image')
            prop_list.append([0,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, [np.nan], [np.nan]])
        return prop_list

    def _plot_mask_prop(self, dates, param, opath, ending='_filtered', x_title='Date and hour', style='*', save=True):
            '''
            plots the evolution of the cpa, aw and apex radius for all the masks found and the filtered ones
            dates: list of datetime objects with the date of each event
            param: 3d list with the mask properties for all masks found in each image. 
            Each maks's has the following prop: [id,float(scores[i]),cpa_ang, wide_ang, apex_dist]
            '''
            self.mask_prop_labels=['MASK ID', 'SCORE','CPA_ANG','WIDTH_ANG','APEX_RADIUS']

            self.logger.info('Plotting masks properties to '+opath)
            # repeat dates for all masks
            if ending == '_all':
                x = []
                for i in range(len(dates)):
                    x.append([dates[i]]*len(param[i]))
                x = np.array([i for j in x for i in j])
            elif ending == '_filtered':                
                x = dates.copy()
            else:
                self.logger.warning('Unrecognized value for ending parameter')
                return None
            for par in range(len(param[0][0])):
                cparam = np.array([i[par] for j in param for i in j])           
                y_title = self.mask_prop_labels[par]
                if y_title.endswith("ANG"):        
                    b = [np.degrees(i) for i in cparam]
                else:
                    b = cparam
                fig, ax = plt.subplots()
                ax.plot(x, b, style, label=y_title)
                ax.set_xlabel(x_title)
                ax.set_title(y_title)
                ax.legend()
                ax.grid('both')
                plt.xticks(rotation=45)
                plt.tight_layout()
                if save:
                    os.makedirs(opath, exist_ok=True)
                    fig.savefig(opath+'/'+str.lower(y_title)+ending+".png") 
                    plt.close()
                else:
                    return fig, ax
                



    def _plot_mask_prop2(self, df, opath, ending='_filtered', x_title='Date and hour', style='*', save=True,save_pickle=False):
            '''
            plots the evolution of the cpa, aw and apex radius for all the masks found and the filtered ones
            dates: list of datetime objects with the date of each event
            param: 3d list with the mask properties for all masks found in each image. 
            Each maks's has the following prop: [id,float(scores[i]),cpa_ang, wide_ang, apex_dist]
            '''
            parameters= ['CPA', 'MASK_ID', 'SCR', 'AW', 'APEX', 'CME_ID','AW_MAX','AW_MIN','APEX_ANGL','AREA_SCORE','APEX_DIST_PER','APEX_ANGL_PER']
            colors=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w'] 
        
            df["DATE_TIME"]=pd.to_datetime(df["DATE_TIME"], format='%H:%M')
            dict={}
            #df["DATE_TIME"]=df["DATE_TIME"].dt.strftime('%H:%M')
            optimal_n_clusters=int(df["CME_ID"].max())+1
            for par in parameters:
                fig, ax = plt.subplots()
                dt_list=[]
                for k in range(optimal_n_clusters):
                    cluster_data=df.loc[(df["CME_ID"])==k]
                    x_points=cluster_data["DATE_TIME"].tolist()
                    y_points=cluster_data[par].tolist() 
                    y_title = par
                    dt_list.extend(x_points)
                    if par == 'APEX_DIST_PER' or par == 'APEX_ANGL_PER':
                        y_points_mean = [np.mean(i) for i in y_points]
                        ax.plot(x_points, y_points_mean, style,color=colors[k])
                    else:
                        ax.plot(x_points, y_points, style,color=colors[k])
                
                hours = [str(timestamp.time()) for timestamp in dt_list]
                hours1 = [datetime.strptime(timestamp, "%H:%M:%S") for timestamp in hours]
                hours2 = [timestamp.strftime("%H:%M") for timestamp in hours1]

                ax.set_title(y_title)    
                ax.set_xlabel(x_title)
                plt.xticks(dt_list,hours2,rotation=90)
                plt.grid()
                plt.tight_layout()
                if save:
                    os.makedirs(opath, exist_ok=True)
                    fig.savefig(opath+'/'+str.lower(y_title)+ending+".png")
                    plt.close()
                else:
                    return fig, ax 
               
            
            
        
    def _filter_param(self, in_x, in_y, error_func, fit_func, in_cond, criterion, percentual=True, weights=[2]):
        '''
        Deletes the points that are more than criterion from the fit function given by fit_func
        in_x: timestamps of the points to be filtered
        in_y: y values of the points to be filtered
        error_func: function to calculate the error between the fit and the data
        fit_func: function to fit the data
        in_cond: initial condition for the fit
        criterion: maximum distance from the fit to consider a point as ok
        percentual: if True, the criterion is a percentage of the y value
        weights: data points weights for the fit. Must be a cevtor of len len(in_x) or int. 
        If It's a scalar int gives more weight to the last weights dates bcause CME is supoused to be larger and better defined
        '''
        #deletes points with y is nan
        ok_ind = np.where(~np.isnan(in_y))[0]
        x = in_x[ok_ind]
        y = in_y[ok_ind]

        # fit the function using least_squares
        used_weights = np.ones(len(x))
        if len(weights) > 1:
            used_weights = weights
        fit=least_squares(error_func, in_cond , method='lm', kwargs={'x': x-x[0], 'y': y, 'w': used_weights}) # first fit to get a good initial condition
        fit=least_squares(error_func, fit.x, loss='soft_l1', kwargs={'x': x-x[0], 'y': y, 'w': used_weights}) # second fit to ingnore outliers    

        #calculate the % distance from the fit
        if percentual:
            dist = np.abs(fit_func(x-x[0], *fit.x)-y)/y
        else:
            dist = np.abs(fit_func(x-x[0], *fit.x)-y)

        #get the index of the wrong points
        ok_ind2 = np.where(dist<criterion)[0]

        #get the index of the ok points
        ok_ind = ok_ind[ok_ind2]

        return ok_ind
    
    def _select_mask(self,axis,k, in_x, in_y, error_func, fit_func, in_cond, weights=[2]):
        '''
        Fits diferent tipe of functions to the giving xy points, according to the fit function given (fit_func). 
        Also calculates the distance from the original data to the fitted function.
        in_x: timestamps of the points to be filtered
        in_y: y values of the points to be filtered
        error_func: function to calculate the error between the fit and the data
        fit_func: function to fit the data
        in_cond: initial condition for the fit
        criterion: maximum distance from the fit to consider a point as ok
        percentual: if True, the criterion is a percentage of the y value
        weights: data points weights for the fit. Must be a cevtor of len len(in_x) or int. 
                If It's a scalar int gives more weight to the last weights dates bcause CME is supoused to be larger and better defined
        Returns the distance (in an array form), parabola (indacates the concavity, beeing this negative=True and positive=False) and axis(plots the fit function)
        '''
        colors=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w']
        vel_threshold=1 #km/s
        #deletes points with y is nan
        ok_ind = np.where(~np.isnan(in_y))[0]
        x = in_x[ok_ind]
        y = in_y[ok_ind]

        # fit the function using least_squares
        used_weights = np.ones(len(x))
        if len(weights) > 1:
            used_weights = weights
        fit=least_squares(error_func, in_cond , method='lm', kwargs={'x': x-x[0], 'y': y, 'w': used_weights}) # first fit to get a good initial condition
        fit=least_squares(error_func, fit.x, loss='soft_l1', kwargs={'x': x-x[0], 'y': y, 'w': used_weights}) # second fit to ingnore outliers    

        #calculate the % distance from the fit
        dist = np.abs(fit_func(x-x[0], *fit.x)-y)/y
        
        if fit_func == quadratic:
            a=fit.x[0]
            velocity = quadratic_velocity(x - x[0], *fit.x[:-1])
            average_velocity = np.median(velocity)
            if a<=0:
                parabola=True
            else:
                parabola=False    
        else:
            parabola=False
            average_velocity=False

        label = f"y={fit.x[0]:.2f}x+{fit.x[1]:.2f}\nR={np.corrcoef(x, y)[0,1]:.2f}"
        axis.plot(x, fit_func(x-x[0], *fit.x), color=colors[k], linestyle='--', label=label, linewidth=1.5)
       
        return dist,parabola,average_velocity,axis
    
    

    def _filter_masks2(self, dates, masks, scores, labels, boxes, mask_prop,plate_scl,opath,MAX_CPA_DIST,MIN_CPA_DIFF,MIN_CLUSTER_POINTS):
        '''
        Filters the masks by creating clusters based on the cpa and fitting functions to every cluster found, according to cpa, wa and apex_dist.
        The filter criterion is based on the minimal distances from the mask properties(cpa, wa and apex_dist) to the fitted functions, from wich the euclidian distance its calculated. 
        Also keeps only one mask per date per cluster.
        It returns min_error dataframe, containing all the filtered masks and their properties, including the CME_ID.
        '''
        colors=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w'] 

        #transforms inputs in a dataframe
        data=[]
        for i in range(len(dates)):
            for j in range(len(masks[i])):
                mask_id=mask_prop[i][j][0]
                scr=mask_prop[i][j][1]
                cpa = mask_prop[i][j][2]
                wa=mask_prop[i][j][3]
                date= dates[i]
                label=labels[i][j]
                box=boxes[i][j]
                mask=masks[i][j]
                apex = mask_prop[i][j][4]  #apex in same Units as calculated in compute_mask_prop (based on plate_scl)
                apex_angl = mask_prop[i][j][5]
                aw_min    = mask_prop[i][j][6]
                aw_max    = mask_prop[i][j][7]
                area_score= mask_prop[i][j][8]
                apex_dist_per = mask_prop[i][j][9]
                apex_angl_per = mask_prop[i][j][10]
                data.append((date,mask_id,scr,cpa,wa,apex,label,box,mask,apex_angl,aw_min,aw_max,area_score,apex_dist_per,apex_angl_per))
        df = pd.DataFrame(data, columns=["DATE_TIME","MASK_ID","SCR","CPA","AW","APEX","LABEL","BOX","MASK","APEX_ANGL","AW_MIN","AW_MAX", "AREA_SCORE","APEX_DIST_PER","APEX_ANGL_PER"])
        df["DATE_TIME"]=pd.to_datetime(df["DATE_TIME"])
        
        #Gets the optimal number of clusters for the data
        data= df["CPA"].values.reshape(-1, 1)
        silhouette_scores = []
        if len(data)<5:
            n_clusters_range = range(2, len(data))
        else: 
            n_clusters_range = range(2, 5) 
        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0,n_init=10)
            km_labels = kmeans.fit_predict(data)
            #Verify if every cluster has the minimum amount of points
            cluster_counts = pd.Series(km_labels).value_counts()
            condition = all(count > MIN_CLUSTER_POINTS for count in cluster_counts)
            if condition:
                silhouette_avg = silhouette_score(data, km_labels)
                silhouette_scores.append(silhouette_avg)

        #gets the optimal amount of clusters
        if len(silhouette_scores)>0:
            optimal_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
            #Adjust KMEANS to optimal cluster number
            kmeans = KMeans(n_clusters=optimal_n_clusters,random_state=0,n_init=10) #n_init=10 is relevant from sklearn 1.4+
            labels = kmeans.fit_predict(data)
            df['CME_ID'] = [int(i) for i in labels]
            
            clusters_median=[]
            for i in df['CME_ID'].unique():
                median=np.median(df.loc[df['CME_ID']==i,'CPA'])
                clusters_median.append(median)
            for j in range(len(clusters_median) - 1):
                for h in range(i + 1, len(clusters_median)):
                    diff = abs(clusters_median[j] - clusters_median[h])
                    if diff < MIN_CPA_DIFF:
                        if optimal_n_clusters==2:
                            optimal_n_clusters=1
                            df['CME_ID'] = 0
                        else:
                            df.loc[df['CME_ID'] == h, 'CME_ID'] = j
                            optimal_n_clusters=len(df['CME_ID'].unique())
        else:
            optimal_n_clusters=1
            df['CME_ID'] = 0


        fig,axs= plt.subplots(1,3, figsize=(12, 4))
        min_error=[]
        dt_list=[]
        hours=[]
        filter_criterion=[]
        dist=[]
        for k in range(optimal_n_clusters):
            filtered_df = df[df['CME_ID'] == k]
            
            #filtering small events (less than 3 images per event with one cluster), we keep all the masks for this event
            if (optimal_n_clusters==1) & (len(filtered_df["CME_ID"])<=3):
                for i in range(len(filtered_df)):
                    min_error.append(filtered_df.iloc[i])
                min_error_df = pd.DataFrame(min_error)
            
            else:
                x_points =np.array([i.timestamp() for i in filtered_df["DATE_TIME"]])
                y_cpa=np.array(filtered_df["CPA"])
                y_wa=np.array(filtered_df["AW"])
                y_apex=np.array(filtered_df["APEX"])

                #Fits the corresponding function type and calculates the distance from the masks
                # to the fitted function, the concavity of the function and the median velocity of it   
                cpa_dist,cpa_parabola,cpa_vel,axis1 = self._select_mask(axs[0],k,x_points, y_cpa, linear_error, linear, [1.,1.])
                wa_dist,wa_parabola,wa_vel,axis2= self._select_mask(axs[1],k,x_points, y_wa, linear_error, linear, [1.,1.])
                apex_dist,apex_parabola,apex_vel,axis3=self._select_mask(axs[2],k,x_points, y_apex, quadratic_error, quadratic, [1.,1.,0])   
                dist.append([cpa_dist,wa_dist,apex_dist])
                filter_criterion.append([apex_parabola,apex_vel])
                
                #Plotting data and the fitted function before filtering
                hours.extend([str(i.time()) for i in filtered_df["DATE_TIME"]])
                dt_list.extend(x_points)
                axs[0].scatter(x_points, filtered_df["CPA"], color=colors[k])
                axs[1].scatter(x_points, filtered_df["AW"], color=colors[k])
                axs[2].scatter(x_points, filtered_df["APEX"], color=colors[k])    

        if len(filtered_df["CME_ID"])>=3:        
            #filtering cases where all apex parabolas are negative
            parabola_criterion = [filter_criterion[v][0] for v in range(optimal_n_clusters)]
            count = all(valor is True for valor in parabola_criterion)
            if count:
                velocity_criterion = [filter_criterion[v][1] for v in range(optimal_n_clusters)]
                optimal_vel_idx = np.argmax(np.abs(velocity_criterion))
                selected_cluster= df[df['CME_ID'] == optimal_vel_idx]
                min_error_df = pd.DataFrame(selected_cluster)

            #filtering cases where at least one apex parabola is positive 
            else:
                for r in range(optimal_n_clusters):
                    filtered_df = df[df['CME_ID'] == r]
                    #Droping negative apex parabola cases
                    if (filter_criterion[r][0] & (optimal_n_clusters>1)):
                        idx = df[df['CME_ID'] == r].index
                        df = df.drop(idx)
                        df = df.reset_index(drop=True)

                    else:
                        filtered_df["CPA_DIST"] = dist[r][0]
                        filtered_df["AW_DIST"] = dist[r][1]
                        filtered_df["APEX_DIST"] = dist[r][2]
                        error=np.sqrt(dist[r][0]**2+dist[r][1]**2+dist[r][2]**2)
                        filtered_df["ERROR"] = error
                        
                        #filtering disperse masks according to cpa distance criterion 
                        filter_cpa = filtered_df.loc[filtered_df["CPA_DIST"]>MAX_CPA_DIST]
                        filtered_df = filtered_df[~filtered_df.isin(filter_cpa)].dropna()
                        unique_dates = filtered_df["DATE_TIME"].unique()
                        #filtering cases with more than one mask per day per cluster, keeps one per cluster (the one with the minimal error)
                        for m in unique_dates:
                            event = filtered_df[filtered_df['DATE_TIME'] == str(m)]
                            filtered_mask = event.loc[event['ERROR'].idxmin()]
                            min_error.append(filtered_mask)
                            
        
                min_error_df = pd.DataFrame(min_error)
        min_error_df = min_error_df.reset_index(drop=True)
        
        hours1 = [datetime.strptime(timestamp, "%H:%M:%S") for timestamp in hours]
        hours2 = [timestamp.strftime("%H:%M") for timestamp in hours1]
        axs[0].set_xticks(dt_list,hours2,rotation=45)
        axs[1].set_xticks(dt_list,hours2,rotation=45)
        axs[2].set_xticks(dt_list,hours2,rotation=45)
        axs[0].grid()
        axs[1].grid()
        axs[2].grid()
        axs[0].set_title("CPA")
        axs[1].set_title("AW")
        axs[2].set_title("APEX")
        plt.tight_layout()
        fig.savefig(opath+"/fitted_data.png")
        #ploting filtered data
        min_error_df['CME_ID'] = min_error_df['CME_ID'].astype(int)
        clusters = min_error_df['CME_ID'].unique()
        fig2,ax= plt.subplots(1,3, figsize=(12, 4))
        x_ax=[]
        time=[]
        for l in clusters:
            filtered_event=min_error_df[min_error_df['CME_ID'] == l]
            x=np.array([i.timestamp() for i in filtered_event["DATE_TIME"]])
            x_ax.extend(x)
            time.extend([str(i.time()) for i in filtered_event["DATE_TIME"]])
            ax[0].scatter(x, filtered_event["CPA"], color=colors[l])
            ax[1].scatter(x, filtered_event["AW"], color=colors[l])
            ax[2].scatter(x, filtered_event["APEX"], color=colors[l])
        time1 = [datetime.strptime(timestamp, "%H:%M:%S") for timestamp in time]
        time2 = [timestamp.strftime("%H:%M") for timestamp in time1]
        ax[0].set_xticks(x_ax,time2,rotation=45)
        ax[1].set_xticks(x_ax,time2,rotation=45)
        ax[2].set_xticks(x_ax,time2,rotation=45)
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        ax[0].set_title("CPA")
        ax[1].set_title("AW")
        ax[2].set_title("APEX")

        plt.tight_layout()
        fig2.savefig(opath+"/filtered_fitted_data.png")
        plt.close() 
        return min_error_df
        

        

    def _filter_masks(self, dates, masks, scores, labels, boxes, mask_prop):
        '''
        Filters the masks based on the cpa, aw and apex radius evolution consistency
        mask_prop: list of lists with the mask properties for each mask. Each maks's list has the following format: [id,float(scores[i]),cpa_ang, wide_ang, apex_dist]
        '''
        cpa_criterion = np.radians(20.) # criterion for the cpa angle [deg]
        apex_rad_crit = 0.2 # criterion for the apex radius [%]
        date_crit = 5. # criterion for the date, max [hours] for an acceptable gap. if larger it keeps later group only
        aw_crit = np.radians(15.) # criterion for the angular width [deg]
        
        # keeps only a max of two masks per image with the highest score
        for i in range(len(dates)):
            ok_idx = []
            if len(masks[i]) > 2:
                score_idx = np.argsort(scores[i])
                ok_idx.append(score_idx[-1])
                ok_idx.append(score_idx[-2])
                ok_idx = np.array(ok_idx)
                masks[i] = [masks[i][ok_idx[j]] for j in range(len(ok_idx))]
                scores[i] = [scores[i][ok_idx[j]] for j in range(len(ok_idx))]
                labels[i] = [labels[i][ok_idx[j]] for j in range(len(ok_idx))]
                boxes[i] = [boxes[i][ok_idx[j]] for j in range(len(ok_idx))]
                mask_prop[i] = [mask_prop[i][ok_idx[j]] for j in range(len(ok_idx))]
                        
            
        # filters DATES too far from the median
        x_points = np.array([i.timestamp() for i in dates])
        date_diff = x_points[1:]-x_points[:-1]
        date_diff = np.abs(date_diff-np.median(date_diff))
        gap_idx = np.where(date_diff>date_crit*3600.)[0]
        
        # keeps only the later group
        if len(gap_idx)>0:
            gap_idx = gap_idx[0]
            ok_dates = dates[gap_idx+1:]
            ok_masks = masks[gap_idx+1:]
            ok_scores = scores[gap_idx+1:]
            ok_labels = labels[gap_idx+1:]
            ok_boxes = boxes[gap_idx+1:]
            ok_mask_prop = mask_prop[gap_idx+1:]
        else:
            ok_dates = dates
            ok_masks = masks
            ok_scores = scores
            ok_labels = labels
            ok_boxes = boxes
            ok_mask_prop = mask_prop

        # flatten all lists
        all_masks = np.array([i for j in ok_masks for i in j])
        all_scores = np.array([i for j in ok_scores for i in j])
        all_lbl = np.array([i for j in ok_labels for i in j])
        all_boxes = np.array([i for j in ok_boxes for i in j])
        all_mask_prop = np.array([i for j in ok_mask_prop for i in j])
        all_dates = []
        for i in range(len(ok_masks)):
            all_dates.append([ok_dates[i]]*len(ok_masks[i]))
        all_dates = np.array([i for j in all_dates for i in j])

        # filters on 'CPA_ANG'
        x_points =np.array([i.timestamp() for i in all_dates])
        y_points =np.array(all_mask_prop[:,2]).astype('float32')
        ok_idx = self._filter_param(x_points, y_points, linear_error, linear, [1.,1.], cpa_criterion, percentual=False)
        all_masks = all_masks[ok_idx]
        all_scores = all_scores[ok_idx]
        all_lbl = all_lbl[ok_idx]
        all_boxes = all_boxes[ok_idx]
        all_mask_prop = all_mask_prop[ok_idx]
        all_dates = all_dates[ok_idx]        

        # # filters on 'APEX_RADIUS'
        # x_points =np.array([i.timestamp() for i in all_dates])
        # y_points =np.array(all_mask_prop[:,4]).astype('float32')
        # # use the mask score as fit weight
        # weights = np.array(all_mask_prop[:,1]).astype('float32')
        # ok_idx = self._filter_param(x_points,y_points, quadratic_error, quadratic,[0., 1.,1.], apex_rad_crit, weights=weights, percentual=True)
        # all_masks = all_masks[ok_idx]
        # all_scores = all_scores[ok_idx]
        # all_lbl = all_lbl[ok_idx]
        # all_boxes = all_boxes[ok_idx]
        # all_mask_prop = all_mask_prop[ok_idx]
        # all_dates = all_dates[ok_idx]
        
        # # filters on 'WIDTH_ANG'
        # x_points=np.array([i.timestamp() for i in all_dates])
        # y_points =np.array(all_mask_prop[:,3]).astype('float32')
        # ok_idx = self._filter_param(x_points,y_points, linear_error, linear,[0.,1.], aw_crit, percentual=False)
        # all_masks = all_masks[ok_idx]
        # all_scores = all_scores[ok_idx]
        # all_lbl = all_lbl[ok_idx]
        # all_boxes = all_boxes[ok_idx]
        # all_mask_prop = all_mask_prop[ok_idx]
        # all_dates = all_dates[ok_idx]

        # filters on 'WIDTH_ANG'
        # if multiple maks remain for a single date, keeps the one with WD closest to the median
        ok_idx = []
        x_points=np.array([i.timestamp() for i in all_dates])
        y_points =np.array(all_mask_prop[:,3]).astype('float32')
        median_aw = np.median(y_points)
        for i in np.unique(all_dates):
            date_idx = np.where(all_dates == i)[0]
            if len(date_idx) > 1:
                aw_idx = np.argmin(np.abs(y_points[date_idx]-median_aw))
                ok_idx.append(date_idx[aw_idx])
            else:
                ok_idx.append(date_idx[0])                  
        all_masks = all_masks[ok_idx]
        all_scores = all_scores[ok_idx]
        all_lbl = all_lbl[ok_idx]
        all_boxes = all_boxes[ok_idx]
        all_mask_prop = all_mask_prop[ok_idx]
        all_dates = all_dates[ok_idx]

        # #if multiple maks remain for a single date, keeps the one with highest score
        # ok_idx = []
        # for i in np.unique(all_dates):
        #     date_idx = np.where(all_dates == i)[0]
        #     if len(date_idx) > 1:
        #         score_idx = np.argmax(all_scores[date_idx])
        #         ok_idx.append(date_idx[score_idx])
        #     else:
        #         ok_idx.append(date_idx[0])                  
        # all_masks = all_masks[ok_idx]
        # all_scores = all_scores[ok_idx]
        # all_lbl = all_lbl[ok_idx]
        # all_boxes = all_boxes[ok_idx]
        # all_mask_prop = all_mask_prop[ok_idx]
        # all_dates = all_dates[ok_idx]
        
        #converts the first dim of all ouput np arrays to list of arrays
        all_masks = [all_masks[i,:,:] for i in range(len(all_dates))]
        all_scores = [all_scores[i] for i in range(len(all_dates))]
        all_lbl = [all_lbl[i] for i in range(len(all_dates))]
        all_boxes = [all_boxes[i,:] for i in range(len(all_dates))]
        all_mask_prop = [all_mask_prop[i,:] for i in range(len(all_dates))]
        all_dates = [all_dates[i] for i in range(len(all_dates))]
        
        return all_dates,all_masks, all_scores, all_lbl, all_boxes,all_mask_prop
        
    def infer_event2(self, imgs, dates, filter=True, model_param=None, resize=True, plate_scl=None, occulter_size=None,occulter_size_ext=None,
                    centerpix=None, mask_threshold=None, scr_threshold=None, plot_params=None, filter_halos=True,modified_masks=None,
                    percentiles=[5,95],increase_contrast=None):
        '''
        Updated version of infer_event, it recognices more than one CME per event.
        Infers masks for a temporal series of images belonging to the same event. It filters the masks found in the
        individual images using morphological and kinematic consistency criteria

        dates: list of datetime objects with the date of each image

        filter: if True, filters the masks found in the individual images using morphological and kinematic consistency criteria
        model_param: model parameters to use for inference. If None, the model parameters given at initialization are used
        resize: if True, the input image is resized to the size of the training images
        occulter_size: size of the artifitial occulter in the image [px]. If 0, the occulter is not masked out    
        mask_threshold: threshold for the predicted box pixels to be considered as part of the mask
        scr_threshold: only detections with score larger than this value are considered    
        plot_params: if dedined to an output dir path, plots the evolution of the cpa, aw and apex radius for all the masks found and the filtered ones
        plate_scl: plate scale [Unit/px] for each input image to scale the apex radius accordingly
        filter_halos: if True, filters the masks which boxes center is within the occulter size

        returns the filtered original images and a dataframe with the parameters of the CME including the CME ID.  
        
        '''
        MAX_CPA_DIST=np.radians(30) #max disctance between a point and the cpa median in one cluster
        MIN_CLUSTER_POINTS=5 # minimum amount of points in every cluster
        MIN_CPA_DIFF = 0.35 # minimal difference between two posible CMEs in radians

        if mask_threshold is not None:
            self.mask_threshold = mask_threshold
        if scr_threshold is not None:
            self.scr_threshold = scr_threshold

        self.plot_params = plot_params

        #sorts imgs based on date
        idx = np.argsort(dates)
        dates = [dates[i] for i in idx]
        in_imgs = [imgs[i] for i in idx]
        if occulter_size is not None:
            in_occulter_size = [occulter_size[i] for i in idx]
        else:
            in_occulter_size = [0 for i in idx]

        if occulter_size_ext is not None:
            in_occulter_size_ext = [occulter_size_ext[i] for i in idx]
        else:
            in_occulter_size_ext = [None for i in idx]

        if plate_scl is not None:
            in_plate_scl = [plate_scl[i] for i in idx]
        else:
            in_plate_scl = [1 for i in idx]

        if centerpix is not None:
            in_centerpix = [centerpix[i] for i in idx]
        else:
            in_centerpix = [None for i in idx]

        #Get the mask for all images in imgs
        all_masks = []
        all_scores = []
        all_lbl = []
        all_boxes = []
        all_orig_img = []
        all_mask_prop = []
        all_dates = []
        all_plate_scl=[]

        if modified_masks is not None:
            with open(modified_masks, "rb") as f:
                data = pd.compat.pickle_compat.load(f) #pickle.load(f) works if pandas version is same on both sides.
            in_imgs = data["OK_ORIG_IMG"].copy()
            all_orig_img = data["OK_ORIG_IMG"].copy()
            dates = data["OK_DATES"]
            #breakpoint()

        for i in range(len(in_imgs)):
            if modified_masks is None:
                #infer masks
                orig_img, mask, score, lbl, box = self.infer(in_imgs[i], model_param=model_param, resize=resize, 
                                                            occulter_size=in_occulter_size[i],occulter_size_ext=in_occulter_size_ext[i],
                                                            centerpix=in_centerpix[i],increase_contrast=increase_contrast)
                all_orig_img.append(orig_img)
                #print(i,dates[i])

            if modified_masks is not None:
                #Usefull in case of using modified masks in real images. Ask D. Lloveras.
                score = data["SCR"][i]
                lbl = data["LABEL"][i]
                box = data["BOX"][i]
                mask = [data["MASK"][i]]
                #print(i,dates[i])
            # compute cpa, aw and apex. Already filters by score and other aspects
            self.debug_flag = i
            mask_prop = self._compute_mask_prop(mask, score, lbl, box, plate_scl=in_plate_scl[i], 
                                                filter_halos=filter_halos, occulter_size=in_occulter_size[i],centerpix=in_centerpix[i],percentiles=percentiles)
            
            # appends results only if mask_prop[1] is not NaN, i.e., if it fulfills the filters in compute_mask_prop
            mask_prop_aux = []
            for j in range(len(mask_prop)):
                mask_prop_aux.append(mask_prop[j][1])
            ok_ind_tmp = np.where(~np.isnan(np.array(mask_prop_aux)))
            ok_ind = ok_ind_tmp[0]
            if len(ok_ind) >0:
                all_masks.append([mask[j] for j in ok_ind])
                all_scores.append([score[j] for j in ok_ind])
                all_lbl.append([lbl[j] for j in ok_ind])
                all_boxes.append([box[j] for j in ok_ind])
                all_mask_prop.append([mask_prop[j] for j in ok_ind])
                all_plate_scl.append(in_plate_scl[i])
                all_dates.append(dates[i])
            
        if len(all_masks)>=2:
            # keeps only one mask per img based on cpa, aw and apex radius evolution consistency
            if filter:
                df = self._filter_masks2(all_dates, all_masks, all_scores, all_lbl, all_boxes, all_mask_prop,
                                        all_plate_scl,self.plot_params,MAX_CPA_DIST,MIN_CPA_DIFF,MIN_CLUSTER_POINTS)
                # plot parameters
                if plot_params is not None:
                    self._plot_mask_prop2(df, self.plot_params , ending='_filtered',save_pickle=True)
                #ok_dates=sorted(df['DATE_TIME'].unique())
                ok_dates=df['DATE_TIME'].unique() 
                #no aplico un sort ya que al hacerlo puede que el index del df no coincida con el de all_orig_img. Creo que es mas robusto asi.
                #En realidad era necesario xq se mete un reset index al df en filter_masks2.
                for m in ok_dates:
                    event = df[df['DATE_TIME'] == m]
                    if event["MASK"].isnull().all():
                        idx = dates.index(m)
                        all_orig_img.pop(idx)

                all_idx=[]
                for date in dates:
                    if date not in ok_dates:
                        idx = dates.index(date)
                        all_idx.append(idx) #sin esta linea el for no cummple ningun roll, ni la linea posterior
                all_orig_img = [elemento for s, elemento in enumerate(all_orig_img) if s not in all_idx]      
                df = df.dropna(subset=['MASK']) 
                return all_orig_img,ok_dates, df
        return  all_orig_img, all_dates, all_masks, all_scores, all_lbl, all_boxes, all_mask_prop
                        


