import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import matplotlib as mpl
import math
from datetime import datetime
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
mpl.use('Agg')

__author__ = "Francisco Iglesias"
__copyright__ = "Grupo de Estudios en Heliofisica de Mendoza - https://sites.google.com/um.edu.ar/gehme"
__version__ = "0.1"
__maintainer__ = "Francisco Iglesias"
__email__ = "franciscoaiglesias@gmail.com"

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
    def __init__(self, device, pre_trained_model=None, version='v4', imsize=[512,512]):
        '''
        Initializes the model
        device: device to use for training and inference
        pre_trained_model: path to model parameters to use for inference
        version: version of the model to use. Options are 'v3' and 'v4'
        '''
        self.device = device
        self.pre_trained_model = pre_trained_model    
        self.imsize = imsize    
        # model param
        if version == 'v3':
            self.num_classes = 3 # background, CME, occulter
            # number of trainable layers in the backbone resnet, int in [0,5] range. Specifies the stage number.
            # Stages 2-5 are composed of 6 convolutional layers each. Stage 1 is more complex
            # See https://arxiv.org/pdf/1512.03385.pdf
            self.trainable_backbone_layers = 3
        if version == 'v4':
            self.labels=['Background','Occ','CME'] # labels for the different classes
            self.num_classes = 3 # background, CME, occulter
            self.trainable_backbone_layers = 4
            self.mask_threshold = 0.60 # value to consider a pixel belongs to the object
            self.scr_threshold = 0.25 # only detections with score larger than this value are considered
        # innitializes the model
        self.model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=self.trainable_backbone_layers) 
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features 
        self.model.roi_heads.box_predictor=FastRCNNPredictor(self.in_features,num_classes=self.num_classes)
        if self.pre_trained_model is not None:
            model_param = torch.load(self.pre_trained_model, map_location=device)
            self.model.load_state_dict(model_param)      
        self.model.to(self.device)
        return

    def normalize(self, image):
        '''
        Normalizes the input image to
            - gaussian filter to reduce noise in the image
        '''
        sd_range=1.5
        #smooth_kernel=3
        #image = cv2.GaussianBlur(image, (smooth_kernel, smooth_kernel), 0)
        m = np.mean(image)
        sd = np.std(image)
        image = (image - m + sd_range * sd) / (2 * sd_range * sd)
        image[image >1]=1
        image[image <0]=0
        return image
    
    def train(self, opt_type='adam' , lr=1e-5):
        '''
        Sets optimizer type and train mode
        '''
        if opt_type == 'adam':
            self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr) # optimization technique that comes under gradient decent algorithm    
        else:
            print('Optimizer opt_type not implemented')
            return None
        self.model.train()        
        return 
    
    def resize(self, img):
        '''
        Resizes the input image to the given size
        '''
        return cv2.resize(img, self.imsize, cv2.INTER_LINEAR)
    
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
            img[mask==1] = np.mean(img)
        else:
            img[mask==1] = repleace_value
        return img

    
    def infer(self, img, model_param=None, resize=True, occulter_size=0,centerpix=None):
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
        if occulter_size > 0:
            images = self.mask_occulter(images, occulter_size,centerpix)    
        images = self.normalize(images) # normalize to 0,1
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
                if occulter_size > 0 and lbl == self.labels.index('Occ'):
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
            print('Warning, no masks found in the image')
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
        

    def _rec2pol(self, mask):
        '''
        Converts the x,y mask to polar coordinates
        Only pixels above the mask_threshold are considered
        TODO: Consider arbitrary image center
        '''
        nans = np.full(self.imsize, np.nan)
        pol_mask=[]
        #creates an array with zero value inside the mask and Nan value outside             
        masked = nans.copy()
        masked[:, :][mask > self.mask_threshold] = 0   
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

    def _compute_mask_prop(self, masks, scores, labels, boxes, plate_scl=1, filter_halos=True, occulter_size=0,centerpix=None):    
        '''
        Computes the CPA, AW and apex radius for the given 'CME' masks.
        If the mask label is not "CME" and if is not a Halo and filter_halos=True, it returns None
        plate_scl: if defined the apex radius is scaled based on the plate scale of the image. If 0, no filter is applied
        filter_halos: if True, filters the masks with boxes center within the occulter size, and wide_ angle > MAX_WIDE_ANG
        TODO: Compute apex in Rs; Use generic image center
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
                pol_mask=self._rec2pol(masks[i])
                if (pol_mask is not None):            
                    #takes the min and max angles and calculates cpa and wide angles
                    angles = [s[1] for s in pol_mask]
                    if len(angles)>0:
                        # checks for the case where the cpa is close to 0 or 2pi
                        if np.max(angles)-np.min(angles) >= 0.9*2*np.pi:
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
                                                              
                        if filter_halos:
                            if wide_ang < self.max_wide_ang:
                                prop_list.append([i,float(scores[i]),cpa_ang, wide_ang, apex_dist])  
                            else:
                                prop_list.append([i,np.nan, np.nan, np.nan, np.nan])
                        else:
                            prop_list.append([i,float(scores[i]),cpa_ang, wide_ang, apex_dist])
                else:
                    prop_list.append([i,np.nan, np.nan, np.nan, np.nan])
            else:
                prop_list.append([i,np.nan, np.nan, np.nan, np.nan])  
        return prop_list         

    def _plot_mask_prop(self, dates, param, opath, ending='_filtered', x_title='Date and hour', style='*', save=True):
            '''
            plots the evolution of the cpa, aw and apex radius for all the masks found and the filtered ones
            dates: list of datetime objects with the date of each event
            param: 3d list with the mask properties for all masks found in each image. Each maks's has the following prop: [id,float(scores[i]),cpa_ang, wide_ang, apex_dist]
            '''
            self.mask_prop_labels=['MASK ID', 'SCORE','CPA_ANG','WIDTH_ANG','APEX_RADIUS']

            print('Plotting masks properties to '+opath)
            # repeat dates for all masks
            if ending == '_all':
                x = []
                for i in range(len(dates)):
                    x.append([dates[i]]*len(param[i]))
                x = np.array([i for j in x for i in j])
            elif ending == '_filtered':                
                x = dates.copy()
            else:
                print('Unrecognized value for ending parameter')
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
                



    def _plot_mask_prop2(self, df, opath, ending='_filtered', x_title='Date and hour', style='*', save=True):
            '''
            plots the evolution of the cpa, aw and apex radius for all the masks found and the filtered ones
            dates: list of datetime objects with the date of each event
            param: 3d list with the mask properties for all masks found in each image. Each maks's has the following prop: [id,float(scores[i]),cpa_ang, wide_ang, apex_dist]
            '''
            parameters= ['CPA', 'MASK_ID', 'SCR', 'WA', 'APEX', 'CME_ID']
            colors=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w'] 

            df["DATE_TIME"]=pd.to_datetime(df["DATE_TIME"])
            unique_dates=df["DATE_TIME"].unique()
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
                    ax.plot(x_points, y_points, style,color=colors[k])
                hours = [str(timestamp.time()) for timestamp in dt_list]

                ax.set_title(y_title)    
                ax.set_xlabel(x_title)
                plt.xticks(dt_list,hours,rotation=90)
                plt.grid()
                plt.tight_layout()
                if save:
                    os.makedirs(opath, exist_ok=True)
                    fig.savefig(opath+'/'+str.lower(y_title)+ending+".png")
                    plt.close()
                else:
                    return fig, ax 
               
            
            # self.mask_prop_labels=['MASK ID', 'SCORE','CPA_ANG','WIDTH_ANG','APEX_RADIUS',"CME_ID"]

            # print('Plotting masks properties to '+ str(opath))
            # # repeat dates for all masks
            # if ending == '_all':
            #     x = []
            #     for i in range(len(dates)):
            #         x.append([dates[i]]*len(param[i]))
            #     x = np.array([i for j in x for i in j])
            # elif ending == '_filtered':                
            #     x = np.array(dates.copy())
            # else:
            #     print('Unrecognized value for ending parameter')
            #     return None
                        
      

            # colors = ["red","blue","orange","yellow","purple"]           
            
            # for par in range(len(param[0][0])): #loops on props
            #     fig, ax = plt.subplots()   
            #     for cid in range(len(param[0])):#loops on id to plot
            #         cparam=[]
            #         for t in range(len(param)):#loops in time
            #             flag=0
            #             for id in range(len(param[0])):#loops in all id
            #                 if param[t][id][5] == cid:
            #                     cparam.append(param[t][id][par])
            #                     flag=1
            #             if flag==0:
            #                 cparam.append(np.nan)

            #         y_title = self.mask_prop_labels[par]

            #         if y_title.endswith("ANG"):
            #             b = np.array([np.degrees(i) for i in cparam])
            #         else:
            #             b = np.array(cparam)

            #         ok_idx=~np.isnan(cparam)
            #         ax.plot(x[ok_idx], b[ok_idx], style,color=colors[cid])
            #     ax.set_xlabel(x_title)
            #     plt.xticks(rotation=45)
            #     plt.grid()
            #     plt.tight_layout()
            #     if save:
            #         os.makedirs(opath, exist_ok=True)
            #         fig.savefig(opath+'/'+str.lower(y_title)+ending+".png")
            #         plt.close()
            #     else:
            #         return fig, ax               
        
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
        colors=["orange","cyan","yellow"]
        vel_threshold=100 #km/s
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
            velocity = quadratic_velocity(x - x[0], *fit.x[:-1])
            average_velocity = np.mean(velocity)
            if average_velocity < vel_threshold:
                dist=None

            
        label = f"y={fit.x[0]:.2f}x+{fit.x[1]:.2f}\nR={np.corrcoef(x, y)[0,1]:.2f}"
        axis.plot(x, fit_func(x-x[0], *fit.x), color=colors[k], linestyle='--', label=label, linewidth=1.5)
       
        return dist, axis
    
        # sorted_indices = np.argsort(in_x)
        # x = in_x[sorted_indices]
        # y = in_y[sorted_indices]
        # selected_x = []
        # selected_y = []
        # current_x = x[0]
        # current_x_values = []
        # current_y_values = []

        # for i in range(len(x)):
        #     if x[i] == current_x:
        #         current_x_values.append(x[i])
        #         current_y_values.append(y[i])
        #     else:
        #         fit = least_squares(error_func, in_cond, kwargs={'x': current_x_values - current_x_values[0], 'y': current_y_values, 'w': weights})
        #         dist = np.abs(fit_func(current_x_values - current_x_values[0], *fit.x) - current_y_values)
        #         min_dist_idx= np.argmin(dist)
        #         selected_x.append(current_x_values[min_dist_idx])
        #         selected_y.append(current_y_values[min_dist_idx])

        #         # Update the current x-coordinate and the lists of values
        #         current_x = x[i]
        #         current_x_values = [x[i]]
        #         current_y_values = [y[i]]
        
        # # Add the last set of data.
        # fit = least_squares(error_func, in_cond, kwargs={'x': current_x_values - current_x_values[0], 'y': current_y_values, 'w': weights})
        # dist = np.abs(fit_func(current_x_values - current_x_values[0], *fit.x) - current_y_values)
        # min_dist_idx = np.argmin(dist)
        # selected_x.append(current_x_values[min_dist_idx])
        # selected_y.append(current_y_values[min_dist_idx])
        


        # return selected_x, selected_y
    

    def _filter_masks2(self, dates, masks, scores, labels, boxes, mask_prop):
        dist_threshold=50
        style='*'
        colors=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w'] 
        #transforms inputs in a dataframe
        data=[]
        for i in range(len(dates)):
            for j in range(len(masks[i])):
                mask_id=mask_prop[i][j][0]
                scr=mask_prop[i][j][1]
                cpa = mask_prop[i][j][2]
                wa=mask_prop[i][j][3]
                apex=mask_prop[i][j][4]
                date= dates[i]
                label=labels[i][j]
                box=boxes[i][j]
                mask=masks[i][j]
                data.append((date,mask_id,scr,cpa,wa,apex,label,box,mask))
        df = pd.DataFrame(data, columns=["DATE_TIME","MASK_ID","SCR","CPA","WA","APEX","LABEL","BOX","MASK"])
        
        #Gets the optimal number of clusters for the data
        data= df["CPA"].values.reshape(-1, 1)
        silhouette_scores = []
        n_clusters_range = range(2, 5) 
        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            km_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, km_labels)
            silhouette_scores.append(silhouette_avg)

        optimal_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]

        #Adjust KMEANS to optimal cluster number
        kmeans = KMeans(n_clusters=optimal_n_clusters,random_state=0)
        labels = kmeans.fit_predict(data)
        df['CME_ID'] = [int(i) for i in km_labels]
        
        all_filtered_x = []
        all_filtered_y = []
        fig,axs= plt.subplots(1,3,figsize=(12,4))
        for k in range(optimal_n_clusters):
            filtered_df = df[df['CME_ID'] == k]
            x_points =np.array([i.timestamp() for i in filtered_df["DATE_TIME"]])
            y_cpa=np.array(filtered_df["CPA"])
            y_wa=np.array(filtered_df["WA"])
            y_apex=np.array(filtered_df["APEX"])
            apex_dist,axis3=self._select_mask(axs[2],k,x_points, y_apex, quadratic_error, quadratic, [1.,1.,0])
            if apex_dist is None:
                idx = df[df.isin(filtered_df.to_dict(orient='list')).all(axis=1)].index
                df = df.drop(idx)
            else:
                cpa_dist,axis1 = self._select_mask(axs[0],k,x_points, y_cpa, linear_error, linear, [1.,1.])
                wa_dist,axis2= self._select_mask(axs[1],k,x_points, y_wa, linear_error, linear, [1.,1.])
                for h in range(len(x_points)):
                    error=np.sqrt(cpa_dist**2+wa_dist**2+apex_dist**2)
           
        breakpoint()
        

        #all_filtered_x = []
        #all_filtered_y = []
        # gets one mask per cluster
        # for k in range(optimal_n_clusters):
        #     filtered_df = df[df['CME_ID'] == k]
        #     x_points =np.array([i.timestamp() for i in filtered_df["DATE_TIME"]])
        #     y_points =np.array(filtered_df["CPA"])
        #     filtered_x,filtered_y = self._select_mask(x_points, y_points, linear_error, linear, [1.,1.])
        #     #fills the dates that dosen't appear in filter_x with Nan in filterd_y
        #     for j in range(len(x_points)-1):
        #         if x_points[j] != filtered_x[j]:
        #             filtered_x.insert(j, x_points[j])
        #             filtered_y.insert(j, None)

        #         if  (j==len(x_points)-2) & (len(x_points)-1 == len(filtered_x)):
        #             filtered_x.insert(j+1, x_points[j+1])
        #             filtered_y.insert(j+1, None)

        #     filtered_x = [pd.Timestamp(i, unit='s') for i in filtered_x]
        #     all_filtered_x.extend(filtered_x)
        #     all_filtered_y.extend(filtered_y)

        # filtered_df = pd.DataFrame({'DATE_TIME': all_filtered_x, 'CPA': all_filtered_y})
        # full_df = filtered_df.merge(df, on=['DATE_TIME', 'CPA'], how='left')
        #return full_df
        
        
    
        

        

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
        
    def infer_event2(self, imgs, dates, filter=True, model_param=None, resize=True, plate_scl=None, occulter_size=None,centerpix=None, mask_threshold=None, 
                    scr_threshold=None, plot_params=None, filter_halos=True):
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
        plate_scl: plate scale [arcsec/px] for each input image to scale the apex radius accordingly
        filter_halos: if True, filters the masks which boxes center is within the occulter size

        returns the filtered original images and a dataframe with the parameters of the CME including the CME ID.  
        
        '''
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
        if plate_scl is not None:
            in_plate_scl = [plate_scl[i] for i in idx]
        else:
            in_plate_scl = [1 for i in idx]

        #Get the mask for all images in imgs
        all_masks = []
        all_scores = []
        all_lbl = []
        all_boxes = []
        all_orig_img = []
        all_mask_prop = []
        all_dates = []
        for i in range(len(in_imgs)):
            #infer masks
            orig_img, mask, score, lbl, box = self.infer(in_imgs[i], model_param=model_param, resize=resize, occulter_size=in_occulter_size[i],centerpix=centerpix[i])
            all_orig_img.append(orig_img)
            # compute cpa, aw and apex. Already filters by score and other aspects
            self.debug_flag = i
            mask_prop = self._compute_mask_prop(mask, score, lbl, box, plate_scl=in_plate_scl[i]/in_plate_scl[0], filter_halos=filter_halos, occulter_size=in_occulter_size[i],centerpix=centerpix[i])
            # appends results only if mask_prop[1] is not NaN, i.e., if it fulfills the filters in compute_mask_prop
            ok_ind = np.where(~np.isnan(np.array(mask_prop)[:,1]))[0]
            if len(ok_ind) >0:
                all_masks.append([mask[i] for i in ok_ind])
                all_scores.append([score[i] for i in ok_ind])
                all_lbl.append([lbl[i] for i in ok_ind])
                all_boxes.append([box[i] for i in ok_ind])
                all_mask_prop.append([mask_prop[i] for i in ok_ind])  
                all_dates.append(dates[i])
        if len(all_masks)>=2:
            # plots parameters
            if plot_params is not None:
                self._plot_mask_prop(all_dates, all_mask_prop, self.plot_params , ending='_all')
            # keeps only one mask per img based on cpa, aw and apex radius evolution consistency
            if filter:
                df = self._filter_masks2(all_dates, all_masks, all_scores, all_lbl, all_boxes, all_mask_prop)
                self._plot_mask_prop2(df, self.plot_params , ending='_filtered')
                ok_dates=sorted(df['DATE_TIME'].unique())
                for m in ok_dates:
                    event = df[df['DATE_TIME'] == m]
                    if event["MASK"].isnull().all():
                        idx = dates.index(m)
                        all_orig_img.pop(idx)

                all_idx=[]
                for date in dates:
                    if date not in ok_dates:
                        idx = dates.index(date)
                all_orig_img = [elemento for s, elemento in enumerate(all_orig_img) if s not in all_idx]      
                df = df.dropna(subset=['MASK']) 
                return all_orig_img,ok_dates, df
                        


