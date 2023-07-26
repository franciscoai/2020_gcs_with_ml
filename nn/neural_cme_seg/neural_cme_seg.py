import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import matplotlib as mpl
mpl.use('Agg')

__author__ = "Francisco Iglesias"
__copyright__ = "Grupo de Estudios en Heliofisica de Mendoza - https://sites.google.com/um.edu.ar/gehme"
__version__ = "0.1"
__maintainer__ = "Francisco Iglesias"
__email__ = "franciscoaiglesias@gmail.com"

def normalize(image):
    '''
    Normalizes the values of the input image to have a given range (as fractions of the sd around the mean)
    maped to [0,1]. It clips output values outside [0,1]
    '''
    sd_range=1.
    m = np.mean(image)
    sd = np.std(image)
    image = (image - m + sd_range * sd) / (2 * sd_range * sd)
    image[image >1]=1
    image[image <0]=0
    return image


def neural_cme_segmentation(model_param, img, device):
    '''
    
    Infers a cme segmentation mask in the input coronograph image (img) using the trauined R-CNN with weigths given by model_param

    '''    
    #loads model
    model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) 
    in_features = model.roi_heads.box_predictor.cls_score.in_features 
    model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=3)
    model.load_state_dict(model_param) #loads the last iteration of training 
    model.to(device)# move model to the right device
    model.eval()#set the model to evaluation state

    #inference

    images = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)        
    images = normalize(images) #cv2.normalize(images, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # normalize to 0,1
    oimages = images.copy()                                   
    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
    images=images.swapaxes(1, 3).swapaxes(2, 3)
    images = list(image.to(device) for image in images)
    with torch.no_grad(): #runs the image through the net and gets a prediction for the object in the image.
        pred = model(images)

    # returns the network input image along with the infered masks, labels and scores
    orig_img =  np.array(oimages)[:,:,0]
    nmasks = len(pred[0]['masks'])
    all_lbl = []
    all_boxes = []
    all_scores = []    
    all_masks = [] 
    if nmasks > 0:
        for i in range(nmasks):
            msk=pred[0]['masks'][i,0].detach().cpu().numpy()
            all_masks.append(msk)
            scr=pred[0]['scores'][i].detach().cpu().numpy()
            all_scores.append(scr)
            lbl = pred[0]['labels'][i].detach().cpu().numpy()
            all_lbl.append(lbl)
            box = pred[0]['boxes'][i].detach().cpu().numpy()
            all_boxes.append(box)              
    else:
        os.error('Found no maks in the current image')
    return orig_img, all_masks, all_scores, all_lbl, all_boxes