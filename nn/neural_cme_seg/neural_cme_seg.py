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

class neural_cme_segmentation():
    '''
    Class to perform CME segmentation using Mask R-CNN

    Based on Pytorch vision model MASKRCNN_RESNET50_FPN from the paper https://arxiv.org/pdf/1703.06870.pdf

    see also:
    https://pytorch.org/vision/0.12/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    https://towardsdatascience.com/train-mask-rcnn-net-for-object-detection-in-60-lines-of-code-9b6bbff292c3
    '''
    def __init__(self, device, pre_trained_model = None, version='v4'):
        '''
        Initializes the model
        device: device to use for training and inference
        pre_trained_model: path to model parameters to use for inference
        version: version of the model to use. Options are 'v3' and 'v4'
        '''
        self.device = device
        self.pre_trained_model = pre_trained_model        
        # model param
        if version == 'v3':
            self.num_classes = 3 # background and CME
            self.trainable_backbone_layers = 3
        if version == 'v4':
            self.num_classes = 2 # background and CME
            self.trainable_backbone_layers = 4
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
        Normalizes the values of the input image to have a given range (as fractions of the sd around the mean)
        maped to [0,1]. It clips output values outside [0,1]
        '''
        sd_range=1.5
        m = np.mean(image)
        sd = np.std(image)
        image = (image - m + sd_range * sd) / (2 * sd_range * sd)
        image[image >1]=1
        image[image <0]=0
        return image
    
    def train(self, opt_type='adam' , lr=1e-6):
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
    
    def infer(self, img, model_param=None):
        '''        
        Infers a cme segmentation mask in the input coronograph image (img) using the trauined R-CNN with weigths given by model_param
        '''    
        #loads model
        if model_param is not None:
            self.model.load_state_dict(model_param) 
        elif self.pre_trained_model is None:
            os.error('No model parameters given')
        
        self.model.eval() #set the model to evaluation state

        #inference
        images = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)        
        images = self.normalize(images) #cv2.normalize(images, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # normalize to 0,1
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