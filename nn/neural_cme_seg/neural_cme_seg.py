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
            self.num_classes = 3 # background, CME, occulter
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
    
    def mask_occulter(self, img, occulter_size):
        '''
        Replace a circular area of radius occulter_size in input image[h,w,3] with a constant value
        '''
        h,w = img.shape[:2]
        mask = np.zeros((h,w), dtype=np.uint8)
        cv2.circle(mask, (int(w/2), int(h/2)), occulter_size, 1, -1)
        img[mask==1] = np.mean(img)
        return img

    
    def infer(self, img, model_param=None, resize=True, occulter_size=0):
        '''        
        Infers a cme segmentation mask in the input coronograph image (img) using the trained R-CNN
        model_param: model parameters to use for inference. If None, the model parameters given at initialization are used
        resize: if True, the input image is resized to the size of the training images
        occulter_size: size of the artifitial occulter in the image. If 0, the occulter is not masked out
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
            images = self.mask_occulter(images, occulter_size)    
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
    
    def test_mask(self, img, target, mask_threshold=0.5, model_param=None, resize=True, occulter_size=0):
        """
        Makes an inference on img and return the mask that has the smallest loss wrt target binary mask
        mask_threshold: threshold for the predicted box pixels to be considered as part of the mask
        """
        orig_img, all_masks, all_scores, all_lbl, all_boxes = self.infer(img, model_param=model_param, resize=resize, occulter_size=occulter_size)
        if len(all_masks) == 0:
            print('Warning, no masks found in the image')
            return None
        # compute loss for all masks
        all_loss = []
        for i in range(len(all_masks)):
            msk = all_masks[i]
            msk[msk > mask_threshold] = 1
            msk[msk <= mask_threshold] = 0
            all_loss.append(np.sum(np.abs(msk - target))/np.sum(target))
        # return the mask with the smallest loss
        imin = np.argmin(all_loss)
        return orig_img, all_masks[imin], all_scores[imin], all_lbl[imin], all_boxes[imin], all_loss[imin]
        
