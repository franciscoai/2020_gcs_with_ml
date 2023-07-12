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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from ext_libs.rebin import rebin
from neural_cme_seg import neural_cme_segmentation
import csv

__author__ = "Francisco Iglesias"
__copyright__ = "Grupo de Estudios en Heliofisica de Mendoza - https://sites.google.com/um.edu.ar/gehme"
__version__ = "0.1"
__maintainer__ = "Francisco Iglesias"
__email__ = "franciscoaiglesias@gmail.com"

def read_fits(file_path):
    imageSize=[512,512]
    try:       
        img = fits.open(file_path)[0].data
        img = rebin(img, imageSize)   
        return img  
    except:
        print(f'WARNING. I could not find file {file_path}')
        return None

def plot_to_png(ofile,orig_img, masks, title=None, labels=None, boxes=None, scores=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
    mask_threshold = 0.5 # value to consider a pixel belongs to the object
    scr_threshold = 0.7 # only detections with score larger than this value are considered
    color=['r','b','g','k','y']
    obj_labels = ['Occ', 'CME','N/A','N/A']
    #
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(orig_img[0]), np.nan)
    fig, axs = plt.subplots(1, len(orig_img)*2, figsize=(20, 10))
    axs = axs.ravel()
    for i in range(len(orig_img)):
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
                nb+=1
    # axs[0].set_title(f'Cor A: {len(boxes[0])} objects detected') 
    # axs[1].set_title(f'Cor B: {len(boxes[1])} objects detected')               
    # axs[2].set_title(f'Lasco: {len(boxes[2])} objects detected')     
    #if title is not None:
    #    fig.suptitle('\n'.join([title[i]+' ; '+title[i+1] for i in range(0,len(title),2)]) , fontsize=16)   

    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()
     
      
#main
#------------------------------------------------------------------Testing the CNN-----------------------------------------------------------------
model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v3"
opath= model_path + "/infer_neural_cme_seg_corona_back_cor2"
ipath=  "/gehme/projects/2020_gcs_with_ml/data/corona_back_database/cor2/cor2_a"
file_ext=".fits"
trained_model = '3999.torch'

#main
gpu=1 # GPU to use
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unless its not available
print(f'Using device:  {device}')

os.makedirs(opath, exist_ok=True)
#loads model
model_param = torch.load(model_path + "/"+ trained_model)
# list .fits files in ipath
files = [f for f in os.listdir(ipath) if f.endswith(file_ext)]
#vars to store all results
results = []
for f in files:
    print(f'Processing {f}')
    #read fits
    img = read_fits(os.path.join(ipath,f))
    # infers
    imga, maska, scra, labelsa, boxesa  = neural_cme_segmentation(model_param, img, device)
    #save results
    results.append({'file':f, 'img':imga, 'mask':maska, 'scr':scra, 'labels':labelsa, 'boxes':boxesa})
    #plot results
    ofile = os.path.join(opath,f)+'.png'
    plot_to_png(ofile, [imga], [maska], title=[f], labels=[labelsa], boxes=[boxesa], scores=[scra])