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
from neural_cme_seg import neural_cme_segmentation
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from ext_libs.rebin import rebin

def read_fits(file_path, header=False, imageSize=[512,512]):
    try:       
        img = fits.open(file_path)[0].data
        img = rebin(img, imageSize)   
        if header:
            return img, fits.open(file_path)[0].header
        else:
            return img 
    except:
        print(f'WARNING. I could not find file {file_path}')
        return None
    
def plot_to_png(ofile, orig_img, masks, scr_threshold=0.3, mask_threshold=0.8 , title=None, labels=None, boxes=None, scores=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
     # only detections with score larger than this value are considered
    color=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w']
    obj_labels = ['Back', 'Occ','CME','N/A']
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
#------------------------------------------------------------------Testing the CNN--------------------------------------------------------------------------
model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4"
model_version="v4"
repo_path = os.dirname(os.dirname(os.dirname(os.path.abspath(__file__))))
ipath = '/gehme/data/solo/fsi/20220325/'
opath= repo_path + '/output/fsi/20220325/'
file_ext=".fits" # files to use
trained_model = '6000.torch'
occ_size = 50 # Artifitial occulter radius in pixels. Use 0 to avoid. [Stereo a/b C1, Stereo a/b C2, Lasco C2]

#main
gpu=0 # GPU to use
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unless its not available
print(f'Using device:  {device}')
#loads images
files = os.listdir(ipath)
files = [os.path.join(ipath, e) for e in files]
files = [e for e in files if os.path.splitext(e)[1] == file_ext]
#loads nn model
nn_seg = neural_cme_segmentation(device, pre_trained_model = model_path + "/"+ trained_model, version=model_version)
os.makedirs(opath, exist_ok=True)
#inference on all images
for f in files:
    print(f'Processing file {f}')
    img = cv2.imread(f)
    orig_img, masks, scores, labels, boxes  = nn_seg.infer(img, model_param=None, resize=False, occulter_size=0)
    # plot the predicted mask
    ofile = opath+"/"+os.path.basename(f)+'.png'
    plot_to_png(ofile, [img], [masks], scores=[scores], labels=[labels], boxes=[boxes])