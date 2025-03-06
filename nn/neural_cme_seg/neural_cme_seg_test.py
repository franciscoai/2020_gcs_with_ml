import random
import numpy as np
import torch.utils.data
import cv2
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib as mpl
from astropy.io import fits
mpl.use('Agg')
import sys
from neural_cme_seg import neural_cme_segmentation

def plot_to_png(ofile, orig_img, masks, true_mask, scr_threshold=0.3, mask_threshold=0.3 , title=None, 
                labels=None, boxes=None, scores=None, version='v4'):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
     # only detections with score larger than this value are considered
    color=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w']
    if version=='v4':
        obj_labels = ['Back', 'Occ','CME','N/A']
    elif version=='v5':
        obj_labels = ['Back', 'CME']
    else:
        print(f'ERROR. Version {version} not supported')
        sys.exit()
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
        #add true mask
        masked = nans.copy()
        masked[true_mask[i] > 0] = 3 
        axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=4)
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

#------------------------------------------------------------------Testing the CNN-----------------------------------------------------------------
testDir =  '/gehme/projects/2020_gcs_with_ml/data/cme_seg_20240912/'
model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v5"
test_cases_file = model_path+"/validation_cases.csv"
model_version="v5"
opath= model_path+"/test_output_diego"
file_ext=".png"
trained_model = '49.torch'
imageSize=[512,512]
test_ncases = 10
mask_thresholds = [0.7] #[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99] # only px with scrore avobe this value are considered in the mask.
gpu=0# GPU to use
masks2use=[2] # index of the masks to read (should be the CME mask)
breakpoint()
#main
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unless its not available
print(f'Using device:  {device}')
#load image paths from test_cases_file
imgs = []
with open(test_cases_file, 'r') as f:
    for line in f:
        imgs.append(line.strip())
nn_seg = neural_cme_segmentation(device, pre_trained_model = model_path + "/"+ trained_model, version=model_version)
rnd_idx = random.sample(range(len(imgs)), test_ncases)
all_scr = []
all_loss = []
for mask_threshold in mask_thresholds:
    this_case_scr =[]
    this_case_loss = []    
    for ind in range(len(rnd_idx)):
        idx = rnd_idx[ind]
        #reads image
        file=os.listdir(imgs[idx])
        file=[f for f in file if f.endswith(file_ext)]
        print(f'Inference No. {ind}/{len(rnd_idx)} using image: {idx} and mask threshold case No. {mask_thresholds.index(mask_threshold)+1}/{len(mask_thresholds)}')
        images = cv2.imread(os.path.join(imgs[idx], file[0]))
        images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)

        # reads mask
        maskDir=os.path.join(imgs[idx], "mask") #path to the mask iamge corresponding to the random image
        masks=[]
        vesMask=os.listdir(maskDir) #all masks in the mask directory
        vesMask.sort(key=lambda x: int(x.split("_")[-1].split(".")[0])) #sort masks files by the ending number
        vesMask = [vesMask[i] for i in masks2use][0]
        vesMask = cv2.imread(maskDir+'/'+vesMask, 0) #reads the mask image in greyscale 
        vesMask = (vesMask > 0).astype(np.uint8) #The mask image is stored in 0–255 format and is converted to 0–1 format
        if np.mean(vesMask) == 0 and np.max(vesMask) == 0:
            print('Error: Full zero mask in path', maskDir, 'skipping')
            breakpoint()
        # makes inference and returns only the mask with smallest loss
        img, masks, scores, labels, boxes, loss  = nn_seg.test_mask(images, vesMask, mask_threshold=mask_threshold)
        breakpoint()

        if masks is not None:
            this_case_loss.append(loss) 
            this_case_scr.append(scores)
            # plot the predicted mask
            if len(mask_thresholds)==1:
                os.makedirs(opath, exist_ok=True)
                ofile = opath+"/img_"+str(idx)+'.png'
                plot_to_png(ofile, [img], [[masks]], [vesMask], scores=[[scores]], labels=[[labels]], boxes=[[boxes]], 
                            mask_threshold=mask_threshold, scr_threshold=0.1, version=model_version)
    all_scr.append(this_case_scr)
    all_loss.append(this_case_loss)

    # plot stats for a single mask threshold
    if len(mask_thresholds)==1:
        this_case_scr = np.array(this_case_scr)
        fig= plt.figure(figsize=(10, 5)) 
        ax = fig.add_subplot() 
        ax.hist(this_case_scr,bins=30)
        ax.set_title(f'Mean scr: {np.mean(this_case_scr)} ; For mask threshold: {mask_threshold}  and test_ncases: {test_ncases} \n\
                     % of scr>0.8: {len(this_case_scr[this_case_scr>0.8])/len(this_case_scr)}')
        ax.set_yscale('log')
        fig.savefig(opath+"/test_scores.png")

        this_case_loss = np.array(this_case_loss)
        fig= plt.figure(figsize=(10, 5))
        ax = fig.add_subplot()
        ax.hist(this_case_loss,bins=30)
        ax.set_title(f'Mean loss: {np.mean(this_case_loss)} ; For mask threshold: {mask_threshold} and test_ncases: {test_ncases}')
        ax.set_yscale('log')
        fig.savefig(opath+"/test_loss.png")

# for many mask thresholds plots mean and std loss and scr vs mask threshold
if len(mask_thresholds)>1:
    all_scr = np.array(all_scr)
    fig= plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    ax.errorbar(mask_thresholds, np.mean(all_scr,axis=1), yerr=np.std(all_scr,axis=1), fmt='o', color='black', ecolor='lightgray', elinewidth=3)
    ax.set_title(f'Mean scr vs mask threshold')
    ax.set_xlabel('mask threshold')
    ax.set_ylabel('Score')
    ax.grid()
    fig.savefig(opath+"/test_scr_vs_mask_threshold.png")

    all_loss = np.array(all_loss)
    fig= plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    ax.errorbar(mask_thresholds, np.mean(all_loss,axis=1), yerr=np.std(all_loss,axis=1), fmt='o', color='black', ecolor='lightgray', elinewidth=3)
    ax.set_title(f'Mean loss vs mask threshold')
    ax.set_xlabel('mask threshold')
    ax.set_ylabel('Loss')
    ax.grid()
    fig.savefig(opath+"/test_mean_loss_vs_mask_threshold.png")
print('Results saved in:', opath)
print('Done :-D')


