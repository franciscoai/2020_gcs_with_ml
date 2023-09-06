def normalize(image):
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

def plot_to_png(ofile,filter_label,orig_img, masks, title=None, labels=None, boxes=None, scores=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
    mask_threshold = 0.5 # value to consider a pixel belongs to the object
    scr_threshold = 0.3 # only detections with score larger than this value are considered
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
            b=boxes[i][filter_label]
            if scores is not None:
                scr = scores[i][filter_label]
            else:
                scr = 0   
            if scr > scr_threshold:             
                masked = nans.copy()
                masked[:, :][masks[i][filter_label] > mask_threshold] = 0              
                axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1) # add mask
                box =  mpl.patches.Rectangle(b[0:2],b[2]-b[0],b[3]-b[1], linewidth=2, edgecolor=color[filter_label], facecolor='none') # add box
                axs[i+1].add_patch(box)
                if labels is not None:
                        axs[i+1].annotate(obj_labels[labels[i][filter_label]]+':'+'{:.2f}'.format(scr),xy=b[0:2], fontsize=15, color=color[filter_label])
            
                
    # axs[0].set_title(f'Cor A: {len(boxes[0])} objects detected') 
    # axs[1].set_title(f'Cor B: {len(boxes[1])} objects detected')               
    # axs[2].set_title(f'Lasco: {len(boxes[2])} objects detected')     
    #if title is not None:
    #    fig.suptitle('\n'.join([title[i]+' ; '+title[i+1] for i in range(0,len(title),2)]) , fontsize=16)   
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

import os
import pickle
import glob
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from ext_libs.rebin import rebin



odir="/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat_L1/cor2_a/"
ext_folders = os.listdir(odir)
for ext_folder in ext_folders:
    odir_filter=odir+ext_folder+"/filtered"
    
    #if ext_folder=="299":
    if os.path.exists(odir_filter):
        
        del_files=os.listdir(odir_filter)
        if len(del_files)>0:
            [os.remove(odir_filter+"/"+i) for i in del_files if i.endswith("2A.png")]
        print('Processing '+ext_folder)
        csv_path=odir_filter+"/"+ext_folder+".csv"
        df=pd.read_csv(csv_path)
        for i in range(len(df)):
            filename=os.path.basename(df["PATH"][i])[0:-4]
            print('Processing file '+filename+".fts")
            opath =odir_filter+"/"+filename+".png"
            pkl_path=odir+ext_folder+"/"+filename+"_nn_seg.pkl"
            img_path=df["PATH"][i]
            img=read_fits(img_path)
            images = normalize(img) # normalize to 0,1
            with open(pkl_path, 'rb') as archivo_pickle:
                mask = pickle.load(archivo_pickle)
                scores = pickle.load(archivo_pickle)
                labels= pickle.load(archivo_pickle)
                boxes= pickle.load(archivo_pickle)
            filter_label=df["MASK_LABEL"][i]
            plot_to_png(opath,filter_label, [images], [mask], title=[filename], labels=[labels], boxes=[boxes], scores=[scores])
        