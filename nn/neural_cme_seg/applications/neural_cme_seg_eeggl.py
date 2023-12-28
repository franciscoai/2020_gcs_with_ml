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
asd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(asd)
sys.path.append(asd)
asd2=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(asd2)
sys.path.append(asd2)
asd3=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
print(asd3)
sys.path.append(asd3)
from neural_cme_seg_diego import neural_cme_segmentation
from ext_libs.rebin import rebin

def read_fits(file_path, header=False, imageSize=[512,512]):
    try:       
        img = fits.open(file_path)[0].data
        if imageSize:
            
            img = rebin(img, imageSize)  
            if header:
                hdr = fits.open(file_path)[0].header
                naxis_original = hdr["naxis1"]
                hdr["naxis1"]=imageSize[0]
                hdr["naxis2"]=imageSize[1]
                hdr["crpix1"]=hdr["crpix1"]/(naxis_original/imageSize[0])
                hdr["crpix2"]=hdr["crpix2"]/(naxis_original/imageSize[0])
        # flips y axis to match the orientation of the images
        img = np.flip(img, axis=0) 
        if header:
            return img, hdr
        else:
            return img 
    except:
        print(f'WARNING. I could not find file {file_path}')
        return None
    
def plot_to_png(ofile, orig_img, masks, scr_threshold=0.05, mask_threshold=0.55 , title=None, labels=None, boxes=None, scores=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
     # only detections with score larger than this value are considered
    color=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w']
    obj_labels = ['Back', 'Occ','CME','N/A']
    #
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(orig_img[0]), np.nan)
    fig, axs = plt.subplots(1, len(orig_img)*2, figsize=(20, 10))
    axs = axs.ravel()
    for i in range(len(orig_img)): #1 iteracion por imagen?
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
                print(nb)
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
#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2011_02_15/data/cor2b/'
#----------------eeggl
#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2011_02_15/eeggl_synthetic/run005/'

#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2012-07-12/Cor2A/lvl1/'
#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2012-07-12/Cor2B/lvl1/'
#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2012-07-12/C2/lvl1/'

#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2011-02-14/Cor2A/lvl1/'
#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2011-02-14/Cor2B/lvl1/'
#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2011-02-15/C2/lvl1/'

#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2013-03-15/Cor2A/lvl1/'
#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2013-03-15/Cor2B/lvl1/'
#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2013-03-15/C2/lvl1/'

#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2013-09-29/Cor2A/lvl1/'
#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2013-09-29/Cor2B/lvl1/'
#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2013-09-29/C2/lvl1/'

#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2010-04-03/Cor2A/lvl1/'
#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2010-04-03/Cor2B/lvl1/'
ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2010-04-03/C2/lvl1/'

#----------------
#aux="oculter_60_250/"
aux="occ_medida_RD/"

#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2012-07-12/Cor2A/'+aux
#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2012-07-12/Cor2B/'+aux
#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2012-07-12/C2/'+aux

#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2011-02-15/Cor2A/'+aux
#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2011-02-15/Cor2B/'+aux
#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2011-02-15/C2/'+aux

#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2013-03-15/Cor2A/'+aux
#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2013-03-15/Cor2B/'+aux
#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2013-03-15/C2/'+aux

#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2013-09-29/Cor2A/'+aux
#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2013-09-29/Cor2B/'+aux
#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2013-09-29/C2/'+aux

#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2010-04-03/Cor2A/'+aux
#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2010-04-03/Cor2B/'+aux
opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2010-04-03/C2/'+aux

#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2010-04-03/'
file_ext=".fts"
trained_model = '6000.torch'
#-----------------------------
base_difference = False
running_difference = True

#instr='cor2_a'
#instr='cor2_b'
instr='lascoC2'
#----------------------------
occ_center=None
#occ_center=[[256,256],[260,247]]
occ_size = [50,52]

#main
gpu=0 # GPU to use
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unless its not available
print(f'Using device:  {device}')

#OBS: files is a list of all files but NOT in correct temporal order.
#loads images
files = os.listdir(ipath)
files = [os.path.join(ipath, e) for e in files]
files = [e for e in files if os.path.splitext(e)[1] == file_ext]

#loads nn model
nn_seg = neural_cme_segmentation(device, pre_trained_model = model_path + "/"+ trained_model, version=model_version)

os.makedirs(opath, exist_ok=True)
#inference on all images

#Select masks for different instruments
#if instrument == 'cor2':
#    mask = "/gehme-gpu/projects/2023_eeggl_validation/repo_diego/2020_gcs_with_ml/instrument_mask/mask_cor2_512.fts"
#    img_mask = read_fits(mask)

#El infer debe pasar el header que se utilizara para las nuevas mascaras!! 

#list of all images in temporal order.
with open(ipath+'list.txt', 'r') as file:
    lines = file.readlines()
image_names = [line.strip() for line in lines]

if instr=="cor2_a":
    occ_center=occ_center[0]
    if occ_center:
        occ_size= occ_size[0]
elif instr=="cor2_b":
    occ_size= occ_size[1]
    if occ_center:
        occ_center=occ_center[1]
    #else:
    #    occ_center=[hdr1["crpix1"],hdr1["crpix2"]]

for j in range(1,len(image_names)):
        
    if base_difference:
        #First image set to background
        img0, hdr0 = read_fits(ipath+image_names[0],header=True)
        img1, hdr1 = read_fits(ipath+image_names[j],header=True)
#    img_diff[img_mask == 0] = 0
        
    if running_difference:
        img0, hdr0 = read_fits(ipath+image_names[j-1],header=True)
        img1, hdr1 = read_fits(ipath+image_names[j  ],header=True)
   
    if not occ_center:
        occ_center=[hdr1["crpix1"],hdr1["crpix2"]]
        #En esta resta asumimos que ambas imagenes tienen igual centerpix1/2, y ambas son north up.
    img_diff = img1 - img0
    
    #orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, occulter_size=60,centerpix=[255,255+10],occulter_size_ext=240) 
    
    #Cor2 con mascara original
    #orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, getmask=True,hdr=hdr1) 
    
    #Cor2 con mascara centrada ---> Cor2B me funciona mejor con esto, usando crpix1/2
    #orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, occulter_size=60,centerpix=occ_center,occulter_size_ext=255)
    
    #Cor2 con mascara centrada (a medida) ---> Cor2A me funciona mejor con esto
    #orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, occulter_size=occ_size,centerpix=occ_center,occulter_size_ext=250)
    
    #C2
    orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, occulter_size=90,centerpix=occ_center,occulter_size_ext=300)
    
    #centerpix=[255,265])
    # plot the predicted mask
    ofile = opath+"/"+os.path.basename(image_names[j])+'.png'
    plot_to_png(ofile, [orig_img], [masks], scores=[scores], labels=[labels], boxes=[boxes])
    

"""
for f in files:
    print(f'Processing file {f}')
    if file_ext == ".fits" or file_ext == ".fts":
        img = read_fits(f)
    else:
        img = cv2.imread(f)
    breakpoint()
    orig_img, masks, scores, labels, boxes  = nn_seg.infer(img, model_param=None, resize=False, occulter_size=0)
    # plot the predicted mask
    ofile = opath+"/"+os.path.basename(f)+'.png'
    plot_to_png(ofile, [img], [masks], scores=[scores], labels=[labels], boxes=[boxes])
"""

"""    
    #-----------------------------------------------------------------------
    valores = img_diff[~np.isnan(img_diff)]
    hist, bins, _ = plt.hist(valores, bins=30, range=((np.min(valores)+np.std(valores)), (np.max(valores)-np.std(valores))), color='blue', alpha=0.7, density=True)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img_diff, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.plot(hist, color='gray')
    type_info = img_diff.dtype
    min_value = round(np.nanmin(img_diff),2)
    max_value = round(np.nanmax(img_diff),2)
    mean_value = round(np.nanmean(img_diff),2)
    plt.title(f'Type: {type_info}, Min: {min_value}')
    plt.xlabel(f'Max: {max_value}, Mean: {mean_value:.2f}')
    plt.show()
    plt.savefig("/gehme-gpu/projects/2023_eeggl_validation/output/histo_real.png")
    #-----------------------------------------------------------------------
    """
