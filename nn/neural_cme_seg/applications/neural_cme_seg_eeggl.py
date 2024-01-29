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
sys.path.append(asd)
asd2=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(asd2)
asd3=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(asd3)
from neural_cme_seg_diego import neural_cme_segmentation
from ext_libs.rebin import rebin
import pandas as pd
from datetime import datetime, timedelta
import glob
from scipy.ndimage.filters import gaussian_filter
import pickle
from scipy.interpolate import interp2d

def read_fits(file_path, header=False, imageSize=[512,512],smooth_kernel=[0,0]):
    try:       
        img = fits.open(file_path)[0].data
        img=img.astype("float32")
        img = gaussian_filter(img, sigma=smooth_kernel)
        if imageSize:
            img = rebin(img, imageSize, operation='mean')  
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
    
def plot_to_png(ofile, orig_img, masks, scr_threshold=0.25, mask_threshold=0.7 , title=None, labels=None, boxes=None, scores=None):
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

#--------------------------------------------------------------------------------------------------------------------

def plot_to_png2(ofile, orig_img, event, all_center, mask_threshold, scr_threshold, title=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
    color=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w']
    obj_labels = ['Back', 'Occ','CME','N/A']
    masks=event['MASK']
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(orig_img[0]), np.nan)
    fig, axs = plt.subplots(1, len(orig_img)*2, figsize=(20, 10))
    axs = axs.ravel()
    
    for i in range(len(orig_img)):
        axs[i].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')
        axs[i].axis('off')
        axs[i+1].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')        
        axs[i+1].axis('off') 
        for b in range(len(event['LABEL'])):
            scr = event['SCR'][b]
            if scr > scr_threshold:             
                masked = nans.copy()            
                masked[:, :][masks[b] > mask_threshold] = event['CME_ID'][b]           
                axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1) # add mask
                
                box =  mpl.patches.Rectangle(event['BOX'][b][0:2], event['BOX'][b][2]- event['BOX'][b][0], event['BOX'][b][3]- event['BOX'][b][1], linewidth=2, edgecolor=color[int(event['CME_ID'][b])] , facecolor='none') # add box
                axs[i+1].add_patch(box)
                axs[i+1].scatter(round(all_center[0][0]), round(all_center[0][1]), color='red', marker='x', s=100)
                axs[i+1].annotate(obj_labels[event['LABEL'][b]]+':'+'{:.2f}'.format(scr),xy=event['BOX'][b][0:2], fontsize=15, color=color[int(event['CME_ID'][b])])
     
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

#specific for eeggl functions
def rebin_interp(img,new_size=[512,512]):
    #rebinea imagenes, por default a 512
    x1 = np.linspace(0, img.shape[0], img.shape[0])-img.shape[0]
    y1 = np.linspace(0, img.shape[1], img.shape[1])-img.shape[1]
    fun = interp2d(x1, y1, img, kind='linear')
    x = np.linspace(0, img.shape[0], new_size[0])-img.shape[0]
    y = np.linspace(0, img.shape[1], new_size[1])-img.shape[1]
    Z3 = fun(x, y)
    return Z3

def vec_to_matrix(vec,dim_matrix):
    #dim_matrix 300 en el caso de eeggl sta/b
    matrix = [vec[i:i+dim_matrix] for i in range(0, len(vec), dim_matrix)]
    matrix = np.array(matrix)
    matrix = matrix.T #traspongo xq la imagen se lee fila por fila desde 0,0 esquina izq arriba hacia abajo y luego fila por fila hacia la derecha.
    matrix = matrix.astype("float32")   
    return matrix

def create_header(matrix,time):
    header = fits.Header()
    header['NAXIS1'] = matrix.shape[0]
    header['NAXIS2'] = matrix.shape[1]
    header['time'] = time

def read_dat(file,path="",dim_matrix=300):
    pos_x=[]
    pos_y=[]
    pos_wl=[]
    pos_pb=[]
    #print(path+file)
    
    with open(path+file, 'r') as file:
    # Read all lines into a list
        line0, line1, line2, line3, line4, line5, line6, line7 = [file.readline() for _ in range(8)]
        time_event = line3
        time_event_start = line4
        time_seconds = line5

    # Process each line
        for line in file:            
            x, y, wl, pb = map(float, line.split())
            pos_x.append(x)
            pos_y.append(y)
            pos_wl.append(wl)
            pos_pb.append(pb)
    
    x_pos  = vec_to_matrix(pos_x,dim_matrix)
    y_pos  = vec_to_matrix(pos_y,dim_matrix)
    wl_pos = vec_to_matrix(pos_wl,dim_matrix)
    pb_pos = vec_to_matrix(pos_pb,dim_matrix)
    hdr = create_header(matrix=wl_pos,time=time_event)
    return x_pos, y_pos, wl_pos, pb_pos,hdr


#main
#------------------------------------------------------------------Testing the CNN--------------------------------------------------------------------------
#aux_in = "/gehme-gpu"
aux_in = "/gehme-gpu"
model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4"
model_version="v4"
#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2011_02_15/data/cor2b/'
#----------------eeggl
#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2011_02_15/eeggl_synthetic/run005/'

ipath = aux_in+'/projects/2023_eeggl_validation/data/2012-07-12/Cor2A/lvl1/'
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
#ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2010-04-03/C2/lvl1/'

#----------------
#aux="oculter_60_250/"
aux="occ_medida_RD_infer2/"
aux_out='/gehme'
opath= aux_out+'/projects/2023_eeggl_validation/output/2012-07-12/Cor2A/'+aux
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
#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2010-04-03/C2/'+aux

#opath= '/gehme-gpu/projects/2023_eeggl_validation/output/2010-04-03/'
#-------------

file_ext=".dat"
#file_ext=".fts"
trained_model = '6000.torch'
#-----------------------------
eeggl = 'True'
base_difference = False
running_difference = True
instr='cor2_a'
#instr='cor2_b'
#instr='lascoC2'
infer_event2=False
infer_event1=True

#----------------------------

if instr != 'lascoC2':
    occ_center=[[256,256],[260,247]]
occ_size = [50,52]
mask_threshold = 0.6 # value to consider a pixel belongs to the object
scr_threshold = 0.4 # only detections with score larger than this value are considered



#main
gpu=0 # GPU to use
#If gpu 1 is out of ram, use gpu=0 or cpu. Check gpu status using nvidia-smi command on terminal.
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unless its not available
print(f'Using device:  {device}')
#breakpoint()
#OBS: files is a list of all files but NOT in correct temporal order.
#loads images
files = os.listdir(ipath)
files = [os.path.join(ipath, e) for e in files]
breakpoint()
files = [e for e in files if os.path.splitext(e)[1] == file_ext]
breakpoint()
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
list_name = 'list.txt'
if eeggl:
    list_name = 'lista_sta_cor2_ordenada.txt'
with open(ipath+list_name, 'r') as file:
    lines = file.readlines()
image_names = [line.strip() for line in lines]


if infer_event2:
    all_images=[]
    all_dates=[]
    all_occ_size=[]
    all_plate_scl=[]
    file_names=[]
    all_headers=[]
    all_center=[]


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
if instr=="lascoC2":
    occ_size = 90
    
if infer_event2:
    os.makedirs(opath+'mask_props/', exist_ok=True)

for j in range(1,len(image_names)):

    if ~eeggl:        
        if base_difference:
            #First image set to background
            img0, hdr0 = read_fits(ipath+image_names[0],header=True)
            img1, hdr1 = read_fits(ipath+image_names[j],header=True)
        #    img_diff[img_mask == 0] = 0
            
        if running_difference:
            img0, hdr0 = read_fits(ipath+image_names[j-1],header=True)
            img1, hdr1 = read_fits(ipath+image_names[j  ],header=True)

        #if 'occ_center' != locals():
        if instr =="lascoC2":    
            occ_center=[hdr1["crpix1"],hdr1["crpix2"]]
            #En esta resta asumimos que ambas imagenes tienen igual centerpix1/2, y ambas son north up.
    if eeggl:
        if running_difference:
            x_pos, y_pos, wl_mat0, pb_mat0,hdr0 = read_dat(image_names[j-1],path=ipath,dim_matrix=300)
            img0 = rebin_interp(wl_mat0,new_size=[512,512])
            x_pos, y_pos, wl_mat1, pb_mat1,hdr1 = read_dat(image_names[j  ],path=ipath,dim_matrix=300)
            img1 = rebin_interp(wl_mat1,new_size=[512,512])


    img_diff = img1 - img0
    breakpoint()
    if infer_event1:
    #Infer1
        #orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, occulter_size=60,centerpix=[255,255+10],occulter_size_ext=240) 
    
    #Cor2 con mascara original
        #orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, getmask=True,hdr=hdr1) 
    
        if instr=='cor2_b':
            #Cor2 con mascara centrada ---> Cor2B me funciona mejor con esto, usando crpix1/2
            orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, occulter_size=70,centerpix=occ_center,occulter_size_ext=250)
        
        if instr=='cor2_a':
            #Cor2 con mascara centrada (a medida) ---> Cor2A me funciona mejor con esto
            orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, occulter_size=occ_size,centerpix=occ_center,occulter_size_ext=250)
    
    
        if instr=='lascoC2':    
            #C2
            orig_img, masks, scores, labels, boxes  = nn_seg.infer(img_diff, model_param=None, resize=False, occulter_size=occ_size,centerpix=occ_center,occulter_size_ext=300)
        
    # plot the predicted mask
        ofile = opath+"/"+os.path.basename(image_names[j])+'infer1.png'
        plot_to_png(ofile, [orig_img], [masks], scores=[scores], labels=[labels], boxes=[boxes])
        
        
        #contour_plot = plt.contour(asd)
        #plt.colorbar(contour_plot, label='Matrix Values')
        #plt.savefig('contour'+ofile)
        #plt.close()
    #----------------------------------------------------------------------------------------
    #infer2
    if infer_event2:
        
        if instr != 'lascoC2':    
            date = datetime.strptime(image_names[j][0:-10],'%Y%m%d_%H%M%S')
        else:
            date_string = hdr1['fileorig'][:-4]
            if int(date_string[:2]) < 94: #soho was launched in 1995
                aux = '20'
            else:
                aux = '19'
            date_string = aux + date_string
            date = datetime.strptime(date_string,'%Y%m%d_%H%M%S')
        #breakpoint()
    
        plt_scl = hdr1['CDELT1']
        all_center.append(occ_center)
        all_plate_scl.append(plt_scl)                    
        all_images.append(img_diff)
        all_dates.append(date)
        all_occ_size.append(occ_size)
        file_names.append(image_names[j])
        all_headers.append(hdr1)

if infer_event2:
    ok_orig_img,ok_dates, df =  nn_seg.infer_event2(all_images, all_dates, filter=filter, plate_scl=all_plate_scl, occulter_size=all_occ_size,centerpix=all_center,plot_params=opath+'mask_props')
    zeros = np.zeros(np.shape(ok_orig_img[0]))
    all_idx=[]
    #new_ok_dates=[datetime.utcfromtimestamp(dt.astype(int) * 1e-9) for dt in ok_dates]

    for date in all_dates:
        if date not in ok_dates:
            idx = all_dates.index(date)
            all_idx.append(idx)
        
    file_names = [file_name for h, file_name in enumerate(file_names) if h not in all_idx]
    all_center =[all_center for h,all_center in enumerate(all_center) if h not in all_idx]
    all_plate_scl =[all_plate_scl for h,all_plate_scl in enumerate(all_plate_scl) if h not in all_idx]
    all_dates =[all_dates for h,all_dates in enumerate(all_dates) if h not in all_idx]
    all_occ_size =[all_occ_size for h,all_occ_size in enumerate(all_occ_size) if h not in all_idx]
    all_headers =[all_headers  for h,all_headers in enumerate(all_headers) if h not in all_idx]

    for m in range(len(ok_dates)):
        event = df[df['DATE_TIME'] == ok_dates[m]].reset_index(drop=True)
        image=ok_orig_img[m]
        for n in range(len(event['MASK'])):
            if event['SCR'][n] > scr_threshold:             
                masked = zeros.copy()
                masked[:, :][(event['MASK'][n]) > mask_threshold] = 1
                # safe fits
                
                ofile_fits = os.path.join(os.path.dirname(opath), file_names[m]+"_CME_ID_"+str(int(event['CME_ID'][n]))+'.fits')
                h0 = all_headers[m]
                # adapts hdr because we use smaller im size
                sz_ratio = np.array(masked.shape)/np.array([h0['NAXIS1'], h0['NAXIS2']])
                #h0['NAXIS1'] = masked.shape[0]
                #h0['NAXIS2'] = masked.shape[1]
                h0['CDELT1'] = h0['CDELT1']/sz_ratio[0]
                h0['CDELT2'] = h0['CDELT2']/sz_ratio[1]
                #h0['CRPIX2'] = int(h0['CRPIX2']*sz_ratio[1])
                #h0['CRPIX1'] = int(h0['CRPIX1']*sz_ratio[1]) 
                fits.writeto(ofile_fits, masked, h0, overwrite=True, output_verify='ignore')
        plot_to_png2(opath+file_names[m]+"infer2.png", [ok_orig_img[m]], event,[all_center[m]],mask_threshold=mask_threshold,scr_threshold=scr_threshold, title=[file_names[m]])  

print("Program finished without errors")







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
