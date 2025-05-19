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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

from ext_libs.rebin import rebin
import csv
from nn.neural_cme_seg.neural_cme_seg import neural_cme_segmentation
import datetime
from scipy.ndimage.filters import gaussian_filter
from scipy.io import readsav
from pyGCS_raytrace import pyGCS
from pyGCS_raytrace.rtraytracewcs import rtraytracewcs
from nn_training.get_cme_mask import get_mask_cloud,get_cme_mask

def center_rSun_pixel(headers, plotranges, sat):
    '''
    Gets the location of Suncenter in deg
    '''    
    x_cS = headers[sat]['CRPIX1']*(plotranges[sat][1] - plotranges[sat][0])/headers[sat]['NAXIS1'] + plotranges[sat][0]
    y_cS = headers[sat]['CRPIX2']*(plotranges[sat][3] - plotranges[sat][2])/headers[sat]['NAXIS2'] + plotranges[sat][2]
            
    return x_cS, y_cS

def correct_path(s):
    s = s.replace("(1)", "")
    s = s.replace("(2)", "")
    return s

def convert_string(s, level):   
    if level==1:
        s = s.replace("preped/", "")
        s = s.replace("L0", "L1")
        s = s.replace("_0B", "_1B")
        s = s.replace("_04", "_14")
        s = s.replace("level1/", "")            
    return s

def get_paths_cme_exp_sources():
    """
    Read all files for selected events of the CME exp sources project
    """
    data_path='/gehme/data' #Path to the dir containing /sdo ,/soho and /stereo data directories as well as the /Polar_Observations dir.
    gcs_path='/gehme/projects/2019_cme_expansion/Polar_Observations/Polar_Documents/francisco_backup_20240812_diego_modified/GCSs'     #Path with our GCS data directories
    lasco_path=data_path+'/soho/lasco/level_1/c2'     #LASCO proc images Path
    secchipath=data_path+'/stereo/secchi/L1'
    level=1 # set the reduction level of the images
    #events to read
    dates =  ['20101212', '20101214', '20110317', '20110605', '20130123', '20130129',
             '20130209', '20130424', '20130502', '20130517', '20130527', '20130608']
    # pre event iamges per instrument
    pre_event =["/soho/lasco/level_1/c2/20101212/25354377.fts",
                "/stereo/secchi/L1/a/seq/cor1/20101212/20101212_022500_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20101212/20101212_023500_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20101212/20101212_015400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20101212/20101212_015400_14c2B.fts",
                "/soho/lasco/level_1/c2/20101214/25354679.fts",
                "/stereo/secchi/L1/a/seq/cor1/20101214/20101214_150000_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20101214/20101214_150000_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20101214/20101214_152400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20101214/20101214_153900_14c2B.fts",
                "/soho/lasco/level_1/c2/20110317/25365446.fts",
                "/stereo/secchi/L1/a/seq/cor1/20110317/20110317_103500_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20110317/20110317_103500_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20110317/20110317_115400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20110317/20110317_123900_14c2B.fts",
                "/soho/lasco/level_1/c2/20110605/25374823.fts",
                "/stereo/secchi/L1/a/seq/cor1/20110605/20110605_021000_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20110605/20110605_021000_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20110605/20110605_043900_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20110605/20110605_043900_14c2B.fts",
                "/soho/lasco/level_1/c2/20130123/25445617.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130123/20130123_131500_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130123/20130123_125500_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130123/20130123_135400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130123/20130123_142400_14c2B.fts",
                "/soho/lasco/level_1/c2/20130129/25446296.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130129/20130129_012500_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130129/20130129_012500_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130129/20130129_015400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130129/20130129_015400_14c2B.fts",
                "/soho/lasco/level_1/c2/20130209/25447666.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130209/20130209_054000_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130209/20130209_054500_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130209/20130209_062400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130209/20130209_062400_14c2B.fts",
                "/soho/lasco/level_1/c2/20130424_1/25456651.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130424/20130424_051500_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130424/20130424_051500_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130424/20130424_055400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130424/20130424_065400_14c2B.fts",
                "/soho/lasco/level_1/c2/20130502/25457629.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130502/20130502_045000_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130502/20130502_050000_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130502/20130502_012400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130502/20130502_053900_14c2B.fts"
                "/soho/lasco/level_1/c2/20130517/25459559.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130517/20130517_194500_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130517/20130517_194500_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130517/20130517_203900_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130517/20130517_205400_14c2B.fts",
                "/soho/lasco/level_1/c2/20130527_2/25460786.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130527/20130527_183000_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130527/20130527_183000_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130527/20130527_192400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130527/20130527_195400_14c2B.fts",
                "/soho/lasco/level_1/c2/20130608/25462149.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130607/20130607_221500_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130607/20130607_223000_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130607/20130607_225400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130607/20130607_232400_14c2B.fts"]
    pre_event = [data_path + f for f in pre_event]

    #get file event for each event
    temp = os.listdir(gcs_path)
    temp2 = [name for name in temp if os.path.isdir(os.path.join(gcs_path, name))]
    events_path = [os.path.join(gcs_path,d) for d in temp2 if str.split(d,'_')[-1] in dates]
    
    #gets .savs, andthe cor and lasco file event for each time instant in each event
    event = []
    for ev in events_path:
        cdict = {'date':[],'pro_files':[],'sav_files':[],'ima1':[], 'ima0':[],'imb1':[], 'imb0':[], 'lasco1':[],'lasco0':[],'pre_ima':[],'pre_imb':[],'pre_lasco':[]}
        tinst = os.listdir(ev)
        sav_files = sorted([os.path.join(ev,f) for f in tinst if (f.endswith('.sav') and 'm1.' not in f and 'm2.' not in f)])
        pro_files = sorted([os.path.join(ev,f) for f in tinst if (f.endswith('.pro') and 'fit_' not in f and 'tevo_' not in f and 'm1.' not in f and 'm2.' not in f and 'download' not in f and 'data' not in f)])
        #breakpoint()
        if len(sav_files) != len(pro_files):
            os.error('ERROR. Found different number of .sav and .pro files')
            breakpoint()
            sys.exit
        # reads the lasco and stereo files from within each pro
        ok_pro_files = []
        ok_sav_files = []
        #breakpoint()
        for index ,f in enumerate(pro_files):
            with open(f) as of:
                for line in of:
                    if 'ima=sccreadfits(' in line:
                        cline = secchipath +line.split('\'')[1]
                        if 'cor1' in cline:
                            break
                        #    cor = 'cor1'
                        if 'cor2' in cline:
                            cor = 'cor2'
                        cdate = cline[cline.find('/preped/')+8:cline.find('/preped/')+16]
                        cline = convert_string(cline, level)
                        cline = correct_path(cline)
                        cdict['ima1'].append(cline)
                        ok_pro_files.append(f)
                        ok_sav_files.append(sav_files[index])
                        cpre = [s for s in pre_event if (cdate in s and cor in s and '/a/' in s )]
                        if len(cpre) == 0:
                            print(f'Cloud not find pre event image for {cdate}')
                            breakpoint()
                        cdict['pre_ima'].append(cpre[0])
                        cdict['pre_imb'].append([s for s in pre_event if (cdate in s and  cor in s and '/a/' in s )][0])
                    if 'imaprev=sccreadfits(' in line:
                        cline = convert_string(secchipath +line.split('\'')[1], level)
                        if 'cor1' in cline:
                            break
                        cline = correct_path(cline)
                        cdict['ima0'].append(cline)
                    if 'imb=sccreadfits(' in line:
                        cline = convert_string(secchipath +line.split('\'')[1], level)
                        if 'cor1' in cline:
                            break
                        cline = correct_path(cline)
                        cdict['imb1'].append(cline)
                    if 'imbprev=sccreadfits(' in line:
                        cline = convert_string(secchipath +line.split('\'')[1], level)
                        if 'cor1' in cline:
                            break
                        cline = correct_path(cline)
                        cdict['imb0'].append(cline)          
                    if 'lasco1=readfits' in line:
                        cline = lasco_path +line.split('\'')[1]
                        cline = correct_path(cline)
                        cdict['lasco1'].append(cline)  
                        cdate = cline[cline.find('/preped/')+8:cline.find('/preped/')+16]
                        cpre= [s for s in pre_event if (cdate in s and '/c2/' in s)]
                        if len(cpre) == 0:
                            print(f'Cloud not find pre event image for {cdate}')
                            breakpoint()                        
                        cdict['pre_lasco'].append(cpre[0])                                           
                    if 'lasco0=readfits' in line:
                        cline = lasco_path +line.split('\'')[1]
                        cline = correct_path(cline)
                        cdict['lasco0'].append(cline)
                #ok_sav_files.append(sav_files[index])
        cdict['date']=ev
        cdict['pro_files']=ok_pro_files
        cdict['sav_files']=ok_sav_files                                                      
        event.append(cdict)
        #breakpoint()
    return event

def read_fits_old(file_path, header=False, imageSize=[512,512]):
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

def read_fits(file_path, header=False, imageSize=[512,512],smooth_kernel=[0,0]):
    try:       
        img = fits.open(file_path)[0].data
        img=img.astype("float32")
        img = gaussian_filter(img, sigma=smooth_kernel)
        if imageSize:
            img = rebin(img, imageSize, operation='mean')  
            #img = rebin_interp(img, new_size=[512,512])#si la imagen no es cuadrada usar esto
            if header:
                hdr = fits.open(file_path)[0].header
                naxis_original1 = hdr["naxis1"]
                naxis_original2 = hdr["naxis2"]
                hdr["naxis1"]=imageSize[0]
                hdr["naxis2"]=imageSize[1]
                hdr["crpix1"]=hdr["crpix1"]/(naxis_original1/imageSize[0])
                hdr["crpix2"]=hdr["crpix2"]/(naxis_original2/imageSize[1])
                hdr["CDELT1"]=hdr["CDELT1"]*(naxis_original1/imageSize[0])
                hdr["CDELT2"]=hdr["CDELT2"]*(naxis_original2/imageSize[1])
        # flips y axis to match the orientation of the images
        #img = np.flip(img, axis=0) 
        if header:
            return img, hdr
        else:
            return img 
    except:
        print(f'WARNING. I could not find file {file_path}')
        return None

def calculate_metrics(mask1, mask2):
    """
    Calculates precision, recall, dice coefficient, and intersection over union (IoU) 
    between two binary masks.
    Args:
        mask1: The first binary mask (numpy array).
        mask2: The second binary mask (numpy array). Ground truth mask.
    """
    # Flatten the masks for efficient calculations
    mask1_flat = mask1#.flatten()
    mask2_flat = mask2#.flatten()
    if len(mask1_flat) != len(mask2_flat):
        breakpoint()
    # Calculate true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN)
    TP = np.sum(np.logical_and(mask1_flat, mask2_flat))
    FP = np.sum(np.logical_and(mask1_flat, np.logical_not(mask2_flat)))
    FN = np.sum(np.logical_and(np.logical_not(mask1_flat), mask2_flat))
    TN = np.sum(np.logical_and(np.logical_not(mask1_flat), np.logical_not(mask2_flat)))
    # Calculate precision, recall, dice coefficient, and IoU
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    dice = (2 * TP) / (2 * TP + FP + FN) if 2 * TP + FP + FN > 0 else 0
    iou = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0
    return precision, recall, dice, iou

def best_mask_treshold(masks, orig_img, vesMask, mask_thresholds_list=np.arange(0.2, 0.95, 0.05).tolist()):
    #nans = np.full(np.shape(orig_img[0]), np.nan)
    zero = np.full(np.shape(orig_img[0]), 0)
    iou_list = []
    dice_list = []
    prec_list = []
    rec_list = []
    for mask_thresholds in mask_thresholds_list:
        masked = zero.copy()
        #masked[vesMask > 0] = 1
        masked[:, :][masks > mask_thresholds] = 1
        #intersection = np.logical_and(masked, vesMask)
        #union = np.logical_or(masked, vesMask)
        #iou_score = np.sum(intersection) / np.sum(union)
        precision, recall, dice, iou = calculate_metrics(vesMask, masked)
        dice_list.append(dice)
        prec_list.append(precision)
        rec_list.append(recall)
        iou_list.append(iou)
        
    best_mask_threshold_iou  = mask_thresholds_list[np.argmax(iou_list)]
    max_iou = np.max(iou_list)
    best_mask_threshold_dice = mask_thresholds_list[np.argmax(dice_list)]
    max_dice = np.max(dice_list)
    best_mask_threshold_prec = mask_thresholds_list[np.argmax(prec_list)]
    max_prec = np.max(prec_list)
    best_mask_threshold_rec  = mask_thresholds_list[np.argmax(rec_list)]
    max_rec = np.max(rec_list)    
    return best_mask_threshold_iou, best_mask_threshold_dice, best_mask_threshold_prec, best_mask_threshold_rec, max_iou, max_dice, max_prec, max_rec


def save_png(array, ofile=None, range=None):
    '''
    pltos array to an image in ofile without borders axes or anything else
    ofile: if not give only the image object is generated and returned
    range: defines color scale limits [vmin,vmax]
    '''    
    fig = plt.figure(figsize=(4,4), facecolor='white')
    if range is not None:
        vmin=range[0]
        vmax=range[1]
    else:
        vmin=None
        vmax=None
    plt.imshow(array, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)#, aspect='auto')#,extent=plotranges[sat])
    plt.axis('off')         
    if ofile is not None:
        fig.savefig(ofile, facecolor='white', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return 1
    else:
        return fig
    
def plot_to_png(ofile, orig_img, masks, title=None, labels=None, boxes=None, scores=None, save_masks=None, version='v4', scr_threshold = 0.25, masks_gcs=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    save_masks: set to a list of fits headers to save the masks as fits files
    """    
    mask_threshold = 0.75 # value to consider a pixel belongs to the object
    #scr_threshold = 0.25 # only detections with score larger than this value are considered
    color=['b','r','g','k','y','m','c','w','b','r','g','k','y','m','c','w']
    if version=='v4':
        obj_labels = ['Back', 'Occ','CME','N/A']
    elif version=='v5':
        obj_labels = ['Back', 'CME']
    elif version=='A4':
        obj_labels = ['Back', 'Occ','CME']
    elif version=='A6':
        obj_labels = ['Back', 'Occ','CME']
    else:
        print(f'ERROR. Version {version} not supported')
        sys.exit()
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(orig_img[0]), np.nan)
    if masks_gcs is None:
        fig, axs = plt.subplots(2, 3, figsize=[20,10])
    if masks_gcs is not None:
        fig, axs = plt.subplots(3, 3, figsize=[20,10])
    axs = axs.ravel()
    #breakpoint()
    for i in range(len(orig_img)):
        axs[i].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray', origin='lower')
        axs[i].axis('off')
        axs[i+3].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray', origin='lower')        
        axs[i+3].axis('off')        
        if boxes is not None:
            nb = 0
            nb_aux = 1
            iou_mask_list = [0 for _ in range(len(boxes[i]))]
            for b in boxes[i]:
                scr = 0
                if version in ('v4', 'A4', 'A6') and labels[i][nb] == 1: #avoid plotting occulter
                    nb+=1
                    nb_aux+=1
                    continue
                if version=='v5':
                    nb_aux+=1
                if scores is not None:
                    if scores[i][nb] is not None:
                        scr = scores[i][nb]
                if scr > scr_threshold:    
                    try:
                        if np.max(masks_gcs[i]) == 0:    
                            breakpoint()
                    except:
                        masks_gcs[i] = [np.zeros((512,512))] #if no mask is given by gcs proyection.
                        breakpoint()
                    #precision, recall, dice, iou = calculate_metrics(masks_gcs[i][0], masks[i][nb])
                    #best_iou, best_dice, best_prec, best_rec,max_iou, max_dice, max_prec, max_rec = best_mask_treshold(masks[i][nb], orig_img, masks_gcs[i])
                    masked = nans.copy()
                    masked[:, :][masks[i][nb] > mask_threshold] = nb
                    axs[i+3].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1, origin='lower')
                    box =  mpl.patches.Rectangle(b[0:2],b[2]-b[0],b[3]-b[1], linewidth=2, edgecolor=color[nb], facecolor='none') # add box
                    axs[i+3].add_patch(box)
                    mask_thresholdeada = nans.copy()
                    mask_thresholdeada[:, :][masks[i][nb] > mask_threshold] = 1
                    mask_thresholdeada[:, :][masks[i][nb] < mask_threshold] = 0
                    precision, recall, dice, iou = calculate_metrics(masks_gcs[i][0], mask_thresholdeada)
                    iou_mask_list[nb] = iou
                    if labels is not None:
                        axs[i+3].annotate(obj_labels[labels[i][nb]]+':'+'{:.2f}'.format(scr),xy=b[0:2], fontsize=15, color=color[nb])
                    if masks_gcs is not None:
                        axs[i+6].annotate('IoU: '                            ,xy=[10 ,20]    , fontsize=15, color=color[nb])#pasar a negro
                        axs[i+6].annotate(''+'{:.2f}'.format(iou)            ,xy=[30*nb_aux*3,20]  , fontsize=15, color=color[nb])
                        ##axs[i+6].annotate(''+'{:.2f}'.format(max_iou)        ,xy=[50*nb,20]  , fontsize=15, color=color[nb])
                        #axs[i+6].annotate('Dic: '                            ,xy=[10 ,50]    , fontsize=15, color=color[nb])
                        #axs[i+6].annotate(''+'{:.2f}'.format(dice)           ,xy=[30*nb_aux*3,50]  , fontsize=15, color=color[nb])
                        ##axs[i+6].annotate(''+'{:.2f}'.format(max_dice)       ,xy=[50*nb,50]  , fontsize=15, color=color[nb])
                        #axs[i+6].annotate('Pre: '                            ,xy=[10 ,80]    , fontsize=15, color=color[nb])
                        #axs[i+6].annotate(''+'{:.2f}'.format(precision)      ,xy=[30*nb_aux*3,80]  , fontsize=15, color=color[nb])
                        ##axs[i+6].annotate(''+'{:.2f}'.format(max_prec)       ,xy=[50*nb,80]  , fontsize=15, color=color[nb])
                        #axs[i+6].annotate('Rec: '                            ,xy=[10 ,110]   , fontsize=15, color=color[nb])
                        #axs[i+6].annotate(''+'{:.2f}'.format(recall)         ,xy=[30*nb_aux*3,110] , fontsize=15, color=color[nb])
                        ##axs[i+6].annotate(''+'{:.2f}'.format(max_rec)        ,xy=[50*nb,110] , fontsize=15, color=color[nb])
                nb+=1
                nb_aux+=1
            if len(iou_mask_list) > 0:
                max_scr_index = np.argmax(iou_mask_list)
                best_iou, best_dice, best_prec, best_rec,max_iou, max_dice, max_prec, max_rec = best_mask_treshold(masks[i][max_scr_index], orig_img, masks_gcs[i][0])
                axs[i+6].annotate('max_IoU: '+'{:.2f}'.format(max_iou)              ,xy=[10,450]  , fontsize=15, color=color[max_scr_index])
                axs[i+6].annotate('maxthresh: '+'{:.2f}'.format(best_iou)             ,xy=[10,480]  , fontsize=15, color=color[max_scr_index])

        if masks_gcs is not None:
            #breakpoint()
            mask_aux = masks_gcs[i][0].copy()
            mask_aux[masks_gcs[i][0] == 0] = 1
            mask_aux[masks_gcs[i][0] == 1] = 0
            #borders to improve the context of the cme mask.
            try:
                mask_aux[0 ,: ]  = 0
            except:
                breakpoint()
            mask_aux[-1,: ]  = 0
            mask_aux[: ,0 ]  = 0
            mask_aux[: ,-1]  = 0 
            #axs[i+6].imshow(masks_gcs[i][0], vmin=0, vmax=1, cmap='gray', origin='lower')
            axs[i+6].imshow(mask_aux, vmin=0, vmax=1, cmap='gray', origin='lower')
            axs[i+6].axis('off')
        #calculate the scr value that maximizes the IoU. Use only the box/mask with higher iou value.

    axs[0].set_title(f'Cor2 A: {len(boxes[0])} obj detected.  {title[0]}') 
    axs[1].set_title(f'Cor2 B: {len(boxes[1])} obj detected.  {title[1]}')               
    axs[2].set_title(f'Lasco C2: {len(boxes[2])} obj detected.  {title[2]}')     
    #if title is not None:
    #    fig.suptitle('\n'.join([title[i] for i in range(0,len(title))]) , fontsize=16)   
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

    if save_masks is not None:
        zeros = np.zeros(np.shape(orig_img[0]))
        for i in range(len(orig_img)):
            for nb in range(len(boxes[i])):
                scr = 0
                if scores is not None:
                    if scores[i][nb] is not None:
                        scr = scores[i][nb]
                if scr > scr_threshold:             
                    masked = zeros.copy()
                    masked[:, :][masks[i][nb] > mask_threshold] = 1
                    # safe fits
                    ofile_fits = os.path.join(os.path.dirname(ofile), os.path.basename(ofile).split('.')[0] + '_'+str(i)+'.fits')
                    h0 = save_masks[i]
                    # adapts hdr because we use smaller im size
                    sz_ratio = np.array(masked.shape)/np.array([h0['NAXIS1'], h0['NAXIS2']])
                    h0['NAXIS1'] = masked.shape[0]
                    h0['NAXIS2'] = masked.shape[1]
                    h0['CDELT1'] = h0['CDELT1']/sz_ratio[0]
                    h0['CDELT2'] = h0['CDELT2']/sz_ratio[1]
                    h0['CRPIX2'] = int(h0['CRPIX2']*sz_ratio[1])
                    h0['CRPIX1'] = int(h0['CRPIX1']*sz_ratio[1])                    
                    fits.writeto(ofile_fits, masked, h0, overwrite=True, output_verify='ignore')


def inference(nn_seg, ev, imgs_labels, occ_size, do_run_diff, ev_opath, filter=True):
    imageSize=[512,512]
    all_occ_size = []
    all_diff_img = []    
    all_dates = []
    all_plate_scl = []
    all_headers = []
    for t in range(len(ev['pro_files'])):
        cimga= ev[imgs_labels[0]][t]
        # sets occ radius
        if cimga.find('cor1') > 0:
            cur_occ_size = occ_size[0]
        elif cimga.find('cor2') > 0:
            cur_occ_size = occ_size[1]     
        elif cimga.find('c2') > 0:
            cur_occ_size = occ_size[2]  
        else:
            cur_occ_size = 0   
        all_occ_size.append(cur_occ_size)
        
        if do_run_diff:
            cprea = ev[imgs_labels[1]][t]
        else:
            cprea = ev[imgs_labels[2]][t]
        breakpoint()
        imga, ha = read_fits(cimga, header=True,imageSize=imageSize)
        img = imga-read_fits(cprea, imageSize=imageSize)
        all_diff_img.append(img)
        all_dates.append(datetime.datetime.strptime(ha['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f'))
        if ha['NAXIS1'] != imageSize[0]:
            plt_scl = ha['CDELT1'] * ha['NAXIS1']/imageSize[0] 
        else:
            plt_scl = ha['CDELT1']
        all_plate_scl.append(plt_scl)
        all_headers.append(ha)
    # inference of all images in the event
    #print(f'Running inference for {imgs_labels[0]} dates {all_dates}')
    breakpoint()
    orig_img, dates, mask, scr, labels, boxes, mask_porp =  nn_seg.infer_event2(all_diff_img, all_dates, filter=filter, plate_scl=all_plate_scl, occulter_size=all_occ_size,  plot_params=ev_opath+'/'+imgs_labels[0])
    
    # adds mask_prop to headers as keywords, [mask_id,score,cpa_ang, wide_ang, apex_dist]
    for cdate in all_dates:
        i = all_dates.index(cdate)
        #only if at least one mask was found
        try:
            j = dates.index(cdate)
            all_headers[i]['NN_SCORE'] = mask_porp[j][0][1]
            all_headers[i]['NN_C_ANG'] = np.degrees(mask_porp[j][0][2])
            all_headers[i]['NN_W_ANG'] = np.degrees(mask_porp[j][0][3])
            # apex rom px to solar radii
            apex_sr = mask_porp[j][0][4] * all_plate_scl[i] / all_headers[i]['RSUN']
            all_headers[i]['NN_APEX'] = apex_sr
        except:
            all_headers[i]['NN_SCORE'] = 'None'
            all_headers[i]['NN_C_ANG'] = 'None'
            all_headers[i]['NN_W_ANG'] = 'None'
            all_headers[i]['NN_APEX'] = 'None'
            
    return  orig_img, dates, mask, scr, labels, boxes, mask_porp, all_headers


def inference_base(nn_seg, ev, imgs_labels, occ_size, do_run_diff, ev_opath, filter=True,version=''):
    imageSize=[512,512]
    all_occ_size = []
    all_diff_img = []    
    all_dates = []
    all_plate_scl = []
    all_headers = []
    all_occ_size_ext = []
    all_occ_center = []
    all_orig_img = []
    all_mask = []
    all_score = []
    all_lbl = []
    all_box = []
    #breakpoint()
    #for t in range(len(ev['pro_files'])):
    for t in range(len(ev[imgs_labels[0]])):    
        cimga= ev[imgs_labels[0]][t]
        # sets occ radius
        if cimga.find('cor1') > 0:
            cur_occ_size = occ_size[0]
        elif cimga.find('cor2') > 0:
            if cimga.find('/a/') > 0:
                cur_occ_size = occ_size[1] 
            if cimga.find('/b/') > 0:
                cur_occ_size = occ_size[2]
            occulter_size_ext = 250    
            occ_center=[256,256]
        elif cimga.find('c2') > 0:
            cur_occ_size = occ_size[3]
            occulter_size_ext = 300
        else:
            cur_occ_size = 0
            occulter_size_ext = 0
        #breakpoint()
        all_occ_size.append(cur_occ_size)
        all_occ_size_ext.append(occulter_size_ext)
        
        if do_run_diff:
            cprea = ev[imgs_labels[1]][t]
        else:
            cprea = ev[imgs_labels[2]][t]
        #breakpoint()
        imga, ha = read_fits(cimga, header=True,imageSize=imageSize)
        occ_center=[ha["crpix1"],ha["crpix2"]]
        img = imga-read_fits(cprea, imageSize=imageSize)
        all_occ_center.append(occ_center)
        all_diff_img.append(img)
        all_dates.append(datetime.datetime.strptime(ha['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f'))
        if ha['NAXIS1'] != imageSize[0]:
            plt_scl = ha['CDELT1'] * ha['NAXIS1']/imageSize[0] 
        else:
            plt_scl = ha['CDELT1']
        all_plate_scl.append(plt_scl)
        all_headers.append(ha)
    # inference of all images in the event
    #print(f'Running inference for {imgs_labels[0]} dates {all_dates}')
    #breakpoint()
    #orig_img, dates, mask, scr, labels, boxes, mask_porp =  nn_seg.infer_event2(all_diff_img, all_dates, filter=filter, plate_scl=all_plate_scl, occulter_size=all_occ_size,  plot_params=ev_opath+'/'+imgs_labels[0])

    for i in range(len(all_diff_img)):
        #infer masks
        orig_img, masks, scores, labels, boxes = nn_seg.infer(all_diff_img[i], model_param=None, resize=False, 
                                                            occulter_size=all_occ_size[i],occulter_size_ext=all_occ_size_ext[i],
                                                            centerpix=all_occ_center[i])
        # plot the predicted mask
        #ofile = opath+"/"+os.path.basename(cimga)+'infer1.png'inference_base
        #plot_to_png(ofile, [orig_img], [masks], scores=[scores], labels=[labels], boxes=[boxes],version=version)
        #breakpoint()
        all_orig_img.append(orig_img)
        all_mask.append(masks)
        all_score.append(scores)
        all_lbl.append(labels)
        all_box.append(boxes)
    
    return  all_orig_img, all_dates, all_mask, all_score, all_lbl, all_box, all_headers, all_plate_scl


#main
#------------------------------------------------------------------Testing the CNN--------------------------------------------------------------------------
model = 'A6_DS32' #'A6_DS32' # 'A4_DS31' #'A6_DS32'
if model == "v5":
    model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v5"
    model_version="v5"
    trained_model = '49.torch'

if model == 'A4_DS31':
    model_path= "/gehme-gpu2/projects/2020_gcs_with_ml/output/neural_cme_seg_A4_DS32"
    model_version="A4"
    trained_model = '49.torch'

if model == 'A4_DS32':
    model_path= "/gehme-gpu2/projects/2020_gcs_with_ml/output/neural_cme_seg_A4_DS32"
    model_version="A4"
    trained_model = '49.torch'

if model == 'A6_DS32':
    model_path= "/gehme-gpu2/projects/2020_gcs_with_ml/output/neural_cme_seg_A6_DS32"
    model_version="A6"
    trained_model = '49.torch'

mask_threshold = 0.54 

#model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4"
#model_version="v4"
#trained_model = '9999.torch'
#trained_model = '6000.torch'

opath= model_path + "/infer_neural_cme_seg_exp_paper_filtered_diego"
file_ext=".png"
do_run_diff = True # set to use running diff instead of base diff (False)
#occ_size = [90,35,75] # Artifitial occulter radius in pixels. Use 0 to avoid. [Stereo a/b C1, Stereo a/b C2, Lasco C2]
occ_size = [90,50,60,90] #cor1, cor2a, cor2b, C2
filter = False 
scr_threshold = 0.15
make_gcs_masks = True #if set True, the sav files are used to set GCS masks.
#main
gpu=0 # GPU to use
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unless its not available
print(f'Using device:  {device}')

os.makedirs(opath, exist_ok=True)
event = get_paths_cme_exp_sources() # get all files event
#breakpoint()
file = open(opath + '/all_events.csv', 'w')
writer = csv.writer(file)
writer.writerow(event[0].keys())
for e in event:
    writer.writerow(e.values())
file.close()

#loads nn model
nn_seg = neural_cme_segmentation(device, pre_trained_model = model_path + "/"+ trained_model, version=model_version)
nn_seg.mask_threshold = mask_threshold
breakpoint()
for ev in event:
    print(f'Processing event {ev["date"]}')
    ev_opath = os.path.join(opath, ev['date'].split('/')[-1]) + '_filter_'+str(filter)   
    #cora
    #breakpoint()    
    imgs_labels = ['ima1', 'ima0','pre_ima']
    #orig_imga, datesa, maska, scra, labelsa, boxesa, mask_porpa, ha = inference(nn_seg, ev, imgs_labels, occ_size, do_run_diff, ev_opath, filter=filter)
    orig_imga, datesa, maska, scra, labelsa, boxesa, headersa, plate_scl_a = inference_base(nn_seg, ev, imgs_labels, occ_size, do_run_diff, ev_opath, filter=filter, version=model_version)
    datesa = [d.strftime('%Y-%m-%dT%H:%M:%S.%f') for d in datesa]      
    #breakpoint()
    #corb
    imgs_labels = ['imb1', 'imb0','pre_imb']
    orig_imgb, datesb, maskb, scrb, labelsb, boxesb, headersb, plate_scl_b = inference_base(nn_seg, ev, imgs_labels, occ_size, do_run_diff, ev_opath, filter=filter, version=model_version) 
    datesb = [d.strftime('%Y-%m-%dT%H:%M:%S.%f') for d in datesb]        
    #lasco
    imgs_labels = ['lasco1', 'lasco0','pre_lasco']
    orig_imgl, datesl, maskl, scrl, labelsl, boxesl, headersl, plate_scl_l = inference_base(nn_seg, ev, imgs_labels, occ_size, do_run_diff, ev_opath, filter=filter, version=model_version)
    datesl = [d.strftime('%Y-%m-%dT%H:%M:%S.%f') for d in datesl]    
    #breakpoint()

    if len(orig_imga) != len(orig_imgb) or len(orig_imga) != len(orig_imgl):
        print('Different number of images in the three instruments. We are repeting some images in the triplet plots.')
        max_size = max(len(orig_imga), len(orig_imgb), len(orig_imgl))
        for i in range(3):
            if len(orig_imga) < max_size:
                orig_imga.insert(0,orig_imga[0])
                datesa.insert(0,datesa[0])
                maska.insert(0,maska[0])
                scra.insert(0,scra[0])
                labelsa.insert(0,labelsa[0])
                boxesa.insert(0,boxesa[0])
                headersa.insert(0,headersa[0])
                plate_scl_a.insert(0,plate_scl_a[0])
            if len(orig_imgb) < max_size:
                orig_imgb.insert(0,orig_imgb[0])
                datesb.insert(0,datesb[0])
                maskb.insert(0,maskb[0])
                scrb.insert(0,scrb[0])
                labelsb.insert(0,labelsb[0])
                boxesb.insert(0,boxesb[0])
                headersb.insert(0,headersb[0])
                plate_scl_b.insert(0,plate_scl_b[0])
            if len(orig_imgl) < max_size:
                orig_imgl.insert(0,orig_imgl[0])
                datesl.insert(0,datesl[0])
                maskl.insert(0,maskl[0])
                scrl.insert(0,scrl[0])
                labelsl.insert(0,labelsl[0])
                boxesl.insert(0,boxesl[0])
                headersl.insert(0,headersl[0])
                plate_scl_l.insert(0,plate_scl_l[0])
    headers_list = [[headersa[j],headersb[j],headersl[j]] for j in range(len(ev['pro_files']))]
    os.makedirs(ev_opath, exist_ok=True)
    mask_list = []
    for t in range(len(ev['pro_files'])):
        headers = headers_list[t]
        #read sav file
        if make_gcs_masks:
            file_sav = readsav(ev['sav_files'][t])
            file_save = file_sav['sgui']
            lon, lat,rot, han, hgt, rat = file_save.lon, file_save.lat, file_save.rot, file_save.han, file_save.hgt, file_save.rat
            lon, lat,rot, han, hgt, rat = np.degrees(lon), np.degrees(lat), np.degrees(rot), np.degrees(han), hgt, rat
            imsize=np.array([512, 512], dtype='int32')
            #create gcs cloud
            satpos, plotranges = pyGCS.processHeaders(headers)
            usr_center_off = None
            inner_hole_mask=False
            occ_center   = [[0.,0.], [0.3,-0.3] , [0.,0.]]
            size_occ     = [2.9, 3.9, 1.9]
            size_occ_ext = [16+1, 16+1, 6+1]
            n_sat=3
            mask_list_cor2a = []
            mask_list_cor2b = []
            mask_list_c2    = []
            for sat in range(n_sat):
                if sat==0:
                    occ_size_1024 = 100
                elif sat==1:
                    occ_size_1024 = 120
                elif sat==2:
                    occ_size_1024 = 150
                x = np.linspace(plotranges[sat][0], plotranges[sat][1], num=imsize[0])
                y = np.linspace(plotranges[sat][2], plotranges[sat][3], num=imsize[1])
                xx, yy = np.meshgrid(x, y)
                x_cS, y_cS = center_rSun_pixel(headers, plotranges, sat) 
                # ad hoc correction for the occulter center
                x_cS, y_cS = x_cS+occ_center[sat][0], y_cS+occ_center[sat][1]       
                r = np.sqrt((xx - x_cS)**2 + (yy - y_cS)**2)
                phi = np.arctan2(yy - y_cS, xx - x_cS)
                #breakpoint()                
                btot_mask = rtraytracewcs(headers[sat], float(lon), float(lat), float(rot), float(hgt), float(rat),
                                        float(han), imsize=imsize, occrad=size_occ[sat], in_sig=1., out_sig=0.001, nel=1e5, usr_center_off=usr_center_off)     
                cme_npix= len(btot_mask[btot_mask>0].flatten())
                if cme_npix<=0:
                    print("Empty mask created corresponding to")
                    print(ev['sav_files'][t])
                    breakpoint()
                    break          
                mask = get_cme_mask(btot_mask,inner_cme=inner_hole_mask,occ_size=occ_size_1024)          
                mask_npix= len(mask[mask>0].flatten())
                if mask_npix/cme_npix<0.5:                        
                    print("Empty projection mask created?")
                    breakpoint()
                    break
                #adds occulter to the masks and checks for null masks
                mask[r <= size_occ[sat]] = 0  
                mask[r >= size_occ_ext[sat]] = 0
                #save mask
                if sat == 0:
                    mask_list_cor2a.append(mask)
                if sat == 1:
                    mask_list_cor2b.append(mask)
                    if len(mask) == 0:
                        breakpoint()
                if sat == 2:
                    mask_list_c2.append(mask)
                    if len(mask) == 0:
                        breakpoint()
        mask_list.append([mask_list_cor2a,mask_list_cor2b,mask_list_c2])
        #breakpoint()
        ofile = os.path.join(ev_opath,os.path.basename(ev['pro_files'][t])+'.png')
        #breakpoint()
        if filter:
            plot_to_png(ofile, [orig_imga[t],orig_imgb[t], orig_imgl[t]], [[maska[t]],[maskb[t]],[maskl[t]]], 
                        title=[datesa[t], datesb[t], datesl[t]],labels=[[labelsa[t]],[labelsb[t]], [labelsl[t]]], 
                        boxes=[[boxesa[t]], [boxesb[t]], [boxesl[t]]], scores=[[scra[t]], [scrb[t]], [scrl[t]]],
                        masks_gcs = mask_list[t],
                        version=model_version,scr_threshold=scr_threshold)#, save_masks=[ha[t],hb[t],hl[t]])
            #breakpoint()
        else:
            plot_to_png(ofile, [orig_imga[t],orig_imgb[t], orig_imgl[t]], [maska[t],maskb[t],maskl[t]], 
                        title=[datesa[t], datesb[t], datesl[t]],labels=[labelsa[t],labelsb[t], labelsl[t]], 
                        boxes=[boxesa[t], boxesb[t], boxesl[t]], scores=[scra[t], scrb[t], scrl[t]],
                        masks_gcs = mask_list[t],
                        version=model_version,scr_threshold=scr_threshold) 
            #breakpoint()        
