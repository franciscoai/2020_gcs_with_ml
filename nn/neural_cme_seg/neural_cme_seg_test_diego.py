
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
import matplotlib.gridspec as gridspec
import pandas as pd
import csv 

def create_csv_file(input_csv_file=None, path=None):
    path = '/gehme-gpu2/projects/2020_gcs_with_ml/output/neural_cme_seg_A6_DS32/'
    output_csv_1k = "training_cases_1000.csv"
    output_csv_10k = "training_cases_10000.csv"
    input_csv_file = "training_cases.csv"
    lista = []
    with open(path+input_csv_file, 'r') as f:
        for line in f:
            lista.append(line.strip())
    sample_1k_df  = lista[:1000]
    sample_10k_df = lista[:10000]
    with open(os.path.join(path, output_csv_1k), 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        for line in sample_1k_df:
            writer.writerow([line])
    with open(os.path.join(path, output_csv_10k), 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        for line in sample_10k_df:
            writer.writerow([line])


def calculate_gradients(image):
    """
    Calculates the gradients (Sobel operator) of intensity in a 2D intensity image.
    Args:
    image: A 2D NumPy array representing the intensity image.
    Returns:
    A tuple containing two NumPy arrays:
        - gx: Gradients in the x-direction (horizontal).
        - gy: Gradients in the y-direction (vertical).
    """
    kernel_size = (5, 5)
    sigma = 1
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)

    # Sobel filter kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Pad the image for edge handling (optional)
    # padded_image = np.pad(image, (1, 1), mode='edge')

    # Calculate gradients using convolution
    gx = cv2.filter2D(blurred_image, -1, sobel_x, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(blurred_image, -1, sobel_y, borderType=cv2.BORDER_REPLICATE)
    
    # You can calculate the magnitude and direction of the gradient:
    gxx = np.where(np.isnan(gx), 0, gx)
    gyy = np.where(np.isnan(gy), 0, gy)
    magnitude = np.sqrt(gxx**2 + gyy**2)
    direction = np.arctan2(gyy, gxx) * 180 / np.pi

    return gx, gy,magnitude, direction

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

def image_to_polar(image):
    """
    Converts a square image to polar coordinates.
    """
    # Get image dimensions
    height, width = image.shape
    # Calculate center of the image
    center_x = width // 2 #definir con crpix1 y2
    center_y = height // 2
    # Create meshgrid for x and y coordinates
    x, y = np.meshgrid(np.arange(width) - center_x, np.arange(height) - center_y)
    # Calculate radius and angle
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    # Create polar image
    polar_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            r_index = int(r[i, j])
            theta_index = int(theta[i, j] * (width / (2 * np.pi)))
            polar_image[r_index, theta_index] = image[i, j]
    #polar_image[r>280] =1
    #polar_image[r<60 ] =1 #definir con los valores reales del oculter.
    #polar_image = np.fliplr(polar_image)
    polar_image = polar_image[50:280, :]
    polar_image = np.flip(polar_image, (0, 1))
    return polar_image


def plot_to_png2(ofile, orig_img, masks, true_mask, scr_threshold=0.3, mask_threshold=0.3 , title=None,string=None, 
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
    elif version=='A4':
        obj_labels = ['Back', 'Occ','CME']
    elif version=='A6':
        obj_labels = ['Back', 'Occ','CME']
    else:
        print(f'ERROR. Version {version} not supported')
        sys.exit()
    #        
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(orig_img[0]), np.nan)
    
    fig = plt.figure(figsize=(30, 20))
    gs0 = gridspec.GridSpec(2, 3, figure=fig)
    gs00 = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs0[0])
    gs01 = gs0[1].subgridspec(4, 4)
    gs02 = gs0[2].subgridspec(4, 4)
    ax1 = fig.add_subplot(gs00[:, :])
    ax2 = fig.add_subplot(gs01[:, :])
    ax3 = fig.add_subplot(gs02[:-2, :])
    ax4 = fig.add_subplot(gs02[-2:, :])
    gs03 = gs0[3].subgridspec(4, 4)
    gs04 = gs0[4].subgridspec(4, 4)
    gs05 = gs0[5].subgridspec(4, 4)
    ax5 = fig.add_subplot(gs03[:, :])
    ax6 = fig.add_subplot(gs04[:, :])
    ax7 = fig.add_subplot(gs05[:, :])
    #fig, axs = plt.subplots(1, len(orig_img)*3, figsize=(30, 10))
    #axs = axs.ravel()
    for i in range(len(orig_img)):
        ax1.imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')
        ax1.axis('off')
        if string is not None:
            ax1.text(0, 0, string,horizontalalignment='left',verticalalignment='bottom',
            fontsize=15,color='white',transform=ax1.transAxes)

        ax2.imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')        
        ax2.axis('off')        
        #add true mask
        masked = nans.copy()
        masked[true_mask[i] > 0] = 3 
        ax2.imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=4)
        if boxes is not None:
            nb = 0
            for b in boxes[i]:
                if scores is not None:
                    scr = scores[i][nb]
                else:
                    scr = 0   
                if scr > scr_threshold:             
                    best_iou, best_dice, best_prec, best_rec,max_iou, max_dice, max_prec, max_rec = best_mask_treshold(masks[i][nb], orig_img, true_mask[i])                    
                    masked = nans.copy()
                    #masked[:, :][masks[i][nb] > mask_threshold] = nb
                    masked[:, :][masks[i][nb] > best_iou] = nb
                    ax2.imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1) # add mask
                    box =  mpl.patches.Rectangle(b[0:2],b[2]-b[0],b[3]-b[1], linewidth=2, edgecolor=color[nb], facecolor='none') # add box
                    ax2.add_patch(box)
                    if labels is not None:
                        ax2.annotate(obj_labels[labels[i][nb]]+':'+'{:.2f}'.format(scr),xy=b[0:2], fontsize=15, color=color[nb])
                    #calculate metrics
                    #add iou_score as an annotation in the image
                    ax2.annotate('IoU : '+'{:.2f}'.format(best_iou) ,xy=[10,20], fontsize=30, color=color[nb])
                    ax2.annotate(''+'{:.2f}'.format(max_iou)        ,xy=[150,20], fontsize=30, color=color[nb])
                    ax2.annotate('Dice: '+'{:.2f}'.format(best_dice),xy=[10,50], fontsize=30, color=color[nb])
                    ax2.annotate(''+'{:.2f}'.format(max_dice)       ,xy=[150,50], fontsize=30, color=color[nb])
                    ax2.annotate('Prec: '+'{:.2f}'.format(best_prec),xy=[10,80], fontsize=30, color=color[nb])
                    ax2.annotate(''+'{:.2f}'.format(max_prec)       ,xy=[150,80], fontsize=30, color=color[nb])
                    ax2.annotate('Rec : '+'{:.2f}'.format(best_rec) ,xy=[10,110], fontsize=30, color=color[nb])
                    ax2.annotate(''+'{:.2f}'.format(max_rec)        ,xy=[150,110], fontsize=30, color=color[nb])
                    #convert x,y image coordinates into r,theta coordinates
                    polar_image = image_to_polar(orig_img[i])
                    #plot image in r, theta
                    #ax2.imshow(masked_polar, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1)
                    ax3.imshow(polar_image, vmin=0, vmax=1, cmap='gray')        
                    ax3.axis('off')
                    #convert masked with the treshold into polar coordinates. then plot a countour of the masked image in ax3
                    masked_polar = nans.copy()
                    polar_masked = image_to_polar(masks[i][nb])
                    ax3.contour(polar_masked, levels=[best_iou], colors='r')
                    #in ax4 plot an histogram of  the image between the a box of the mask
                    zero = np.full(np.shape(orig_img[0]), 0)
                    zero[round(b[0]):round(b[2]), round(b[1]):round(b[3])] = 1
                    data_for_histo = zero * orig_img[i]
                    #breakpoint()
                    #from data_for_histo remove the zeros
                    data_for_histo = data_for_histo[data_for_histo != 0]
                    ax4.hist(data_for_histo.flatten(), bins=50)
                    ax4.set_title('Histogram of the image inside the bounding box, non zero values')
                    gx, gy, magn, direc = calculate_gradients(orig_img[i])
                    ax5.imshow(gx, cmap='gray')
                    ax5.contour(masks[i][nb], levels=[best_iou], colors='r', alpha=0.4)
                    ax5.axis('off')
                    ax5.set_title('Gradient in x')
                    #ax6.imshow(gy, cmap='gray')
                    ax6.imshow(magn, cmap='gray')
                    #ax6.contour(masks[i][nb], levels=[best_iou], colors='r', alpha=0.7)
                    ax6.axis('off')
                    ax6.set_title('Magnitude of the gradient')
                    ax6.set_title('Gradient in y')
                    ax7.imshow(magn, cmap='gray')
                    ax7.contour(masks[i][nb], levels=[best_iou], colors='r', alpha=0.5)
                    ax7.axis('off')
                    ax7.set_title('Magnitude of the gradient')
                nb+=1
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()
    return best_iou, max_iou


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
    #fig, axs = plt.subplots(1, len(orig_img)*2, figsize=(20, 10))
    fig, axs = plt.subplots(1, len(orig_img)*3, figsize=(30, 10))
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
                    #calculate metrics
                    
                    best_iou, best_dice, best_prec, best_rec,max_iou, max_dice, max_prec, max_rec = best_mask_treshold(masks[i][nb], orig_img, true_mask[i])
                    
                    #add iou_score as an annotation in the image
                    axs[i+1].annotate('IoU : '+'{:.2f}'.format(best_iou) ,xy=[10,20], fontsize=30, color=color[nb])
                    axs[i+1].annotate(''+'{:.2f}'.format(max_iou)        ,xy=[150,20], fontsize=30, color=color[nb])
                    axs[i+1].annotate('Dice: '+'{:.2f}'.format(best_dice),xy=[10,50], fontsize=30, color=color[nb])
                    axs[i+1].annotate(''+'{:.2f}'.format(max_dice)       ,xy=[150,50], fontsize=30, color=color[nb])
                    axs[i+1].annotate('Prec: '+'{:.2f}'.format(best_prec),xy=[10,80], fontsize=30, color=color[nb])
                    axs[i+1].annotate(''+'{:.2f}'.format(max_prec)       ,xy=[150,80], fontsize=30, color=color[nb])
                    axs[i+1].annotate('Rec : '+'{:.2f}'.format(best_rec) ,xy=[10,110], fontsize=30, color=color[nb])
                    axs[i+1].annotate(''+'{:.2f}'.format(max_rec)        ,xy=[150,110], fontsize=30, color=color[nb])
                    #convert x,y image coordinates into r,theta coordinates
                    polar_image = image_to_polar(orig_img[i])
                    #plot image in r, theta
                    axs[i+2].imshow(polar_image, vmin=0, vmax=1, cmap='gray')        
                    axs[i+2].axis('off')
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
#select the model
model = 'A6_DS32' #'A6_DS32' # 'A4_DS31' #'A6_DS32'

if model == 'A4_DS31':
    testDir =  '/gehme-gpu2/projects/2020_gcs_with_ml/data/cme_seg_20250320/'
    model_path= "/gehme-gpu2/projects/2020_gcs_with_ml/output/neural_cme_seg_A4_DS31"
    model_version="A4"
    trained_model = ['49.torch']

if model == 'A4_DS32':
    testDir =  '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_20250320/'
    model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_A4_DS32"
    model_version="A4"
    trained_model = ['49.torch']

if model == 'A6_DS32':
    testDir =  '/gehme-gpu2/projects/2020_gcs_with_ml/data/cme_seg_20250320/'
    model_path= "/gehme-gpu2/projects/2020_gcs_with_ml/output/neural_cme_seg_A6_DS32"
    model_version="A6"
    trained_model = [f"{i}.torch" for i in range(50)]
    #trained_model = ['48.torch','49.torch']
    original_DF = "/gehme-gpu2/projects/2020_gcs_with_ml/data/cme_seg_20250320/20250320_Set_Parameters_unpacked_filtered_DS32.csv"

if model == 'v5':
    testDir =  '/gehme/projects/2020_gcs_with_ml/data/cme_seg_20240912/'
    model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v5"
    model_version="v5"
    trained_model = ['49.torch']


#Select the desired mode, one at a time
calculate_best_mask_treshold  = True #estimate the best mask treshold based on the IoU score
normal_test_one_mask_treshold = False #run the test with a single mask treshold and plot the result images
statistics_using_best_mask_treshold = False #run massive test statistics using the best mask treshold and saving DF.

if calculate_best_mask_treshold:
    #Si longitud de mask_thresholds es mayor a 1, se hace una estadistica de IoU y score vs mask_thresholds, SIN ploteo de imagenes.
    create_validation_cases = False  # if True, it will create a file with the test cases. Select False to use a specific csv file with fixed cases.
    use_random_cases        = False # if True, it will use random cases from the testDir. Select False to use a specific csv file with fixed cases.
    use_fixed_cases         = True # if True, it will use a specific csv file with fixed cases. Select False to use random cases.
    test_ncases = 1000 #amount of cases to take into account
    #mask thresholds values to test
    mask_thresholds = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99] 
    expand_mask_thresholds = True # if True, mask_thresholds will be expanded to 0.5-0.6 range
    if expand_mask_thresholds:
        aux1 = np.round(np.linspace(0.5, 0.6, 11),2).tolist()
        aux2 = np.round(np.linspace(0.4, 0.5, 11),2).tolist()
        mask_thresholds = np.unique(mask_thresholds + aux1 + aux2)
        mask_thresholds.sort()
        mask_thresholds = mask_thresholds.tolist()

if normal_test_one_mask_treshold:
    #Si longitud de mask_thresholds es 1, se hace un solo plot de los resultados, se correrá el plot_to_png2 y se hará un barrido,
    # en el mask_threshold para el ploteo 
    mask_thresholds = [0.7] 
    create_validation_cases = True  # if True, it will create a file with the test cases. Select False to use a specific csv file with fixed cases.
    use_random_cases        = False # if True, it will use random cases from the testDir. Select False to use a specific csv file with fixed cases.
    use_fixed_cases         = False # if True, it will use a specific csv file with fixed cases. Select False to use random cases.

if statistics_using_best_mask_treshold:

    mask_thresholds = [0.7] 
    plot_images = False # if True, it will plot the images with the selected mask treshold
    create_validation_cases = False  # if True, it will create a file with the test cases. Select False to use a specific csv file with fixed cases.
    use_random_cases        = False # if True, it will use random cases from the testDir. Select False to use a specific csv file with fixed cases.
    use_fixed_cases         = True # if True, it will use a specific csv file with fixed cases. Select False to use random cases.
    DF_to_use = original_DF  #DF created when DataSet was created. It contains statistics of the synthetic images.
    test_ncases = 1000 #llevar a 10k
    #df_new_output = model_path+"/"+model+trained_model.replace('.', '')+"_training_cases_1000_IOU.csv"

opath= model_path+"/test_output_diego"
file_ext="btot.png"
imageSize=[512,512]
if create_validation_cases or use_random_cases:
    test_cases_file = model_path+"/validation_cases.csv"

if not create_validation_cases and not use_random_cases:
    test_cases_file = model_path+"/test_cases.csv"

if use_fixed_cases:
    #select csv file that should be the outful of running this code with create_validation_cases=True.
    test_cases_file = model_path+"/training_cases_1000.csv"

gpu=0# GPU to use
masks2use=[2] # index of the masks to read (should be the CME mask)

#main
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unless its not available
print(f'Using device:  {device}')
#load image paths from test_cases_file
imgs = []
try:
    with open(test_cases_file, 'r') as f:
        for line in f:
            imgs.append(line.strip())
except FileNotFoundError:
    if model_version == 'v4':
        breakpoint()
        with open('/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v5/validation_cases.csv', 'r') as f:
            for line in f:
                imgs.append(line.strip())
        print('File not found, using default validation cases (V5)')
except Exception as e:
    # This block will catch any other exceptions that might occur during file opening (optional, but good practice)
    print(f"An error occurred while trying to open '{test_cases_file}': {e}")
    breakpoint()

#nn_seg = neural_cme_segmentation(device, pre_trained_model = model_path + "/"+ trained_model, version=model_version)

if create_validation_cases or use_random_cases:
    rnd_idx = random.sample(range(len(imgs)), test_ncases)
if create_validation_cases:
    # create a csv file with the test cases
    with open(test_cases_file, 'w') as f:
        for idx in rnd_idx:
            f.write(f"{imgs[idx]}\n")
    print(f"Test cases saved to {test_cases_file}")

if use_fixed_cases:
    rnd_idx = imgs

#create
if statistics_using_best_mask_treshold:
    data_path = '/gehme-gpu2/projects/2020_gcs_with_ml/data/cme_seg_20250320/'
    df_original = pd.read_csv(DF_to_use)
    df_original['folder_name'] = df_original['folder_name'].astype(str)

for torch_models in trained_model:
    nn_seg = neural_cme_segmentation(device, pre_trained_model = model_path + "/"+ torch_models, version=model_version)
    all_scr = []
    all_iou = []
    all_best_iou = []
    all_max_iou = []
    list_of_selected_dfs = []
    list_of_rows_selected_dfs = []
    for mask_threshold in mask_thresholds:
        this_case_scr =[]
        this_case_iou = []    
        for ind in range(len(rnd_idx)):
            idx = rnd_idx[ind]

            #crear el DF de los casos de prueba
            #buscar el evento leido dede el csv fixed en el DF original.
            #para cada evento guardar todos los datos del DF original. 
            if statistics_using_best_mask_treshold:
                folder_check = idx.split('/')[-1]
                mask = df_original['folder_name'] == folder_check
                row_indices = df_original.index[mask].tolist()
                list_of_rows_selected_dfs.append(row_indices)
                list_of_selected_dfs.append(df_original.iloc[row_indices])
                if len(row_indices) != 1:
                    print("this should not happen, please check.")
                    breakpoint()
                

            #reads image
            file=os.listdir(idx)#(imgs[idx])
            file=[f for f in file if f.endswith(file_ext)]
            print(f'Inference No. {ind}/{len(rnd_idx)} using image: {idx} and mask threshold case No. {mask_thresholds.index(mask_threshold)+1}/{len(mask_thresholds)}')
            #print(f'Event: {imgs[idx]}')
            print(f'Event: {idx}')
            #images = cv2.imread(os.path.join(imgs[idx], file[0]))
            images = cv2.imread(os.path.join(idx, file[0]))
            images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)

            # reads mask
            #maskDir=os.path.join(imgs[idx], "mask") #path to the mask iamge corresponding to the random image
            maskDir=os.path.join(idx, "mask")
            masks=[]
            vesMask=os.listdir(maskDir) #all masks in the mask directory
            vesMask.sort(key=lambda x: int(x.split("_")[-1].split(".")[0])) #sort masks files by the ending numbe
            vesMask = [vesMask[i] for i in masks2use][0]
            vesMask = cv2.imread(maskDir+'/'+vesMask, 0) #reads the mask image in greyscale 
            vesMask = (vesMask > 0).astype(np.uint8) #The mask image is stored in 0–255 format and is converted to 0–1 format
            if np.mean(vesMask) == 0 and np.max(vesMask) == 0:
                print('Error: Full zero mask in path', maskDir, 'skipping')
                breakpoint()
            # makes inference and returns only the mask with smallest loss
            try:
                img, masks, scores, labels, boxes, iou  = nn_seg.test_mask(images, vesMask, mask_threshold=mask_threshold)
            except Exception as e:
                continue
            
            if masks is None:
                this_case_iou.append(None)

            if masks is not None:
                this_case_iou.append(iou) 
                this_case_scr.append(scores)
                # plot the predicted mask
                if len(mask_thresholds)==1 and plot_images:
                    os.makedirs(opath, exist_ok=True)
                    ofile = opath+"/img_"+str(idx)+'.png'
                    best_iou_for_plot, max_iou_for_plot = plot_to_png2(ofile, [img], [[masks]], [vesMask], scores=[[scores]], labels=[[labels]], boxes=[[boxes]], 
                                mask_threshold=mask_threshold, scr_threshold=0.1, version=model_version,string=imgs[idx])
                    all_best_iou.append(best_iou_for_plot)
                    all_max_iou.append(max_iou_for_plot)
        all_scr.append(this_case_scr)
        all_iou.append(this_case_iou)

        # plot stats for a single mask threshold
        if len(mask_thresholds)==1:
            this_case_scr = np.array(this_case_scr)
            fig= plt.figure(figsize=(10, 5)) 
            ax = fig.add_subplot() 
            ax.hist(this_case_scr,bins=30)
            ax.set_title(f'Mean scr: {np.mean(this_case_scr)} ; For mask threshold: {mask_threshold}  and test_ncases: {test_ncases} \n\
                        % of scr>0.8: {len(this_case_scr[this_case_scr>0.8])/len(this_case_scr)}')
            ax.set_yscale('log')
            #breakpoint()
            fig.savefig(opath+"/"+model+"_"+torch_models.replace('.', '')+"_test_scores.png")

            this_case_iou = np.array(this_case_iou)
            fig= plt.figure(figsize=(10, 5))
            ax = fig.add_subplot()
            ax.hist(this_case_iou,bins=30)
            ax.set_title(f'Mean loss: {np.mean(this_case_iou)} ; For mask threshold: {mask_threshold} and test_ncases: {test_ncases}')
            ax.set_yscale('log')
            fig.savefig(opath+"/"+model+"_"+torch_models.replace('.', '')+"_test_loss.png")

            all_best_iou = np.array(all_best_iou)
            fig= plt.figure(figsize=(10, 5))
            ax = fig.add_subplot()
            ax.hist(all_best_iou,bins=30,density=True)
            ax.set_title(f'Mean best iou vs mask threshold')
            ax.set_xlabel('mask threshold')
            ax.set_ylabel('Best iou')
            ax.grid()
            fig.savefig(opath+"/"+model+"_"+torch_models.replace('.', '')+"_test_mean_best_iou_vs_mask_threshold.png")

            if plot_images:
                all_max_iou = np.array(all_max_iou)
                fig= plt.figure(figsize=(10, 5))
                ax = fig.add_subplot()
                ax.hist(all_max_iou,bins=30,density=True)
                ax.set_title(f'Mean max iou vs mask threshold')
                ax.set_xlabel('mask threshold')
                ax.set_ylabel('Max iou')
                ax.grid()
                fig.savefig(opath+"/"+model+"_"+torch_models.replace('.', '')+"_test_mean_max_iou_vs_mask_threshold.png")

    if statistics_using_best_mask_treshold:
        df_new = pd.concat(list_of_selected_dfs, ignore_index=True)
        df_new = df_new.drop_duplicates().reset_index(drop=True)
        #breakpoint()
        df_new['IoU'] = all_iou[0]
        #save df_new to csv
        df_new_output = model_path+"/"+model+"_"+torch_models.replace('.', '')+"_training_cases_1000_IOU.csv"
        df_new.to_csv(df_new_output, index=False)
                                                    
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

        new_list= [np.array(todo_iou) for todo_iou in all_iou]
        fig= plt.figure(figsize=(10, 5))
        ax = fig.add_subplot()
        boxplot_results = ax.boxplot(new_list)
        median_values = []
        for median_line in boxplot_results['medians']: median_values.append(median_line.get_ydata()[0])
        ax.axhline(np.max(median_values))
        #ax.axvline(x=mask_thresholds[np.argmax(median_values)], color='red', linestyle='--', linewidth=1.5)
        ax.text(x=0.6, y=0.7, s=f'IoU: {np.max(median_values):.2f}',color='red', fontsize=15) 
        ax.text(x=0.6, y=0.65, s=f'Mask Tresh: {mask_thresholds[np.argmax(median_values)]:.2f}',color='red', fontsize=15)
        ax.set_title(f'IoU vs mask threshold')
        ax.set_xlabel('mask threshold')
        ax.set_ylabel('IoU')
        ax.grid()
        ax.set_xticks([y + 1 for y in range(len(new_list))],labels=[str(thresh_num) for thresh_num in mask_thresholds])
        fig.savefig(opath+"/"+model+"_"+torch_models.replace('.', '')+"max_iou_vs_mask_threshold.png")
    #breakpoint()

print('Results saved in:', opath)
print('Done :-D')


