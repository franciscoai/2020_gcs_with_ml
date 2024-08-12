#this code is to read all png training images and the corresponding masks images to check if they are zeros. If they are, remove the folder.
#This may happen if the masked CME is small and is covered by the oculter. It may happen in a faulty synthetic Btot image.
import random
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import scipy


training_path = '/gehme/projects/2020_gcs_with_ml/data/cme_seg_20240702/'
training_folders = os.listdir(training_path)

for folder in training_folders:
    print('checking folder: ', folder)
    img_path = os.path.join(training_path, folder)
    mask_path = os.path.join(training_path, folder+'/mask/')
    img_files = os.listdir(img_path)
    images_synt = [f for f in img_files if f.endswith(".png")]
    mask_files = os.listdir(mask_path)
    masks_synt = [f for f in mask_files if f.endswith(".png")]
    #breakpoint()
    for img_file in images_synt:
        img = cv2.imread(os.path.join(img_path, img_file), cv2.IMREAD_GRAYSCALE)
        if np.max(img) == 0 and np.mean(img) ==0:
            print('removing folder: ', folder)
            breakpoint()
            os.system('rm -r ' + os.path.join(training_path, folder))
            break
        if np.max(img) == 0 and np.mean(img) ==0:
            print('removing folder: ', folder)
            breakpoint()
            os.system('rm -r ' + os.path.join(training_path, folder))
            break
    for mask_file in masks_synt:
        mask = cv2.imread(os.path.join(mask_path, mask_file), 0)
        if np.max(mask) == 0 and np.mean(mask) ==0:
            print('removing folder: ', folder)
            breakpoint()
            os.system('rm -r ' + os.path.join(training_path, folder))
            break
        if np.max(mask) == 0 and np.mean(mask) ==0:
            print('removing folder: ', folder)
            breakpoint()
            os.system('rm -r ' + os.path.join(training_path, folder))
            break
    print('finished checking folder: ', folder)
breakpoint()
