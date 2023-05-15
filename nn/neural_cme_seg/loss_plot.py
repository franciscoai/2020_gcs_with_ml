import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import os
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
mpl.use('Agg')

'''
Plots the content of the loss pickle files
'''
opath = "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_new"
training_loss_file = "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_new/all_loss"
testing_loss_file = "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_new/test_loss"

training_loss = None
testing_loss = None
try:
    with open(training_loss_file, 'rb') as file:
        training_loss = pickle.load(file)
    with open(testing_loss_file, 'rb') as file:
        testing_loss = pickle.load(file)
except:
    print('Warning, could not find one of the loss files')

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
if training_loss is not None:
    axs[0].plot(training_loss)
    axs[0].set_title('Training loss')
    axs[0].set_yscale("log")
if testing_loss is not None:
    axs[1].hist([i[0] for i in testing_loss], bins=50)
    axs[1].set_title('Testing loss')
fig.savefig(opath + '/loss_from_file.png')
print(f'Plot saved to {opath}/loss_from_file.png')
