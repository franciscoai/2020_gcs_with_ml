import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib as mpl
from astropy.io import fits
mpl.use('Agg')

def normalize(image):
    '''
    Normalizes the values of the input image to have a given range (as fractions of the sd around the mean)
    maped to [0,1]. It clips output values outside [0,1]
    '''
    sd_range=1.
    m = np.mean(image)
    sd = np.std(image)
    image = (image - m + sd_range * sd) / (2 * sd_range * sd)
    image[image >1]=1
    image[image <0]=0
    return image


#------------------------------------------------------------------Testing the CNN-----------------------------------------------------------------
dataDir = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_training_v3'
model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v3"
opath= model_path+"/test_output"
file_ext=".png"
trained_model = '3999.torch'
testDir=  dataDir 
imageSize=[512,512]
test_ncases = 100


#main
os.makedirs(opath, exist_ok=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unles its not available
print(f'Using device:  {device}')
test_dirs = os.listdir(testDir)
test_dirs= [d for d in test_dirs if not d.endswith(".csv")][0:test_ncases]
imgs = [os.path.join(testDir, dir) for dir in test_dirs]
all_scr =[]
ind = 0
for num in test_dirs:
    idx=random.randint(0,len(test_dirs)-1) #select a random image from the testing batch
    #loads model
    model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) 
    in_features = model.roi_heads.box_predictor.cls_score.in_features 
    model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=3)
    model.load_state_dict(torch.load(model_path + "/"+ trained_model)) #loads the last iteration of training 
    model.to(device)# move model to the right device
    model.eval()#set the model to evaluation state
    #inference
    file=os.listdir(imgs[idx])
    file=[f for f in file if f.endswith(file_ext)]
    print(f'Inference No. {ind}/{len(test_dirs)} of imge: {idx}')
    images = cv2.imread(os.path.join(imgs[idx], file[0]))
    images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)
    im = images.copy()
    images = normalize(images) #cv2.normalize(images, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # normalize to 0,1
    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
    images=images.swapaxes(1, 3).swapaxes(2, 3)
    images = list(image.to(device) for image in images)
    with torch.no_grad(): #runs the image through the net and gets a prediction for the object in the image.
        pred = model(images)

    # To plot the true mask
    maskDir=os.path.join(imgs[idx], "mask") #path to the mask iamge corresponding to the random image
    masks=[]
    for mskName in os.listdir(maskDir):
        vesMask = cv2.imread(maskDir+'/'+mskName,0) #reads the mask image in greyscale
        vesMask = (vesMask > 0).astype(np.uint8) #The mask image is stored in 0–255 format and is converted to 0–1 format
        vesMask=cv2.resize(vesMask,imageSize,cv2.INTER_NEAREST) #resizes the mask image to the same size of the random image
        #cv2.imshow("mask",vesMask.astype('float64'))#for plotting if the dtype is uint8 it tries to plot as 0-255 format being 0 black and 255 withe, so 1 its close to 0 so it will be black. But if we use float64 it will take 0 as 0 and 1 as 255. 
        

    #The predicted object ‘masks’ are saved as a matrix in the same size as the image with each pixel 
    #having a value that corresponds to how likely it is part of the object. And only displays the ones with scores larger than 0.8
    #ssume that only pixels which values larger than 0.5 are likely to be part of the objects.
    #We display this by marking these pixels with a different random color for each object

    im= im.astype(np.uint8)
    im2 = im.copy()
    im3 = im.copy()
    nmasks = len(pred[0]['masks'])
    colors = [[255,0,0],[0,255,0],[0,0,255],[0,0,0],[0,255,255],[255,255,0]]
    for i in range(nmasks):
        msk=pred[0]['masks'][i,0].detach().cpu().numpy()
        scr=pred[0]['scores'][i].detach().cpu().numpy()
        all_scr.append([scr,idx])
        if scr>0.8 :
            im2[:, :, 0][msk > 0.5] = colors[i][0]
            im2[:, :, 1][msk > 0.5] = colors[i][1]
            im2[:, :, 2][msk > 0.5] = colors[i][2]
        else:
            scr="below_0.8" 

    im3[vesMask > 0] = 255
    pic = np.hstack([im,im2,im3])
    cv2.imwrite(opath+"/img_"+str(idx)+'_scr_'+str(scr)+'.png', pic)
    ind+=1

score = np.array([float(i[0]) for i in all_scr])
fig= plt.figure(figsize=(10, 5)) 
ax = fig.add_subplot() 
ax.hist(score,bins=30)
ax.set_title(f'Mean scr: {np.mean(score)}; % of scr>0.8: {len(score[score>0.8])/len(score)}')
ax.set_yscale('log')
fig.savefig(model_path+"/all_scores.png")

