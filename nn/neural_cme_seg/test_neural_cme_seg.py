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
import matplotlib.image as mpimg


dataDir = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_dataset'
trainDir=  dataDir 
testDir=  dataDir 
file_ext=".png"
opath= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg"
pretrained_nn="200.torch"


batchSize=1
 #number of images used in each iteration
imageSize=[512,512] 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unles its not available

#------------------------------------------------------------------Testing the CNN-----------------------------------------------------------------
for num in os.listdir(testDir)[0]:
    imgs=[]
    dirs=os.listdir(trainDir)
    dirs= [d for d in dirs if not d.endswith(".csv")]
    for pth in dirs:
        imgs.append(testDir+"/"+pth)
    
    idx=random.randint(0,len(imgs)-1) #select a random image from the testing batch
    



    model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) 
    in_features = model.roi_heads.box_predictor.cls_score.in_features 
    model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=2)
    model.load_state_dict(torch.load(opath + "/"+ pretrained_nn)) #loads the last iteration of training 
    model.to(device)# move model to the right device
    model.eval()#set the model to evaluation state
    file=os.listdir(imgs[idx])
    file=[f for f in file if f.endswith(file_ext)]
    images = cv2.imread(os.path.join(imgs[idx], file[0])) #reads the random image
    images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)
    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
    images=images.swapaxes(1, 3).swapaxes(2, 3)
    images = list(image.to(device) for image in images)


    maskDir=os.path.join(imgs[idx], "mask") #path to the mask iamge corresponding to the random image
    masks=[]
    for mskName in os.listdir(maskDir):

        vesMask = cv2.imread(maskDir+'/'+mskName,0) #reads the mask image in greyscale

        vesMask = (vesMask > 0).astype(np.uint8) #The mask image is stored in 0–255 format and is converted to 0–1 format
        vesMask=cv2.resize(vesMask,imageSize,cv2.INTER_NEAREST) #resizes the mask image to the same size of the random image
    
        # print(mask)
        #cv2.imshow("mask",vesMask.astype('float64'))#for plotting if the dtype is uint8 it tries to plot as 0-255 format being 0 black and 255 withe, so 1 its close to 0 so it will be black. But if we use float64 it will take 0 as 0 and 1 as 255. 
        

    with torch.no_grad(): #runs the image through the net and gets a prediction for the object in the image.
        pred = model(images)


    #he predicted object ‘masks’ are saved as a matrix in the same size as the image with each pixel 
    # having a value that corresponds to how likely it is part of the object. And only displays the ones with scores larger than 0.8
    #ssume that only pixels which values larger than 0.5 are likely to be part of the objects.
    # We display this by marking these pixels with a different random color for each object


    im= images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)
    im2 = im.copy()

    for i in range(len(pred[0]['masks'])):
        msk=pred[0]['masks'][i,0].detach().cpu().numpy()
        scr=pred[0]['scores'][i].detach().cpu().numpy()
        #if scr>0.8 :
        im2[:,:,0][msk>0.5] = random.randint(0,255)
        im2[:, :, 1][msk > 0.5] = random.randint(0,255)
        im2[:, :, 2][msk > 0.5] = random.randint(0, 255)
        # else:
        #     scr="the score is too low"
            

    pic = np.hstack([im,im2])
    cv2.imwrite(opath+"/"+"Saved"+str(num)+".png", pic)
    cv2.imshow(str(scr), np.hstack([im,im2]))

    cv2.waitKey()
