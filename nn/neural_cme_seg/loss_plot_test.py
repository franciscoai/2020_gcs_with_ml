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
#------------------------------------------------------------trainign of the CNN--------------------------------------------------------------------------------------#
dataDir = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_dataset_fran_test'
opath= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_fran/test_output"
file_ext=".png"
trainDir=  dataDir 
testDir=  dataDir 
batchSize=8 #number of images used in each iteration
imageSize=[512,512] 
train_ncases=5000 # Total no. of epochs
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unles its not available

#main
print(f'Using device:  {device}')
os.makedirs(opath,exist_ok=True)
imgs=[] #list of images on the trainig dataset
dirs=os.listdir(trainDir)
dirs= [d for d in dirs if not d.endswith(".csv")]
for pth in dirs:
    imgs.append(trainDir+"/"+pth)
print(f'The total number of training images found is {len(imgs)}')

def loadData():
    batch_Imgs=[]
    batch_Data=[]
    #batch_Masks=[]
    for i in range(batchSize):        
        idx=random.randint(0,len(imgs)-1) #takes a random image from the image training list
        file=os.listdir(imgs[idx])
        file=[f for f in file if f.endswith(file_ext)]
        img = cv2.imread(os.path.join(imgs[idx], file[0])) #reads the random image
        img = cv2.resize(img, imageSize, cv2.INTER_LINEAR) #rezise the image  
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) # normalize to 0,1
        maskDir=os.path.join(imgs[idx], "mask") #path to the mask corresponding to the random image
        masks=[]
        for mskName in os.listdir(maskDir):
            vesMask = cv2.imread(maskDir+'/'+mskName, 0) #reads the mask image in greyscale 
            vesMask = (vesMask > 0).astype(np.uint8) #The mask image is stored in 0–255 format and is converted to 0–1 format
            vesMask=cv2.resize(vesMask,imageSize,cv2.INTER_NEAREST) #resizes the mask image to the same size of the random image
            masks.append(vesMask) # get bounding box coordinates for each mask  
        num_objs = len(masks) #amount of objects in the image
        if num_objs==0: return loadData()        # if image have no objects just load another image
        boxes = torch.zeros([num_objs,4], dtype=torch.float32) #Returns a tensor filled with the scalar value 0 of the defined size
    
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(masks[i]) #draws an approximate rectangle around the binary  mask image and returns "x" and "y" coordinates and the "width" and "height" of the object
            boxes[i] = torch.tensor([x, y, x+w, y+h]) #creates a tensor with the parameters obtained before
            if h==0 or w==0:
                plt.imshow(masks[i])
                plt.title(str(x)+"/"+str(y)+"/"+str(w)+"/"+str(h))
                plt.show()
        masks = torch.as_tensor(masks, dtype=torch.uint8) #converts the mask list into a tensor
        img = torch.as_tensor(img, dtype=torch.float32)   #converts the image list into a tensor     
        
        data = {} #creates data dictionary
        data["boxes"] =  boxes
        data["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   # returns a tensor filled with ones, (there is only one class)
        data["masks"] = masks        
        
        batch_Imgs.append(img)
        batch_Data.append(data)  # load images and masks
        
    #greyscale to 3 identical RGB ??    
    batch_Imgs=torch.stack([torch.as_tensor(d) for d in batch_Imgs],0) #Concatenates a sequence of tensors along a new dimension formed from the images in greyscale
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data #, batch_Masks

#---------------------------------------------------------Defines the CNN by a pre trained R-CNN----------------------------------------------------------
model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)   # load an instance segmentation model pre-trained on COCO dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features # get number of input features for the classifier
model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=2) # replace the pre-trained head with a new one
model.to(device) # move model to the right device
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5) # optimization technique that comes under gradient decent algorithm
model.train()#sets the model to train mode

test_loss=[]
for i in range(train_ncases): #Number of iterations
    images, targets= loadData() #call the function, images=batch_img and targets=batch_data
    images = list(image.to(device) for image in images) #send images to the selected device
    targets=[{k: v.to(device) for k,v in t.items()} for t in targets] #send targets to the selected device
    #masks=list(image.to(device) for image in masks)
   
    #optimizer.zero_grad() #Sets the gradients of all optimized torch.Tensors to zero.
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values()) #The loss is composed of several parts: class loss, bounding box loss, and mask loss. We sum all of these parts together to get the total loss as a single number
   

   
    test_loss.append(losses.item())
    print(i,'loss:', losses.item())
    if i%1000==0:
        torch.save(model.state_dict(),opath + "/" + str(i)+".torch")

torch.save(model.state_dict(),opath + "/" + str(i)+".torch")

#saves all losses in a pickle file
with open(opath + "/test_loss", 'wb') as file:
    pickle.dump(test_loss, file, protocol=pickle.HIGHEST_PROTOCOL)
