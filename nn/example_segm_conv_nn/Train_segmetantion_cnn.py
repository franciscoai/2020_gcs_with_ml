import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import os

dataDir = "/gehme-gpu/projects/2020_gcs_wiht_ml/data/example_segm_conv_nn/LabPicsChemistry/LabPicsChemistry"
trainDir=  dataDir + "/Train"
testDir=  dataDir + "/Test"
opath= "/gehme-gpu/projects/2020_gcs_wiht_ml/output/example_segm_conv_nn"


#------------------------------------------------------------trainign of the CNN--------------------------------------------------------------------------------------#

batchSize=2 #number of images used in each iteration
imageSize=[600,600] 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unles its not available
imgs=[] #list of images on the trainig dataset
for pth in os.listdir(trainDir):
    imgs.append(trainDir+"/"+pth +"//")

def loadData():
    batch_Imgs=[]
    batch_Data=[]
    for i in range(batchSize):        
        idx=random.randint(0,len(imgs)-1) #takes a random image from the image training list
        img = cv2.imread(os.path.join(imgs[idx], "Image.jpg")) #reads the random image
        img = cv2.resize(img, imageSize, cv2.INTER_LINEAR) #rezices the image       
        maskDir=os.path.join(imgs[idx], "Vessels") #path to the mask iamge corresponding to the random image
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
        masks = torch.as_tensor(masks, dtype=torch.uint8) #converts the mask list into a tensor
        img = torch.as_tensor(img, dtype=torch.float32)   #converts the image list into a tensor     
        
        data = {} #creates data dictionary
        data["boxes"] =  boxes
        data["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   # returns a tensor filled with ones, (there is only one class)
        data["masks"] = masks        
        
        batch_Imgs.append(img)
        batch_Data.append(data)  # load images and masks
        
    batch_Imgs=torch.stack([torch.as_tensor(d) for d in batch_Imgs],0) #Concatenates a sequence of tensors along a new dimension formed from the images in greyscale
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data 


#---------------------------------------------------------Defines the CNN by a pre trained R-CNN----------------------------------------------------------

model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)   # load an instance segmentation model pre-trained pre-trained on COCO dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features # get number of input features for the classifier
model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=2) # replace the pre-trained head with a new one
model.to(device) # move model to the right device
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5) # optimization technique that comes under gradient decent algorithm
model.train()#sets the model to train mode

for i in range(5001): #Number of iterations starting in zero, so if we want 5000 iteration we will need 5001 iteration
    images, targets= loadData() #call the function, images=batch_img and targets=batch_data
    images = list(image.to(device) for image in images) #send images to the selected device
    targets=[{k: v.to(device) for k,v in t.items()} for t in targets] #send targets to the selected device

   
    optimizer.zero_grad() #Sets the gradients of all optimized torch.Tensors to zero.
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values()) #The loss is composed of several parts: class loss, bounding box loss, and mask loss. We sum all of these parts together to get the total loss as a single number
   
    losses.backward() #computes the partial derivative of the output f with respect to each of the input variables.
    optimizer.step()
   
    print(i,'loss:', losses.item())
    if i%500==0:
        torch.save(model.state_dict(),opath + "/" + str(i)+".torch")