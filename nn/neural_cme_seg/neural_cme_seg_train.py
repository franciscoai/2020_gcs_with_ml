import random
import numpy as np
import torch.utils.data
import cv2
import torch
import os
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from neural_cme_seg import neural_cme_segmentation
import scipy

mpl.use('Agg')

def loadData(imgs, batchSize, imageSize=[512,512], file_ext=".png", normalization_func=None, masks2use=None, rnd_rot=False):
    '''
    Loads a batch of images and targets
    normalization_func: function to normalize the images, use None for no normalization
    masks2use: list of masks to use, use None to use all masks found in the mask directory
    rnd_rot: if True, the images are randomly rotated
    '''    
    batch_Imgs=[]
    batch_Data=[]
    #batch_Masks=[]
    for i in range(batchSize):        
        idx=random.randint(0,len(imgs)-1) #takes a random image from the image training list
        file=os.listdir(imgs[idx])
        file=[f for f in file if f.endswith(file_ext)]
        img = cv2.imread(os.path.join(imgs[idx], file[0])) #reads the random image
        img = cv2.resize(img, imageSize, cv2.INTER_LINEAR) #rezise the image  
        if normalization_func is not None:
            img = normalization_func(img)
        maskDir=os.path.join(imgs[idx], "mask") #path to the mask corresponding to the random image
        masks=[]
        labels = []
        lbl_idx=1
        ok_masks=os.listdir(maskDir) #list of all masks in the mask directory
        if masks2use is not None:
            ok_masks = [ok_masks[i] for i in masks2use]
        for mskName in ok_masks:
            labels.append(lbl_idx)
            vesMask = cv2.imread(maskDir+'/'+mskName, 0) #reads the mask image in greyscale 
            vesMask = (vesMask > 0).astype(np.uint8) #The mask image is stored in 0–255 format and is converted to 0–1 format
            vesMask=cv2.resize(vesMask,imageSize,cv2.INTER_NEAREST) #resizes the mask image to the same size of the random image
            masks.append(vesMask) # get bounding box coordinates for each mask  
            lbl_idx+=1
        #optional random rotation
        if rnd_rot:
            rot_ang = np.random.randint(low=0, high=360)
            img = scipy.ndimage.rotate(img, rot_ang, axes=(1, 0), reshape=False, order=0)
            masks = [scipy.ndimage.rotate(m, rot_ang, axes=(1, 0), reshape=False, order=0) for m in masks]
            masks = [(m>0).astype(np.uint8) for m in masks]
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
        data["labels"] = torch.tensor(labels, dtype=torch.int64)   # returns a tensor filled with ones, (there is only one class)
        data["masks"] = masks        
        batch_Imgs.append(img)
        batch_Data.append(data)  # load images and masks
        
    #greyscale to 3 identical RGB ??    
    batch_Imgs=torch.stack([torch.as_tensor(d) for d in batch_Imgs],0) #Concatenates a sequence of tensors along a new dimension formed from the images in greyscale
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data #, batch_Masks

#---------------------------------------------------------Fine_tune the pretrained R-CNN----------------------------------------------------------
"""
"""
#Constants
#trainDir = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_training'
trainDir = '/gehme/projects/2020_gcs_with_ml/data/cme_seg_1VP_100k'
#opath= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4"
opath= "/gehme/projects/2020_gcs_with_ml/output/neural_cme_seg_v5"
#full path of a model to use it as initial condition, use None to used the stadard pre-trained model 
pre_trained_model= None # "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v2_running_diff/3999.torch"
batchSize=12 #number of images used in each iteration
train_ncases=66667 # Total no. of epochs
random_rot = True # if True, the images are randomly rotated
gpu=0 # GPU to use
masks2use=[0,1] # list of masks to use, use None to use all masks found in the mask directory
model_version='v4' # version of the model to use

#main
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unles its not available
print(f'Using device:  {device}')
#flush cuda device memory
torch.cuda.empty_cache()
os.makedirs(opath,exist_ok=True)
imgs=[] #list of images on the trainig dataset
dirs=os.listdir(trainDir)
dirs= [d for d in dirs if not d.endswith(".csv")]
for pth in dirs:
    imgs.append(trainDir+"/"+pth)
print(f'The total number of training images found is {len(imgs)}')

# loads nn model and sets it to train mode
nn_seg = neural_cme_segmentation(device, pre_trained_model = pre_trained_model, version=model_version)
nn_seg.train()  

#training
all_loss=[]
for i in range(train_ncases): 
    images, targets= loadData(imgs, batchSize, normalization_func=nn_seg.normalize, masks2use=masks2use, rnd_rot=random_rot) # loads a batch of training data
    images = list(image.to(device) for image in images)
    targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
    nn_seg.optimizer.zero_grad() 
    loss_dict = nn_seg.model(images, targets)
    losses = sum(loss for loss in loss_dict.values()) 
    losses.backward()
    nn_seg.optimizer.step()
   
    all_loss.append(losses.item())
    print(i,'loss:', losses.item())
    if (i>0) and (i%2000==0):
        torch.save(nn_seg.model.state_dict(),opath + "/" + str(i)+".torch")
        #saves all losses in a pickle file
        with open(opath + "/all_loss", 'wb') as file:
            pickle.dump(all_loss, file, protocol=pickle.HIGHEST_PROTOCOL)
        #plots losss
        plt.plot(all_loss)
        plt.title('Training loss')
        plt.grid('both')
        plt.yscale("log")
        plt.savefig(opath + '/all_loss.png')
        plt.close()

torch.save(nn_seg.model.state_dict(),opath + "/" + str(i)+".torch")
#saves all losses in a pickle file
with open(opath + "/all_loss", 'wb') as file:
    pickle.dump(all_loss, file, protocol=pickle.HIGHEST_PROTOCOL)
#plots losss
plt.plot(all_loss)
plt.title('Training loss')
plt.grid('both')
plt.yscale("log")
plt.savefig(opath + '/all_loss.png')
plt.close()
