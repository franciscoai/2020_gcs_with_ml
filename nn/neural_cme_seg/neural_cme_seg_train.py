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
import logging

mpl.use('Agg')

def loadData(paths, batchSize, used_idx, imageSize=None, file_ext=".png", normalization_func=None, masks2use=None, rnd_rot=False):
    '''
    Loads random batchSize traning cases from paths avoiding training cases in used_idx
    paths: list of paths to use
    batchSize: number of images to load
    used_idx: list of indexes to avoid
    normalization_func: function to normalize the images, use None for no normalization
    masks2use: list of masks to use, use None to use all masks found in the mask directory
    rnd_rot: if True, the images are randomly rotated
    imageSize: resize of the images, if None, the original size is used
    '''    
    # elminates used paths
    used_idx = set(used_idx) # It uses set because it is faster to check if an element is in a set than in a list
    ok_paths = [path for i, path in enumerate(paths) if i not in used_idx] 
    if len(ok_paths)==0: 
        return loadData(paths, batchSize, [], imageSize, file_ext, normalization_func, masks2use, rnd_rot)
    batch_Imgs=[]
    batch_Data=[]
    new_used_idx=[]
    for i in range(batchSize):        
        idx=random.randint(0,len(ok_paths)-1) #takes a random image from the clean training list
        new_used_idx.append(paths.index(ok_paths[idx]))
        file=os.listdir(ok_paths[idx])
        file=[f for f in file if f.endswith(file_ext)][0]
        img = cv2.imread(os.path.join(ok_paths[idx], file)) #reads the random image
        if imageSize is not None:
            img = cv2.resize(img, imageSize, cv2.INTER_LINEAR) #rezise the image  
        if normalization_func is not None:
            img = normalization_func(img)
        maskDir=os.path.join(ok_paths[idx], "mask") #path to the mask corresponding to the random image
        masks=[]
        labels = []
        lbl_idx=1
        ok_masks=os.listdir(maskDir) #all masks in the mask directory
        ok_masks.sort(key=lambda x: int(x.split("_")[-1].split(".")[0])) #sort masks files by the ending number
        if masks2use is not None:
            ok_masks = [ok_masks[i] for i in masks2use]
        for mskName in ok_masks:
            labels.append(lbl_idx)
            vesMask = cv2.imread(maskDir+'/'+mskName, 0) #reads the mask image in greyscale 
            vesMask = (vesMask > 0).astype(np.uint8) #The mask image is stored in 0–255 format and is converted to 0–1 format
            if imageSize is not None:
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
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8) #converts the mask list into a tensor
        img = torch.as_tensor(img, dtype=torch.float32)   #converts the image list into a tensor     
        data = {} #creates data dictionary
        data["boxes"] =  boxes
        data["labels"] = torch.tensor(labels, dtype=torch.int64)   # returns a tensor filled with ones, (there is only one class)
        data["masks"] = masks        
        batch_Imgs.append(img)
        batch_Data.append(data)  # load images and masks
        
    #greyscale to 3 identical RGB    
    batch_Imgs=torch.stack([torch.as_tensor(d) for d in batch_Imgs],0) #Concatenates a sequence of tensors along a new dimension formed from the images in greyscale
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data, new_used_idx

#---------------------------------------------------------Fine_tune the pretrained R-CNN----------------------------------------------------------
#Constants
trainDir = '/gehme/projects/2020_gcs_with_ml/data/cme_seg_20240702'
opath= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v5"
#full path of a model to use it as initial condition, use None to used the stadard pre-trained model 
pre_trained_model= None # "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v2_running_diff/3999.torch"
batchSize=12 #number of images used in each iteration
epochs=5 #number of iterations of the full training dataset
train_dataset_prop=0.85 #proportion of the full dataset used for training. The rest is saved for validation
random_rot = False # if True, the images are randomly rotated
gpu=0 # GPU to use
masks2use=[2] # list of masks to use, use None to use all masks found in the mask directory
model_version='v5' # version of the model to use
logfile=opath + "/training_log.txt" # log file

#main
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unles its not available
logging.info(f'Using device:  {device}')
#flush cuda device memory
torch.cuda.empty_cache()
os.makedirs(opath,exist_ok=True)

# saves a copy of this file to opath
os.system(f'cp {__file__} {opath}')
# saves a copy of the model to opath
os.system(f'cp nn/neural_cme_seg/neural_cme_seg.py {opath}')

# logger
logging.basicConfig(filename=logfile, level=logging.INFO)

#list of images on the trainig dataset
imgs=[] #list of images on the trainig dataset
dirs=os.listdir(trainDir)
dirs=[pth for pth in dirs if os.path.isdir(trainDir+"/"+pth)] # keeps only dirs
for pth in dirs:
    imgs.append(trainDir+"/"+pth)
logging.info(f'The total number of images found is {len(imgs)}')

# separates the dataset into training and validation
random.shuffle(imgs)
imgs_train = imgs[:int(len(imgs)*train_dataset_prop)]
imgs_val = imgs[int(len(imgs)*train_dataset_prop):]
logging.info(f'The total number of images used for training is {len(imgs_train)}')
# saves the list of images used for training and validation as csv files
with open(opath + "/training_cases.csv", 'w') as file:
    for i in imgs_train:
        file.write(i + '\n')
with open(opath + "/validation_cases.csv", 'w') as file:
    for i in imgs_val:
        file.write(i + '\n')

# loads nn model and sets it to train mode
nn_seg = neural_cme_segmentation(device, pre_trained_model = pre_trained_model, version=model_version, logger=logging)
nn_seg.train()  

#training
all_loss=[]
for i in range(epochs):
    used_idx=[]
    for j in range(0,len(imgs_train),batchSize):
        images, targets, used_idx_batch = loadData(imgs_train, batchSize, used_idx, normalization_func=nn_seg.normalize, 
                                                   masks2use=masks2use, rnd_rot=random_rot) # loads a batch of training data
        used_idx.extend(used_idx_batch)
        images = list(image.to(device) for image in images)
        targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
        nn_seg.optimizer.zero_grad() 
        loss_dict = nn_seg.model(images, targets)
        losses = sum(loss for loss in loss_dict.values()) 
        losses.backward()
        nn_seg.optimizer.step()
        all_loss.append(losses.item())
        cn_img=j+i*len(imgs_train)
        logging.info(f'Epoch {i} of {epochs}, batch {j//batchSize}, images {cn_img} ({(cn_img)/(epochs*len(imgs_train))*100:.1f}%), loss: {losses.item():.3f}')
    
    #save training results after each epoch
    #model
    torch.save(nn_seg.model.state_dict(),opath + "/" + str(i)+".torch")
    #all losses in a pickle file
    with open(opath + "/all_loss", 'wb') as file:
        pickle.dump(all_loss, file, protocol=pickle.HIGHEST_PROTOCOL)
    #plot loss
    plt.plot(np.arange(len(all_loss))*batchSize+1,all_loss)
    #add vertical line for each epoch
    for k in range(i):
        plt.axvline(x=k*len(imgs_train), color='r', linestyle='--')
    plt.ylabel('Taining loss')
    plt.xlabel('Images')
    plt.grid('both')
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(opath + '/all_loss.png')
    plt.close()
