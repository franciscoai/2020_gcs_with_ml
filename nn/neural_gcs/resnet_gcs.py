import random
import os
import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from mlp_resnet_model import Mlp_Resnet

EPOCHS = 1000
GPU = 1
TRAINDIR = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_training_mariano'
OPATH = "/gehme-gpu/projects/2020_gcs_with_ml/output/cme_seg_training_mariano"


def normalize(image):
    '''
    Normalizes the values of the model input image to have a given range (as fractions of the sd around the mean)
    maped to [0,1]. It clips output values outside [0,1]
    '''
    sd_range = 1.
    m = np.mean(image)
    sd = np.std(image)
    image = (image - m + sd_range * sd) / (2 * sd_range * sd)
    image[image > 1] = 1
    image[image < 0] = 0
    return image


def dataloader(imgs, batch_size=8, image_size=[512, 512], file_ext='.png'):
    batch_imgs = []
    targets = []
    for i in range(batch_size):
        idx = random.randint(0, len(imgs)-1)
        file = os.listdir(imgs[idx])
        file = [f for f in file if f.endswith(file_ext)]
        img = cv2.imread(os.path.join(imgs[idx], file[0]))
        img = cv2.resize(img, image_size, cv2.INTER_LINEAR)
        img = normalize(img)
        img = np.transpose(img, (2, 0, 1))
        parameters = file[0].split('_')
        parameters = parameters[:6]
        parameters = [float(p) for p in parameters]
        parameters = torch.tensor(parameters, dtype=torch.float32)
        targets.append(parameters) # targets is a list where each element is a tensor of shape [6]
        batch_imgs.append(img)
    batch_imgs=torch.stack([torch.as_tensor(d) for d in batch_imgs],0).float()
    targets = torch.stack([torch.as_tensor(d) for d in targets],0).float()
    return batch_imgs, targets


def optimize():
    all_losses = []
    for epoch in range(EPOCHS):
        batch_imgs, targets = dataloader(images)
        batch_imgs = batch_imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        features = model.backbone(batch_imgs)
        #features = torch.flatten(features, start_dim=1)
        output = model(features)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{EPOCHS} Loss: {loss.item():.4f}')
        # if epoch % 10 == 0:
        #     torch.save(model.state_dict(), os.path.join(OPATH, f'model_{epoch+1}.pth'))
        all_losses.append(loss.item())
    # plot loss
    plt.plot(all_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(OPATH, 'loss.png'))


if __name__ == "__main__":
    images = []
    dirs = os.listdir(TRAINDIR)
    dirs= [d for d in dirs if not d.endswith(".csv")]
    for d in dirs:
        images.append(os.path.join(TRAINDIR, d))
    print(f'Found {len(images)} images')
    print(f'Using GPU {GPU}')
    device = torch.device(f'cuda:{GPU}') if torch.cuda.is_available(
    ) else torch.device('cpu')  # runing on gpu unles its not available

    #backbone, in_features = generateBackbone()
    backbone = torchvision.models.resnet101(weights='DEFAULT')
    for param in backbone.parameters():
        param.requires_grad = False
    model = Mlp_Resnet(backbone=backbone, input_size=1000,
                       hidden_size=256, output_size=6)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-2)

    optimize()