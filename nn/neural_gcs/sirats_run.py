import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('TkAgg')
mpl.use('Agg')
from torch.utils.data import DataLoader
from cme_dataset import CmeDataset
from sirats_model import Sirats_net
from human_shape_net import HSnet
from nn.utils.gcs_mask_generator import maskFromCloud
from torch.utils.data import random_split

# Train Parameters
DEVICE = 0
INFERENCE_MODE = False
SAVE_MODEL = True
LOAD_MODEL = False
EPOCHS = 15
BATCH_LIMIT = None
BATCH_SIZE = 128
IMG_SiZE = [512, 512]
GPU = 0
LR = [1e-3, 1e-4]
# CMElon,CMElat,CMEtilt,height,k,ang
GCS_PAR_RNG = torch.tensor([[-180,180],[-70,70],[-90,90],[8,30],[0.2,0.6], [10,60]]) 
LOSS_WEIGHTS = torch.tensor([100,100,100,10,1,10])
TRAINDIR = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_training_mariano'
OPATH = "/gehme-gpu/projects/2020_gcs_with_ml/output/cme_seg_training_mariano_newmodel"
os.makedirs(OPATH, exist_ok=True)


def run_training():
    train_losses_per_batch = []
    #test_losses_per_batch = []
    batch_count = 0
    epoch_list = []
    total_batch_per_epoch = 0
    for epoch in range(EPOCHS):
        epoch_list.append(total_batch_per_epoch)
        stop_flag = False
        total_batch_per_epoch = 0
        # add batch to batch count
        batch_count += 1
        for i, (img, targets) in enumerate(cme_train_dataloader, 0):
            loss_value = model.optimize_model(img, targets, loss_fn, optimizer, scheduler)
            train_losses_per_batch.append(loss_value.detach().cpu())
            # print statistics
            if i % 10 == 0:  # print every 10 batches
                print(f'Epoch: {epoch + 1}, Image: {(i+1)*BATCH_SIZE}, Batch: {i + 1}, Loss: {loss_value:.5f}, learning rate: {optimizer.param_groups[0]["lr"]:.5f}')
            if i % 50 == 0:
                model.plot_loss(train_losses_per_batch, epoch_list, BATCH_SIZE, os.path.join(OPATH, "train_loss.png"))
                #model.plot_loss(test_losses_per_batch, epoch_list, os.path.join(OPATH, "test_loss.png"))
            # check if we reached the images limit
            if i is not None and i == BATCH_LIMIT:
                stop_flag = True
                break
        if stop_flag:
            break
    #save model  
    if SAVE_MODEL:
        # save_model(model, "model.pth")
        pass


if __name__ == '__main__':
    # Generate dataloaders
    dataset = CmeDataset(root_dir=TRAINDIR, img_size=IMG_SiZE)
    train_dataset, test_dataset = random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])
    cme_train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    cme_test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define model, criterion, optimizer
    model = Sirats_net(device=DEVICE, output_size=6, imsize=IMG_SiZE)
    #print num of parameters
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    optimizer = torch.optim.Adadelta(model.parameters(), lr=LR[0], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (len(cme_train_dataloader)/BATCH_SIZE)*EPOCHS, eta_min=LR[1])
    loss_fn = torch.nn.MSELoss()

    # Run training
    run_training()
