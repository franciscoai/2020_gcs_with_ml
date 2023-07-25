import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
from torch.utils.data import DataLoader
from cme_dataset import CmeDataset
from mlp_resnet_model import Mlp_Resnet
from nn.utils.gcs_mask_generator import maskFromCloud


EPOCHS = 10
IMAGE_LIMIT = None
BATCH_SIZE = 16
IMG_SiZE = [512, 512]
GPU = 0
LR = 0.1
GCS_PAR_RNG = torch.tensor([[-180,180],[-70,70],[-90,90],[8,30],[0.2,0.6], [10,60]])
TRAINDIR = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_training_mariano'
OPATH = "/gehme-gpu/projects/2020_gcs_with_ml/output/cme_seg_training_mariano"


def plot_masks(mask, mask_infered, predictions):
    '''
    Plots the target and predicted masks
    '''
    mask_squeeze = np.squeeze(mask)
    mask_infered_squeeze = np.squeeze(mask_infered)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(mask_squeeze)
    ax[0].set_title('Target Mask')
    ax[1].imshow(mask_infered_squeeze)
    ax[1].set_title(f'Predicted Mask\nparams: {predictions}')
    plt.show()
    plt.close()

def compute_loss(predictions, mask, occulter_mask, satpos, plotranges):
    '''
    Computes mean square error between predicted and target masks
    '''
    losses = []
    predictions = predictions.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy()
    satpos = satpos.cpu().detach().numpy()
    plotranges = plotranges.cpu().detach().numpy()
    occulter_mask = occulter_mask.cpu().detach().numpy()
    occulter_mask = occulter_mask.astype(bool)
    counter = 0
    for i in range(predictions.shape[0]):
        mask_infer = maskFromCloud(predictions[i], sat=0, satpos=satpos[i], imsize=IMG_SiZE, plotranges=plotranges[i])
        mask_infer = mask_infer[None, :, :]
        mask_infer[occulter_mask[i]] = 0 #setting 0 where the occulter is
        loss = torch.mean((torch.tensor(mask_infer) - torch.tensor(mask[i]))**2)
        losses.append(loss)
        counter += 1
    return (sum(losses)/len(losses)), mask_infer

def optimize(images_limit=IMAGE_LIMIT):
    losses_per_batch = []
    batch_count = 0
    for epoch in range(EPOCHS):
        stop_flag = False
        model.train()
        for i, (inputs, targets, mask, occulter_mask, satpos, plotranges) in enumerate(cme_dataloader, 0):
            inputs, targets = inputs.to(device), targets.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # send inputs to model and get predictions
            predictions = model(inputs)
            # calculate loss
            #loss = criterion(predictions, targets)
            loss, mask_infer = compute_loss(predictions, mask, occulter_mask, satpos, plotranges)
            loss.requires_grad = True
            # backpropagate loss
            loss.backward()
            # update weights
            optimizer.step()
            # save loss
            losses_per_batch.append(loss.item())
            # print statistics
            if i % 10 == 9:  # print every 10 batches
                print(
                    f'Epoch: {epoch + 1}, Image: {i*BATCH_SIZE}, Batch: {i + 1}, Loss: {loss.item():.3f}, learning rate: {optimizer.param_groups[0]["lr"]:.3f}')
                # print("targets:",targets[-1])
                # print("predictions:",predictions[-1])
                #plot_masks(mask[-1], mask_infer[-1], predictions[-1])
            # add batch to batch count
            batch_count += 1
            # check if we reached the images limit
            if i is not None and i == images_limit:
                stop_flag = True
                break
        if stop_flag:
            break
    # # plot loss
    plt.plot(losses_per_batch)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(OPATH, 'loss.png'))


if __name__ == "__main__":
    cme_dataset = CmeDataset(root_dir=TRAINDIR, img_size=IMG_SiZE)
    cme_dataloader = DataLoader(
        cme_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device(f'cuda:{GPU}') if torch.cuda.is_available(
    ) else torch.device('cpu')  # runing on gpu unles its not available
    print(f'Running on {device}')

    backbone = torchvision.models.resnet101(weights='DEFAULT')
    backbone = backbone.to(device)
    # model = Mlp_Resnet(backbone=backbone, input_size=1000,
    #                    hidden_size=256, output_size=6)
    model = Mlp_Resnet(backbone=backbone, gcs_par_rng=GCS_PAR_RNG)
    model.to(device)
    model_params = [param for param in model.backbone.parameters() if param.requires_grad] + [param for param in model.regression.parameters() if param.requires_grad]
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model_params, lr=LR)    

    optimize()
