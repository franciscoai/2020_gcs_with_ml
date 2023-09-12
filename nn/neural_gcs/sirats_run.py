import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('TkAgg')
mpl.use('Agg')
from torch.utils.data import DataLoader
from nn.neural_gcs.cme_1VP_dataset import Cme_1VP_Dataset
from sirats_model import Sirats_net
from nn.utils.gcs_mask_generator import maskFromCloud
from torch.utils.data import random_split

# Train Parameters
DEVICE = 0
INFERENCE_MODE = False
SAVE_MODEL = True
LOAD_MODEL = True
EPOCHS = 200
BATCH_LIMIT = None
BATCH_SIZE = 32
IMG_SiZE = [512, 512]
GPU = 0
LR = [1e-3, 1e-5]
# CMElon,CMElat,CMEtilt,height,k,ang
GCS_PAR_RNG = torch.tensor([[-180,180],[-70,70],[-90,90],[8,30],[0.2,0.6], [10,60]]) 
LOSS_WEIGHTS = torch.tensor([100,100,100,10,1,10])
TRAINDIR = '/gehme-gpu/projects/2020_gcs_with_ml/data/gcs_ml_1VP_100k'
OPATH = "/gehme-gpu/projects/2020_gcs_with_ml/output/sirats_v3_200epochs_1VP_100k"
os.makedirs(OPATH, exist_ok=True)

def plot_masks(img, mask, target, prediction, occulter_mask, satpos, plotranges, opath, namefile):
    img = img.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()
    satpos = satpos.cpu().detach().numpy()
    plotranges = plotranges.cpu().detach().numpy()
    occulter_mask = occulter_mask.cpu().detach().numpy()

    loss = loss_fn(torch.tensor(prediction), torch.tensor(target[None,:]))  # Assuming loss_fn is defined
    prediction = np.squeeze(prediction)
    mask_infered = maskFromCloud(prediction, sat=0, satpos=[satpos], imsize=IMG_SiZE, plotranges=[plotranges])
    mask_infered[occulter_mask > 0] = 0  # Setting 0 where the occulter is
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'target: {np.around(target, 3)}\nPrediction: {np.around(prediction, 3)}')
    
    for i in range(2):
        img = np.squeeze(img)
        ax[i].imshow(img, vmin=0, vmax=1, cmap='gray')
        if i == 0:
            ax[i].imshow(mask, alpha=0.4)
            ax[i].set_title('Target Mask')
        else:
            ax[i].imshow(mask_infered, alpha=0.4)
            ax[i].set_title(f'Prediction Mask: {loss}')
    
    masks_dir = os.path.join(opath, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    plt.savefig(os.path.join(masks_dir, namefile))
    plt.close()


def run_training():
    train_losses_per_batch = []
    total_batches_per_epoch = 0
    epoch_list = []

    for epoch in range(EPOCHS):
        epoch_list.append(total_batches_per_epoch)  # Store the total number of batches processed
        total_batches_per_epoch = 0

        for i, (img, targets, mask, occulter_mask, satpos, plotranges, idx) in enumerate(cme_train_dataloader, 0):
            total_batches_per_epoch += 1
            loss_value = model.optimize_model(img, targets, loss_fn, optimizer, scheduler)
            train_losses_per_batch.append(loss_value.detach().cpu())

            if i % 10 == 0:            
                print(f'Epoch: {epoch + 1}, Image: {(i + 1) * BATCH_SIZE}, Batch: {i + 1}, Loss: {loss_value:.5f}, learning rate: {optimizer.param_groups[-1]["lr"]:.7f}')

            if i % 50 == 0:
                model.plot_loss(train_losses_per_batch, epoch_list, BATCH_SIZE, os.path.join(OPATH, "train_loss.png"), plot_epoch=False)

            if i == BATCH_LIMIT:
                break

        if i == BATCH_LIMIT:
            break

    # Save model
    if SAVE_MODEL:
        status = model.save_model(OPATH)
        print(f"\nModel saved at: {status}\n")



if __name__ == '__main__':
    # train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.90), int(len(dataset) * 0.1)])
    dataset = Cme_1VP_Dataset(root_dir=TRAINDIR, img_size=IMG_SiZE)
    total_samples = len(dataset)
    train_size = 95000
    train_indices = random.sample(range(total_samples), train_size)
    test_indices = list(set(range(total_samples)) - set(train_indices))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    cme_train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    cme_test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Sirats_net(device=DEVICE, output_size=6, imsize=IMG_SiZE)
    if LOAD_MODEL:
        status = model.load_model(OPATH)
        if status:
            print(f"Model loaded from: {status}\n")
        else:
            print(f"No model found at: {OPATH}, starting from scratch\n")

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_parameters}\n')

    optimizer = torch.optim.Adadelta(model.parameters(), lr=LR[0])
    #optimizer = torch.optim.Adam(model.parameters(), lr=LR[0])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (len(cme_train_dataloader) / BATCH_SIZE) * EPOCHS, eta_min=LR[1])
    loss_fn = torch.nn.MSELoss()

    if not INFERENCE_MODE:
        run_training()
    else:
        data_iter = iter(cme_test_dataloader)
        for i in range(10):
            img, targets, mask, occulter_mask, satpos, plotranges, idx = next(data_iter)
            img, targets, mask, occulter_mask, satpos, plotranges, idx = img[0], targets[0], mask[0], occulter_mask[0], satpos[0], plotranges[0], idx[0]
            img = img.to(DEVICE)
            predictions = model.infer(img)
            plot_masks(img, mask, targets, predictions, occulter_mask, satpos, plotranges, opath=OPATH, namefile=f'targetVinfered_{idx}.png')
