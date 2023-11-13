import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
import numpy as np
import random
import torchvision
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
#mpl.use('TkAgg')
from pyGCS_raytrace import pyGCS
from astropy.io import fits
from torch.utils.data import DataLoader
from nn.neural_gcs.cme_mvp_dataset import Cme_MVP_Dataset
from nn.neural_gcs.sirats_model import Sirats_net
from nn.utils.gcs_mask_generator import maskFromCloud


def calculate_weights(ranges):
    weights = []

    for r in ranges:
        min_val, max_val = r
        range_size = max_val - min_val

        # Avoid division by zero
        if range_size == 0:
            weight = 10.0  # Assign a weight of 10 if the range size is zero
        else:
            weight = 10.0 / range_size

        weights.append(weight)

    return weights

# Dataset Parameters
TRAINDIR = '/gehme-gpu/projects/2020_gcs_with_ml/data/gcs_ml_3VP_onlyMask_size_100000_seed_59199'
OPATH = "/gehme-gpu/projects/2020_gcs_with_ml/output/gcs_ml_3VP_onlyMask_size_100000_seed_59199"
BINARY_MASK = True
BATCH_SIZE = 8
BATCH_LIMIT = None
SEED = 42
IMG_SiZE = [3, 512, 512] # If [None, x, y], then the image size is not changed, otherwise it is resized to the specified size

# Train Parameters
DEVICE = 1
INFERENCE_MODE = False
SAVE_MODEL = True
LOAD_MODEL = True
EPOCHS = 30
TRAIN_IDX_SIZE = 55000
GPU = 0
LR = [1e-2, 1e-3]
PAR_RNG = [[-180,180],[-70,70],[-90,90],[3,10],[0.2,0.6],[10,60],[1e-1,1e1]]
PAR_LOSS_WEIGHTS = torch.tensor(calculate_weights(PAR_RNG[:6])) #torch.tensor([0.1,0.1,0.1,1,10,0.5])
os.makedirs(OPATH, exist_ok=True)


def test_specific_image():
    sat1_path = "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_exp_paper_filtered/GCS_20110317_filter_True/FMwLASCO201103173_0.fits"
    sat2_path = "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_exp_paper_filtered/GCS_20110317_filter_True/FMwLASCO201103173_1.fits"
    resize = torchvision.transforms.Resize(IMG_SiZE[1:3], torchvision.transforms.InterpolationMode.BILINEAR)
    #read fits
    sat1 = fits.open(sat1_path)[0].data
    sat2 = fits.open(sat2_path)[0].data
    sat1_h = fits.open(sat1_path)[0].header
    sat2_h = fits.open(sat2_path)[0].header
    headers = [sat1_h, sat2_h]
    satpos, plotranges = pyGCS.processHeaders(headers)
    satpos = np.array(satpos)
    plotranges = np.array(plotranges)
    img = torch.tensor([sat1, sat2, sat1]) # it repeats because Lasco is not implemented yet
    img = img.float()
    if not BINARY_MASK:
        sd_range = 1
        m = torch.mean(img)
        sd = torch.std(img)
        img = (img - m + sd_range * sd) / (2 * sd_range * sd)
    else:
        img[img > 1] = 1
        img[img < 0] = 0
    img = resize(img)
    img = img.to(DEVICE)
    predictions = model.infer(img)
    predictions = predictions.cpu().detach().numpy()
    img = img.cpu().detach().numpy()
    predictions = np.squeeze(predictions)
    mask_infered_sat1 = maskFromCloud(predictions, sat=0, satpos=[satpos[0,:]], imsize=IMG_SiZE[1:3], plotranges=[plotranges[0,:]])
    mask_infered_sat2 = maskFromCloud(predictions, sat=0, satpos=[satpos[1,:]], imsize=IMG_SiZE[1:3], plotranges=[plotranges[1,:]])
    
    fig, ax = plt.subplots(1, 2, figsize=(9, 5))

    # Define colors and colormap
    color = ['purple', 'r']
    cmap = mpl.colors.ListedColormap(color)

    # Tight layout
    fig.tight_layout()

    # Set the main title
    fig.suptitle("Lasco specific image")

    # Update masks for saturation
    mask_infered_sat1[mask_infered_sat1 > 0] = 1
    mask_infered_sat2[mask_infered_sat2 > 0] = 1
    mask_infered_sat1[mask_infered_sat1 <= 0] = np.nan
    mask_infered_sat2[mask_infered_sat2 <= 0] = np.nan

    # Set image values
    img[img <= 0] = np.nan
    img[img > 0] = 0

    # Plot the masks and images
    ax[0].imshow(mask_infered_sat1, cmap=cmap)
    ax[0].imshow(img[0, :, :], vmin=0, vmax=1, alpha=0.4, cmap=cmap)

    ax[1].imshow(mask_infered_sat2, cmap=cmap)
    ax[1].imshow(img[1, :, :], vmin=0, vmax=1, alpha=0.4, cmap=cmap)

    # Save the figure
    masks_dir = os.path.join(OPATH, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    plt.savefig(os.path.join(masks_dir, 'test_image.png'))
    plt.close()

def plot_mask_MVP(img, sat_masks, target, prediction, occulter_masks, satpos, plotranges, opath, namefile):
    # Convert tensors to numpy arrays
    img = img.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    prediction = np.squeeze(prediction.cpu().detach().numpy())
    satpos = satpos.cpu().detach().numpy()
    plotranges = plotranges.cpu().detach().numpy()
    occulter_masks = occulter_masks.cpu().detach().numpy()

    IMG_SIZE = (img.shape[0], img.shape[1], img.shape[2])  # Assuming you have IMG_SIZE defined

    fig, ax = plt.subplots(1, IMG_SIZE[0], figsize=(9, 5))
    fig.tight_layout()
    fig.suptitle(f'target: {np.around(target, 3)}\nPrediction: {np.around(prediction, 3)}')

    color = ['purple', 'k', 'r', 'b']
    cmap = mpl.colors.ListedColormap(color)

    for i in range(IMG_SIZE[0]):
        mask_infered_sat = maskFromCloud(prediction, sat=0, satpos=[satpos[i, :]], imsize=IMG_SIZE[1:3], plotranges=[plotranges[i, :]])
        masks_infered = np.zeros(IMG_SIZE)
        masks_infered[i, :, :] = mask_infered_sat

        nan_mask = np.full(IMG_SIZE[1:3], np.nan)
        nan_occulter = np.full(IMG_SIZE[1:3], np.nan)

        img[i, :, :][img[i, :, :] <= 0] = np.nan
        img[i, :, :][img[i, :, :] > 0] = 0

        nan_occulter[occulter_masks[i, :, :] > 0] = 1
        nan_mask[:, :][masks_infered[i, :, :] > 0] = 2

        ax[i].imshow(img[i, :, :], vmin=0, vmax=len(color) - 1, cmap=cmap)
        ax[i].imshow(nan_mask, cmap=cmap, alpha=0.6, vmin=0, vmax=len(color) - 1)
        ax[i].imshow(nan_occulter, cmap=cmap, alpha=0.25, vmin=0, vmax=len(color) - 1)

    masks_dir = os.path.join(opath, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    plt.savefig(os.path.join(masks_dir, namefile))
    plt.close()

def run_training():
    train_losses_per_batch = []
    mean_train_losses_per_batch = []
    test_losses_per_batch = []
    mean_test_error_in_batch = []
    epoch_list = []
    total_batches_per_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        train_onlyepoch_losses = []
        test_onlyepoch_losses = []
        epoch_list.append(total_batches_per_epoch)  # Store the total number of batches processed
        total_batches_per_epoch = 0
        for i, (img, targets, sat_masks, occulter_masks, satpos, plotranges, idx) in enumerate(cme_train_dataloader, 0):
            total_batches_per_epoch += 1
            loss_value = model.optimize_model(img, targets, optimizer, PAR_LOSS_WEIGHTS)
            train_losses_per_batch.append(loss_value.detach().cpu())
            train_onlyepoch_losses.append(loss_value.detach().cpu())

            if i % 10 == 0:            
                print(f'Epoch: {epoch + 1}, Image: {(i + 1) * BATCH_SIZE}, Batch: {i + 1}, Loss: {loss_value:.5f}, learning rate: {optimizer.param_groups[-1]["lr"]:.7f}')

            if i % 50 == 0:
                model.plot_loss(train_losses_per_batch, epoch_list, BATCH_SIZE, os.path.join(OPATH, "train_loss.png"), plot_epoch=False)

            if i == BATCH_LIMIT:
                break
        mean_train_losses_per_batch.append(np.mean(train_onlyepoch_losses))

        # Test
        model.eval()
        with torch.no_grad():
            for i, (img, targets, sat_masks, occulter_masks, satpos, plotranges, idx) in enumerate(cme_test_dataloader, 0):
                loss_test = model.test_model(img, targets, PAR_LOSS_WEIGHTS)
                test_losses_per_batch.append(loss_test.detach().cpu())
                test_onlyepoch_losses.append(loss_test.detach().cpu())
            mean_test_error_in_batch.append(np.mean(test_onlyepoch_losses))

        print(f'Epoch: {epoch + 1}, Test Loss: {loss_test:.5f}\n')
        model.plot_loss(test_losses_per_batch, epoch_list, BATCH_SIZE, os.path.join(OPATH, "test_loss.png"), plot_epoch=False)
        
        # Plot mean loss per epoch
        model.plot_loss(mean_train_losses_per_batch, epoch_list, BATCH_SIZE, os.path.join(OPATH, "mean_train_loss.png"), plot_epoch=False, meanLoss=True)
        model.plot_loss(mean_test_error_in_batch, epoch_list, BATCH_SIZE, os.path.join(OPATH, "mean_test_loss.png"), plot_epoch=False, meanLoss=True)

        # Save model
        if SAVE_MODEL:
            status = model.save_model(OPATH)
            print(f"Model saved at: {status}\n")

if __name__ == '__main__':
    dataset = Cme_MVP_Dataset(root_dir=TRAINDIR, img_size=IMG_SiZE, binary_mask=True)
    random.seed(SEED)
    total_samples = len(dataset)
    train_size = TRAIN_IDX_SIZE
    train_indices = random.sample(range(total_samples), train_size)
    test_indices = list(set(range(total_samples)) - set(train_indices))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    cme_train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    cme_test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Sirats_net(device=DEVICE, output_size=6, img_shape=IMG_SiZE)
    if LOAD_MODEL:
        status = model.load_model(OPATH)
        if status:
            print(f"Model loaded from: {status}\n")
        else:
            print(f"No model found at: {OPATH}, starting from scratch\n")

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_parameters}\n')

    #optimizer = torch.optim.Adam(model.parameters(), lr=LR[0])
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (len(cme_train_dataloader) / BATCH_SIZE) * EPOCHS, eta_min=LR[1])
    optimizer = torch.optim.Adadelta(model.parameters())
    scheduler = None
    loss_fn = torch.nn.MSELoss()

    if not INFERENCE_MODE:
        run_training()
    else:
        data_iter = iter(cme_test_dataloader)
        for i in range(100):
            img, targets, sat_masks, occulter_masks, satpos, plotranges, idx = next(data_iter)
            img, targets, sat_masks, occulter_masks, satpos, plotranges, idx = img[0], targets[0], sat_masks[0], occulter_masks[0], satpos[0], plotranges[0], idx[0]
            img = img.to(DEVICE)
            predictions = model.infer(img)
            plot_mask_MVP(img, sat_masks, targets, predictions, occulter_masks, satpos, plotranges, opath=OPATH, namefile=f'targetVinfered_{idx}.png')
        #test_specific_image()
