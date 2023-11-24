import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
import numpy as np
import random
import pickle
import torchvision
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
# mpl.use('TkAgg')
from pathlib import Path
from pyGCS_raytrace import pyGCS
from astropy.io import fits
from torch.utils.data import DataLoader
from nn.neural_gcs.cme_mvp_dataset import Cme_MVP_Dataset
from nn.neural_gcs.sirats_model import Sirats_net, Sirats_inception
from nn.utils.gcs_mask_generator import maskFromCloud
from nn.neural_gcs.sirats_config import Configuration
from pyGCS_raytrace import pyGCS
from nn.utils.coord_transformation import pnt2arr


def test_specific_image(model, opath, img_size, binary_mask, device):
    sat1_path = Path("/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_exp_paper_filtered/GCS_20110317_filter_True/FMwLASCO201103173_0.fits")
    sat2_path = Path("/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_exp_paper_filtered/GCS_20110317_filter_True/FMwLASCO201103173_1.fits")

    resize = torchvision.transforms.Resize(img_size[1:3], torchvision.transforms.InterpolationMode.BILINEAR)

    sat1_data, sat1_h = fits.getdata(sat1_path, header=True)
    sat2_data, sat2_h = fits.getdata(sat2_path, header=True)
    headers = [sat1_h, sat2_h]

    satpos, plotranges = pyGCS.processHeaders(headers)
    satpos = np.array(satpos)
    plotranges = np.array(plotranges)

    img = torch.tensor([sat1_data, sat2_data, sat1_data], dtype=torch.float32)

    if not binary_mask:
        sd_range = 1
        m = torch.mean(img)
        sd = torch.std(img)
        img = (img - m + sd_range * sd) / (2 * sd_range * sd)
    else:
        img[img > 1] = 1
        img[img < 0] = 0

    img = resize(img)
    img = img.to(device)

    with torch.no_grad():
        predictions = model.infer(img).cpu().numpy()

    img = img.cpu().numpy()
    predictions = np.squeeze(predictions)

    mask_infered_sat1 = maskFromCloud(predictions, sat=0, satpos=[satpos[0, :]], imsize=img_size[1:3], plotranges=[plotranges[0, :]])
    mask_infered_sat2 = maskFromCloud(predictions, sat=0, satpos=[satpos[1, :]], imsize=img_size[1:3], plotranges=[plotranges[1, :]])
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
    masks_dir = os.path.join(opath, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    plt.savefig(os.path.join(masks_dir, 'test_image.png'))
    plt.close()

def calculate_non_overlapping_area(mask1, mask2):
    # Combine masks to identify overlapping areas
    non_overlapping_area_err = np.sum(np.abs(mask1 - mask2)) / np.sum(mask1)
    return non_overlapping_area_err

def plot_histogram(errors, opath, namefile):
    fig, ax = plt.subplots(1, 4, figsize=(14, 7))
    flatten_errors = [item for sublist in errors for item in sublist]
    ax[0].hist(flatten_errors, bins=30)
    ax[0].set_title(f'AllVPs, mean: {np.around(np.mean(flatten_errors), 2)}, std: {np.around(np.std(flatten_errors), 2)}')
    ax[1].hist(errors[0], bins=30)
    ax[1].set_title(f'VP1, mean: {np.around(np.mean(errors[0]), 2)}, std: {np.around(np.std(errors[0]), 2)}')
    ax[2].hist(errors[1], bins=30)
    ax[2].set_title(f'VP2, mean: {np.around(np.mean(errors[1]), 2)}, std: {np.around(np.std(errors[1]), 2)}')
    ax[3].hist(errors[2], bins=30)
    ax[3].set_title(f'VP3, mean: {np.around(np.mean(errors[2]), 2)}, std: {np.around(np.std(errors[2]), 2)}')

    masks_dir = os.path.join(opath, 'infered_masks')

    fig.savefig(os.path.join(masks_dir, namefile), dpi=300)
    plt.close(fig)

def plot_mask_MVP(img, sat_masks, target, prediction, occulter_masks, satpos, plotranges, opath, namefile):
    # Convert tensors to numpy arrays
    img = img.cpu().detach().numpy()
    sat_masks = sat_masks.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    prediction = np.squeeze(prediction.cpu().detach().numpy())
    satpos = satpos.cpu().detach().numpy()
    plotranges = plotranges.cpu().detach().numpy()
    occulter_masks = occulter_masks.cpu().detach().numpy()

    IMG_SIZE = (img.shape[0], img.shape[1], img.shape[2])  # Assuming you have IMG_SIZE defined

    fig, ax = plt.subplots(2, IMG_SIZE[0], figsize=(13, 10))
    fig.tight_layout()
    fig.suptitle(f'target: {np.around(target, 3)}\nPrediction: {np.around(prediction, 3)}')

    color = ['purple', 'k', 'r', 'b']
    cmap = mpl.colors.ListedColormap(color)

    error = []

    for i in range(IMG_SIZE[0]):
        mask_infered_sat = maskFromCloud(prediction, satpos=[satpos[i, :]], imsize=IMG_SIZE[1:3], plotranges=[plotranges[i, :]])
        masks_infered = np.zeros(IMG_SIZE)
        masks_infered[i, :, :] = mask_infered_sat
        sat_mask_for_err = maskFromCloud(target, satpos=[satpos[i, :]], imsize=IMG_SIZE[1:3], plotranges=[plotranges[i, :]])
        sat_masks_for_err = np.zeros(IMG_SIZE)
        sat_masks_for_err[i, :, :] = sat_mask_for_err

        param_clouds = []
        param_clouds += prediction.tolist()
        param_clouds.append([satpos[i, :]])
        clouds = pyGCS.getGCS(*param_clouds, nleg=50, ncirc=100, ncross=100)
        x, y = clouds[0, :, 1], clouds[0, :, 2]
        arr_cloud = pnt2arr(x, y, [plotranges[i, :]], IMG_SIZE[1:3], 0)

        nan_mask = np.full(IMG_SIZE[1:3], np.nan)
        nan_occulter = np.full(IMG_SIZE[1:3], np.nan)

        img[i, :, :][img[i, :, :] <= 0] = np.nan
        img[i, :, :][img[i, :, :] > 0] = 0
        sat_mask_for_err[sat_mask_for_err <= 0] = np.nan
        sat_mask_for_err[sat_mask_for_err > 0] = 0

        arr_cloud[arr_cloud <= 0] = 0
        arr_cloud[arr_cloud > 0] = 1
        arr_cloud = np.flip(arr_cloud, axis=0)

        nan_occulter[occulter_masks[i, :, :] > 0] = 1
        nan_mask[:, :][masks_infered[i, :, :] > 0] = 2

        non_overlapping_area = calculate_non_overlapping_area(sat_masks_for_err[i], masks_infered[i])
        
        error.append(non_overlapping_area)

        ax[0][i].imshow(sat_mask_for_err, vmin=0, vmax=len(color) - 1, cmap=cmap)
        ax[0][i].imshow(nan_mask, cmap=cmap, alpha=0.6, vmin=0, vmax=len(color) - 1)
        #ax[0][i].imshow(nan_occulter, cmap=cmap, alpha=0.25, vmin=0, vmax=len(color) - 1)
        ax[0][i].set_title(f'non-overlapping area: {np.around(non_overlapping_area, 3)}')

        ax[1][i].imshow(img[i, :, :], cmap="gray")
        ax[1][i].imshow(arr_cloud, cmap='Greens', alpha=0.6, vmin=0, vmax=1)

    masks_dir = os.path.join(opath, 'infered_masks')
    os.makedirs(masks_dir, exist_ok=True)
    plt.savefig(os.path.join(masks_dir, namefile), dpi=300)
    plt.close()

    return error

def run_training(model, cme_train_dataloader, cme_test_dataloader, batch_size, epochs, opath, par_loss_weights, save_model):
    train_losses_per_batch = []
    mean_train_losses_per_batch = []
    test_losses_per_batch = []
    mean_test_error_in_batch = []
    epoch_list = []
    total_batches_per_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_onlyepoch_losses = []
        test_onlyepoch_losses = []
        epoch_list.append(total_batches_per_epoch)  # Store the total number of batches processed
        total_batches_per_epoch = 0
        for i, (img, targets, sat_masks, occulter_masks, satpos, plotranges, idx) in enumerate(cme_train_dataloader, 0):
            total_batches_per_epoch += 1
            loss_value = model.optimize_model(img, targets, par_loss_weights)
            train_losses_per_batch.append(loss_value.detach().cpu())
            train_onlyepoch_losses.append(loss_value.detach().cpu())

            if i % 10 == 0:
                print(f'Epoch: {epoch + 1}, Image: {(i + 1) * batch_size}, Batch: {i + 1}, Loss: {loss_value:.5f}, learning rate: {model.optimizer.param_groups[-1]["lr"]:.7f}')

            if i % 50 == 0:
                model.plot_loss(train_losses_per_batch, epoch_list, batch_size, os.path.join(opath, "train_loss.png"), plot_epoch=False)

        mean_train_losses_per_batch.append(np.mean(train_onlyepoch_losses))

        # Test
        model.eval()
        with torch.no_grad():
            for i, (img, targets, sat_masks, occulter_masks, satpos, plotranges, idx) in enumerate(cme_test_dataloader, 0):
                loss_test = model.test_model(img, targets, par_loss_weights)
                test_losses_per_batch.append(loss_test.detach().cpu())
                test_onlyepoch_losses.append(loss_test.detach().cpu())
            mean_test_error_in_batch.append(np.mean(test_onlyepoch_losses))

        print(f'Epoch: {epoch + 1}, Test Loss: {loss_test:.5f}\n')
        model.plot_loss(test_losses_per_batch, epoch_list, batch_size, os.path.join(opath, "test_loss.png"), plot_epoch=False)

        # Plot mean loss per epoch
        model.plot_loss(mean_train_losses_per_batch, epoch_list, batch_size, os.path.join(opath, "mean_train_loss.png"), plot_epoch=False, meanLoss=True)
        model.plot_loss(mean_test_error_in_batch, epoch_list, batch_size, os.path.join(opath, "mean_test_loss.png"), plot_epoch=False, meanLoss=True)

        # Save model
        if save_model:
            status = model.save_model(opath)
            print(f"Model saved at: {status}\n")


def main():
    # Configuración de parámetros
    configuration = Configuration(Path("/gehme-gpu/projects/2020_gcs_with_ml/repo_mariano/2020_gcs_with_ml/nn/neural_gcs/sirats_config/sirats_inception_run1.ini"))

    TRAINDIR = configuration.train_dir
    OPATH = configuration.opath
    BINARY_MASK = configuration.binary_mask
    BATCH_SIZE = configuration.batch_size
    BATCH_LIMIT = configuration.batch_limit
    SEED = configuration.rnd_seed
    IMG_SIZE = configuration.img_size
    DEVICE = configuration.device
    DO_TRAINING = configuration.do_training
    DO_INFERENCE = configuration.do_inference
    IMAGES_TO_INFER = configuration.images_to_infer
    SAVE_MODEL = configuration.save_model
    LOAD_MODEL = configuration.load_model
    EPOCHS = configuration.epochs
    TRAIN_IDX_SIZE = configuration.train_index_size
    LR = configuration.lr
    PAR_RNG = configuration.par_rng
    PAR_LOSS_WEIGHTS = configuration.par_loss_weight
    os.makedirs(OPATH, exist_ok=True)


    # Cargar y procesar datos
    dataset = Cme_MVP_Dataset(root_dir=TRAINDIR,
                              img_size=IMG_SIZE,
                              binary_mask=True)
    random.seed(SEED)
    total_samples = len(dataset)
    train_size = TRAIN_IDX_SIZE
    train_indices = random.sample(range(total_samples), train_size)
    test_indices = list(set(range(total_samples)) - set(train_indices))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    cme_train_dataloader = DataLoader(train_dataset,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
    cme_test_dataloader = DataLoader(test_dataset,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True)

    # Configurar el modelo
    model = Sirats_inception(device=DEVICE,
                             output_size=6,
                             img_shape=IMG_SIZE,
                             loss_weights=PAR_LOSS_WEIGHTS)

    # Configurar optimizer, loss function y scheduler
    optimizer = torch.optim.Adadelta(model.parameters())
    scheduler = None
    loss_fn = None

    # Setear optimizer, loss function y scheduler al modelo
    model.set_optimizer(optimizer)
    model.set_loss_fn(loss_fn)
    model.set_scheduler(scheduler)

    # Cargar o inicializar el modelo
    if LOAD_MODEL:
        status = model.load_model(OPATH)
        if status:
            print(f"Model loaded from: {status}\n")
        else:
            print(f"No model found at: {OPATH}, starting from scratch\n")

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_parameters}\n')

    # Ejecutar entrenamiento o inferencia
    if DO_TRAINING:
        run_training(model, cme_train_dataloader, cme_test_dataloader, BATCH_SIZE, EPOCHS, OPATH, PAR_LOSS_WEIGHTS,
                     SAVE_MODEL)

    if DO_INFERENCE:
        errorVP1 = []
        errorVP2 = []
        errorVP3 = []
    
        img_counter = 0
        stop_flag = False
        for iteration, (img, targets, sat_masks, occulter_masks, satpos, plotranges, idx) in enumerate(cme_test_dataloader, 0):
            # img, targets, sat_masks, occulter_masks, satpos, plotranges, idx = img[0], targets[0], sat_masks[0], occulter_masks[0], satpos[0], plotranges[0], idx[0]
            img = img.to(DEVICE)
            predictions = model.infer(img)
            # I want to do for image in batch
            for i in range(BATCH_SIZE):
                img_counter += 1
                print(f"Plotting image {img_counter} of {IMAGES_TO_INFER}")
                error = plot_mask_MVP(img[i], sat_masks[i], targets[i], predictions[i], occulter_masks[i], satpos[i], plotranges[i], OPATH, f'img_{img_counter}.png')
                errorVP1.append(error[0])
                errorVP2.append(error[1])
                errorVP3.append(error[2])
                if img_counter == IMAGES_TO_INFER:
                    stop_flag = True
                    break
            if stop_flag:
                break
        errors = [errorVP1, errorVP2, errorVP3]
        print("Plotting histogram")
        plot_histogram(errors, OPATH, 'histogram.png')

        # Save errors in a pickle file
        with open(os.path.join(OPATH, 'errors.pkl'), 'wb') as f:
            pickle.dump(errors, f)
            f.close()

if __name__ == '__main__':
    main()
