import sys
import os
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
import numpy as np
import random
import pickle
import torchvision
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
# mpl.use('TkAgg')
from sirats_normalization import *
from pathlib import Path
from pyGCS_raytrace import pyGCS
from astropy.io import fits
from torch.utils.data import DataLoader
from nn.neural_gcs.cme_mvp_dataset import Cme_MVP_Dataset
from nn.neural_gcs.sirats_model import SiratsNet, SiratsInception, SiratsDistribution
from nn.utils.gcs_mask_generator import maskFromCloud
from nn.neural_gcs.sirats_config import Configuration
from pyGCS_raytrace import pyGCS
from nn.utils.coord_transformation import pnt2arr
from torchvision.io import read_image
from pyGCS_raytrace import pyGCS


def correct_path(s):
    s = s.replace("(1)", "")
    s = s.replace("(2)", "")
    return s


def convert_string(s, level):
    if level == 1:
        s = s.replace("preped/", "")
        s = s.replace("L0", "L1")
        s = s.replace("_0B", "_1B")
        s = s.replace("_04", "_14")
        s = s.replace("level1/", "")
    return s


def load_model(model: SiratsNet, model_folder: Path):
    model_path = os.path.join(model_folder, 'model.pth')
    os.makedirs(model_folder, exist_ok=True)  # Ensure directory exists
    if os.path.isfile(model_path):
        status = model.load_model(model_path)  # Load directly
        if status:
            copy_and_rename_existing_model(model_folder)
            logging.info(f"Model loaded from: {model_path}\n")
    else:
        logging.warning(
            f"No model found at: {model_path}, starting from scratch\n")


def copy_and_rename_existing_model(model_folder: Path):
    models_counter = len(os.listdir(model_folder))
    new_path = os.path.join(model_folder, f"model_run{models_counter}")
    os.system(f'cp {os.path.join(model_folder, "model.pth")} {new_path}')


def test_specific_image(model, opath, img_size, binary_mask, device):
    sat1_path = Path(
        "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_exp_paper_filtered/GCS_20110317_filter_True/FMwLASCO201103173_0.fits")
    sat2_path = Path(
        "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_exp_paper_filtered/GCS_20110317_filter_True/FMwLASCO201103173_1.fits")

    resize = torchvision.transforms.Resize(
        img_size[1:3], torchvision.transforms.InterpolationMode.BILINEAR)

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

    mask_infered_sat1 = maskFromCloud(predictions, sat=0, satpos=[
                                      satpos[0, :]], imsize=img_size[1:3], plotranges=[plotranges[0, :]])
    mask_infered_sat2 = maskFromCloud(predictions, sat=0, satpos=[
                                      satpos[1, :]], imsize=img_size[1:3], plotranges=[plotranges[1, :]])
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


def add_occulter(img, occulter_size, centerpix, repleace_value=None):
    '''
    Replace a circular area of radius occulter_size in input image[h,w,3] with a constant value
    repleace_value: if None, the area is replaced with the image mean. If set to scalar float that value is used
    '''
    if centerpix is not None:
        w = int(round(centerpix[0]))
        h = int(round(centerpix[1]))
    else:
        h, w = img.shape[:2]
        h = int(h/2)
        w = int(w/2)

    mask = np.zeros((img.shape[0], img.shape[0]), dtype=np.uint8)
    cv2.circle(mask, (w, h), occulter_size, 1, -1)

    if repleace_value is None:
        img[mask == 1] = np.mean(img)
    else:
        img[mask == 1] = repleace_value
    return img


def radius_to_px(plotranges, imsize, headers, sat):
    x = np.linspace(plotranges[0], plotranges[1], num=imsize[0])
    y = np.linspace(plotranges[2], plotranges[3], num=imsize[1])
    xx, yy = np.meshgrid(x, y)
    x_cS, y_cS = center_rSun_pixel(headers, plotranges, sat)
    return np.sqrt((xx - x_cS)**2 + (yy - y_cS)**2)


def center_rSun_pixel(headers, plotranges, sat):
    '''
    Gets the location of Suncenter in deg
    '''
    x_cS = (headers['CRPIX1']*plotranges[sat]*2) / \
        headers['NAXIS1'] - plotranges[sat]  # headers['CRPIX1']
    y_cS = (headers['CRPIX2']*plotranges[sat]*2) / \
        headers['NAXIS2'] - plotranges[sat]  # headers['CRPIX2']
    return x_cS, y_cS


def plot_histogram(errors, opath, namefile):
    fig, ax = plt.subplots(1, 4, figsize=(14, 7))
    flatten_errors = [item for sublist in errors for item in sublist]
    ax[0].hist(flatten_errors, bins=30)
    ax[0].set_title(
        f'AllVPs, mean: {np.around(np.mean(flatten_errors), 2)}, std: {np.around(np.std(flatten_errors), 2)}')
    ax[1].hist(errors[0], bins=30)
    ax[1].set_title(
        f'VP1, mean: {np.around(np.mean(errors[0]), 2)}, std: {np.around(np.std(errors[0]), 2)}')
    ax[2].hist(errors[1], bins=30)
    ax[2].set_title(
        f'VP2, mean: {np.around(np.mean(errors[1]), 2)}, std: {np.around(np.std(errors[1]), 2)}')
    ax[3].hist(errors[2], bins=30)
    ax[3].set_title(
        f'VP3, mean: {np.around(np.mean(errors[2]), 2)}, std: {np.around(np.std(errors[2]), 2)}')

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

    # Assuming you have IMG_SIZE defined
    IMG_SIZE = (img.shape[0], img.shape[1], img.shape[2])

    fig, ax = plt.subplots(2, IMG_SIZE[0], figsize=(13, 10))
    fig.tight_layout()
    fig.suptitle(
        f'target: {np.around(target, 3)}\nPrediction: {np.around(prediction, 3)}')

    color = ['purple', 'k', 'r', 'b']
    cmap = mpl.colors.ListedColormap(color)

    error = []
    for i in range(IMG_SIZE[0]):
        mask_infered_sat = maskFromCloud(prediction, satpos=[
                                         satpos[i, :]], imsize=IMG_SIZE[1:3], plotranges=[plotranges[i, :]])
        masks_infered = np.zeros(IMG_SIZE)
        masks_infered[i, :, :] = mask_infered_sat
        sat_mask_for_err = maskFromCloud(
            target, satpos=[satpos[i, :]], imsize=IMG_SIZE[1:3], plotranges=[plotranges[i, :]])
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

        sat_mask_for_err[sat_mask_for_err <= 0] = np.nan
        sat_mask_for_err[sat_mask_for_err > 0] = 0

        arr_cloud[arr_cloud <= 0] = 0
        arr_cloud[arr_cloud > 0] = 1
        arr_cloud = np.flip(arr_cloud, axis=0)

        nan_occulter[occulter_masks[i, :, :] > 0] = 1
        nan_mask[:, :][masks_infered[i, :, :] > 0] = 2

        non_overlapping_area = calculate_non_overlapping_area(
            sat_masks_for_err[i], masks_infered[i])

        error.append(non_overlapping_area)
        ax[0][i].imshow(sat_mask_for_err, vmin=0,
                        vmax=len(color) - 1, cmap=cmap)
        ax[0][i].imshow(nan_mask, cmap=cmap, alpha=0.6,
                        vmin=0, vmax=len(color) - 1)
        # ax[0][i].imshow(nan_occulter, cmap=cmap, alpha=0.25, vmin=0, vmax=len(color) - 1)
        ax[0][i].set_title(
            f'non-overlapping area: {np.around(non_overlapping_area, 3)}')

        ax[1][i].imshow(img[i, :, :], cmap="gray", vmin=0, vmax=1)
        ax[1][i].imshow(arr_cloud, cmap='Greens', alpha=0.6, vmin=0, vmax=1)

    masks_dir = os.path.join(opath, 'infered_masks')
    os.makedirs(masks_dir, exist_ok=True)
    plt.savefig(os.path.join(masks_dir, namefile), dpi=300)
    plt.close()

    return error


def plot_real_infer(imgs, prediction, satpos, plotranges, opath, namefile, fixed_satpos=None, fixed_plotranges=None, use_fixed=False):
    # Convert tensors to numpy arrays
    imgs = imgs.squeeze().cpu().detach().numpy()
    prediction = np.squeeze(prediction.cpu().detach().numpy())

    # Convert to numpy non fixed variables
    if type(satpos) is not np.ndarray:
        satpos = np.array(satpos, dtype=np.float32)
    if type(plotranges) is not np.ndarray:
        plotranges = np.array(plotranges, dtype=np.float32)

    # Convert to numpy fixed variables
    if use_fixed and fixed_satpos is not None and fixed_plotranges is not None:
        if type(fixed_satpos) is not np.ndarray:
            fixed_satpos = np.array(fixed_satpos, dtype=np.float32)
        if type(fixed_plotranges) is not np.ndarray:
            fixed_plotranges = np.array(fixed_plotranges, dtype=np.float32)

    # Assuming you have IMG_SIZE defined
    IMG_SIZE = (imgs.shape[0], imgs.shape[1], imgs.shape[2])

    fig, ax = plt.subplots(1, IMG_SIZE[0], figsize=(17, 10))
    fig.tight_layout()

    suptitle = f'ima satpos: {np.round(satpos[0, :], 1)} -- fixed_satpos: {np.round(fixed_satpos[0, :], 1)} -- plotranges: {np.round(plotranges[0, :], 1)} -- fixed_plotranges: {np.round(fixed_plotranges[0, :], 1)}\n\n'
    suptitle += f'imb satpos: {np.round(satpos[1, :], 1)} -- fixed_satpos: {np.round(fixed_satpos[1, :], 1)} -- plotranges: {np.round(plotranges[1, :], 1)} -- fixed_plotranges: {np.round(fixed_plotranges[1, :], 1)}\n\n'
    suptitle += f'lasco satpos: {np.round(satpos[2, :], 1)} -- fixed_satpos: {np.round(fixed_satpos[2, :], 1)} -- plotranges: {np.round(plotranges[2, :], 1)} -- fixed_plotranges: {np.round(fixed_plotranges[2, :], 1)}\n\n'
    suptitle += f'Using fixed satpos = {use_fixed}'
    fig.suptitle(suptitle, x=0.05, y=.95, horizontalalignment='left')

    color = ['purple', 'k', 'r', 'b']
    cmap = mpl.colors.ListedColormap(color)

    for i in range(IMG_SIZE[0]):
        param_clouds = prediction.tolist()
        if not use_fixed:
            param_clouds.append([satpos[i, :]])
            clouds = pyGCS.getGCS(*param_clouds, nleg=50,
                                  ncirc=100, ncross=100)
            x, y = clouds[0, :, 1], clouds[0, :, 2]
            arr_cloud = pnt2arr(x, y, [plotranges[i, :]], IMG_SIZE[1:3], 0)

            # Flip img
            imgs[i, :, :] = np.flip(imgs[i, :, :], axis=0)

            ax[i].imshow(imgs[i, :, :], cmap="gray", vmin=0, vmax=1)
            ax[i].imshow(arr_cloud, cmap='Greens', alpha=0.6, vmin=0, vmax=1)
        else:
            param_clouds.append([fixed_satpos[i, :]])
            clouds = pyGCS.getGCS(*param_clouds, nleg=50,
                                  ncirc=100, ncross=100)
            x, y = clouds[0, :, 1], clouds[0, :, 2]
            arr_cloud = pnt2arr(
                x, y, [fixed_plotranges[i, :]], IMG_SIZE[1:3], 0)

            # Flip img
            imgs[i, :, :] = np.flip(imgs[i, :, :], axis=0)

            ax[i].imshow(imgs[i, :, :], cmap="gray", vmin=0, vmax=1)
            ax[i].imshow(arr_cloud, cmap='Greens', alpha=0.6, vmin=0, vmax=1)

    masks_dir = os.path.join(opath, 'real_img_infer')
    os.makedirs(masks_dir, exist_ok=True)
    plt.savefig(os.path.join(masks_dir, namefile), dpi=300)
    plt.close()


def run_training(model, cme_train_dataloader, cme_test_dataloader, batch_size, epochs, opath, par_loss_weights, save_model):
    train_losses_per_batch = []
    median_train_losses_per_batch = []
    test_losses_per_batch = []
    median_test_error_in_batch = []
    epoch_list = []
    total_batches_per_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_onlyepoch_losses = []
        test_onlyepoch_losses = []
        # Store the total number of batches processed
        epoch_list.append(total_batches_per_epoch)
        total_batches_per_epoch = 0
        for i, (img, targets, sat_masks, occulter_masks, satpos, plotranges, idx) in enumerate(cme_train_dataloader, 0):
            total_batches_per_epoch += 1
            loss_value = model.optimize_model(img, targets, par_loss_weights)
            train_losses_per_batch.append(loss_value.detach().cpu())
            train_onlyepoch_losses.append(loss_value.detach().cpu())

            if i % 10 == 0:
                logging.info(
                    f'Epoch: {epoch + 1}, Image: {(i + 1) * batch_size}, Batch: {i + 1}, Loss: {loss_value:.5f}, learning rate: {model.optimizer.param_groups[-1]["lr"]:.7f}')

            if i % 50 == 0:
                model.plot_loss(train_losses_per_batch, epoch_list, batch_size, os.path.join(
                    opath, "train_loss.png"), plot_epoch=False)

        median_train_losses_per_batch.append(np.median(train_onlyepoch_losses))

        # Test
        model.eval()
        with torch.no_grad():
            for i, (img, targets, sat_masks, occulter_masks, satpos, plotranges, idx) in enumerate(cme_test_dataloader, 0):
                loss_test = model.test_model(img, targets, par_loss_weights)
                test_losses_per_batch.append(loss_test.detach().cpu())
                test_onlyepoch_losses.append(loss_test.detach().cpu())
            median_test_error_in_batch.append(np.median(test_onlyepoch_losses))

        logging.info(f'Epoch: {epoch + 1}, Test Loss: {loss_test:.5f}\n')
        model.plot_loss(test_losses_per_batch, epoch_list, batch_size, os.path.join(
            opath, "test_loss.png"), plot_epoch=False)

        # Plot mean loss per epoch
        model.plot_loss(median_train_losses_per_batch, epoch_list, batch_size, os.path.join(
            opath, "mean_train_loss.png"), plot_epoch=False, medianLoss=True)
        model.plot_loss(median_test_error_in_batch, epoch_list, batch_size, os.path.join(
            opath, "mean_test_loss.png"), plot_epoch=False, medianLoss=True)

        # Save model
        if save_model:
            status = model.save_model(opath)
            logging.info(f"Model saved at: {status}\n")


def get_paths_cme_exp_sources():
    """
    Read all files for selected events of the CME exp sources project
    """
    data_path = '/gehme/data'  # Path to the dir containing /sdo ,/soho and /stereo data directories as well as the /Polar_Observations dir.
    # Path with our GCS data directories
    gcs_path = '/gehme-gpu/projects/2020_gcs_with_ml/repo_mariano/2020_cme_expansion/GCSs'
    lasco_path = data_path+'/soho/lasco/level_1/c2'  # LASCO proc images Path
    secchipath = data_path+'/stereo/secchi/L0'
    level = 0  # set the reduction level of the images
    # events to read
    dates = ['20101212', '20101214', '20110317', '20110605', '20130123', '20130129',
             '20130209', '20130424', '20130502', '20130517', '20130527', '20130608']
    # pre event iamges per instrument
    pre_event = ["/soho/lasco/level_1/c2/20101212/25354377.fts",
                 "/stereo/secchi/L1/a/seq/cor1/20101212/20101212_022500_1B4c1A.fts",
                 "/stereo/secchi/L1/b/seq/cor1/20101212/20101212_023500_1B4c1B.fts",
                 "/stereo/secchi/L1/a/img/cor2/20101212/20101212_015400_14c2A.fts",
                 "/stereo/secchi/L1/b/img/cor2/20101212/20101212_015400_14c2B.fts",
                 "/soho/lasco/level_1/c2/20101214/25354679.fts",
                 "/stereo/secchi/L1/a/seq/cor1/20101214/20101214_150000_1B4c1A.fts",
                 "/stereo/secchi/L1/b/seq/cor1/20101214/20101214_150000_1B4c1B.fts",
                 "/stereo/secchi/L1/a/img/cor2/20101214/20101214_152400_14c2A.fts",
                 "/stereo/secchi/L1/b/img/cor2/20101214/20101214_153900_14c2B.fts",
                 "/soho/lasco/level_1/c2/20110317/25365446.fts",
                 "/stereo/secchi/L1/a/seq/cor1/20110317/20110317_103500_1B4c1A.fts",
                 "/stereo/secchi/L1/b/seq/cor1/20110317/20110317_103500_1B4c1B.fts",
                 "/stereo/secchi/L1/a/img/cor2/20110317/20110317_115400_14c2A.fts",
                 "/stereo/secchi/L1/b/img/cor2/20110317/20110317_123900_14c2B.fts",
                 "/soho/lasco/level_1/c2/20110605/25374823.fts",
                 "/stereo/secchi/L1/a/seq/cor1/20110605/20110605_021000_1B4c1A.fts",
                 "/stereo/secchi/L1/b/seq/cor1/20110605/20110605_021000_1B4c1B.fts",
                 "/stereo/secchi/L1/a/img/cor2/20110605/20110605_043900_14c2A.fts",
                 "/stereo/secchi/L1/b/img/cor2/20110605/20110605_043900_14c2B.fts",
                 "/soho/lasco/level_1/c2/20130123/25445617.fts",
                 "/stereo/secchi/L1/a/seq/cor1/20130123/20130123_131500_1B4c1A.fts",
                 "/stereo/secchi/L1/b/seq/cor1/20130123/20130123_125500_1B4c1B.fts",
                 "/stereo/secchi/L1/a/img/cor2/20130123/20130123_135400_14c2A.fts",
                 "/stereo/secchi/L1/b/img/cor2/20130123/20130123_142400_14c2B.fts",
                 "/soho/lasco/level_1/c2/20130129/25446296.fts",
                 "/stereo/secchi/L1/a/seq/cor1/20130129/20130129_012500_1B4c1A.fts",
                 "/stereo/secchi/L1/b/seq/cor1/20130129/20130129_012500_1B4c1B.fts",
                 "/stereo/secchi/L1/a/img/cor2/20130129/20130129_015400_14c2A.fts",
                 "/stereo/secchi/L1/b/img/cor2/20130129/20130129_015400_14c2B.fts",
                 "/soho/lasco/level_1/c2/20130209/25447666.fts",
                 "/stereo/secchi/L1/a/seq/cor1/20130209/20130209_054000_1B4c1A.fts",
                 "/stereo/secchi/L1/b/seq/cor1/20130209/20130209_054500_1B4c1B.fts",
                 "/stereo/secchi/L1/a/img/cor2/20130209/20130209_062400_14c2A.fts",
                 "/stereo/secchi/L1/b/img/cor2/20130209/20130209_062400_14c2B.fts",
                 "/soho/lasco/level_1/c2/20130424_1/25456651.fts",
                 "/stereo/secchi/L1/a/seq/cor1/20130424/20130424_051500_1B4c1A.fts",
                 "/stereo/secchi/L1/b/seq/cor1/20130424/20130424_051500_1B4c1B.fts",
                 "/stereo/secchi/L1/a/img/cor2/20130424/20130424_055400_14c2A.fts",
                 "/stereo/secchi/L1/b/img/cor2/20130424/20130424_065400_14c2B.fts",
                 "/soho/lasco/level_1/c2/20130502/25457629.fts",
                 "/stereo/secchi/L1/a/seq/cor1/20130502/20130502_045000_1B4c1A.fts",
                 "/stereo/secchi/L1/b/seq/cor1/20130502/20130502_050000_1B4c1B.fts",
                 "/stereo/secchi/L1/a/img/cor2/20130502/20130502_012400_14c2A.fts",
                 "/stereo/secchi/L1/b/img/cor2/20130502/20130502_053900_14c2B.fts"
                 "/soho/lasco/level_1/c2/20130517/25459559.fts",
                 "/stereo/secchi/L1/a/seq/cor1/20130517/20130517_194500_1B4c1A.fts",
                 "/stereo/secchi/L1/b/seq/cor1/20130517/20130517_194500_1B4c1B.fts",
                 "/stereo/secchi/L1/a/img/cor2/20130517/20130517_203900_14c2A.fts",
                 "/stereo/secchi/L1/b/img/cor2/20130517/20130517_205400_14c2B.fts",
                 "/soho/lasco/level_1/c2/20130527_2/25460786.fts",
                 "/stereo/secchi/L1/a/seq/cor1/20130527/20130527_183000_1B4c1A.fts",
                 "/stereo/secchi/L1/b/seq/cor1/20130527/20130527_183000_1B4c1B.fts",
                 "/stereo/secchi/L1/a/img/cor2/20130527/20130527_192400_14c2A.fts",
                 "/stereo/secchi/L1/b/img/cor2/20130527/20130527_195400_14c2B.fts",
                 "/soho/lasco/level_1/c2/20130608/25462149.fts",
                 "/stereo/secchi/L1/a/seq/cor1/20130607/20130607_221500_1B4c1A.fts",
                 "/stereo/secchi/L1/b/seq/cor1/20130607/20130607_223000_1B4c1B.fts",
                 "/stereo/secchi/L1/a/img/cor2/20130607/20130607_225400_14c2A.fts",
                 "/stereo/secchi/L1/b/img/cor2/20130607/20130607_232400_14c2B.fts"]
    pre_event = [data_path + f for f in pre_event]

    # get file event for each event
    temp = os.listdir(gcs_path)
    events_path = [os.path.join(gcs_path, d)
                   for d in temp if str.split(d, '_')[-1] in dates]

    # gets .savs, andthe cor and lasco file event for each time instant in each event
    event = []
    for ev in events_path:
        cdict = {'date': [], 'pro_files': [], 'sav_files': [], 'ima1': [], 'ima0': [], 'imb1': [
        ], 'imb0': [], 'lasco1': [], 'lasco0': [], 'pre_ima': [], 'pre_imb': [], 'pre_lasco': []}
        tinst = os.listdir(ev)
        sav_files = sorted([os.path.join(ev, f)
                           for f in tinst if f.endswith('.sav')])
        pro_files = sorted([os.path.join(ev, f) for f in tinst if (f.endswith(
            '.pro') and 'fit_' not in f and 'tevo_' not in f and 'm1.' not in f)])
        if len(sav_files) != len(pro_files):
            os.error('ERROR. Found different number of .sav and .pro files')
            sys.exit
        # reads the lasco and stereo files from within each pro
        ok_pro_files = []
        ok_sav_files = []
        for f in pro_files:
            with open(f) as of:
                for line in of:
                    if 'ima=sccreadfits(' in line:
                        cline = secchipath + line.split('\'')[1]
                        if 'cor1' in cline:
                            cor = 'cor1'
                        if 'cor2' in cline:
                            cor = 'cor2'
                        cdate = cline[cline.find(
                            '/preped/')+8:cline.find('/preped/')+16]
                        cline = convert_string(cline, level)
                        cline = correct_path(cline)
                        cdict['ima1'].append(cline)
                        ok_pro_files.append(f)
                        cpre = [s for s in pre_event if (
                            cdate in s and cor in s and '/a/' in s)]
                        if len(cpre) == 0:
                            print(
                                f'Cloud not find pre event image for {cdate}')
                            breakpoint()
                        cdict['pre_ima'].append(cpre[0])
                        cdict['pre_imb'].append([s for s in pre_event if (
                            cdate in s and cor in s and '/a/' in s)][0])
                    if 'imaprev=sccreadfits(' in line:
                        cline = convert_string(
                            secchipath + line.split('\'')[1], level)
                        cline = correct_path(cline)
                        cdict['ima0'].append(cline)
                    if 'imb=sccreadfits(' in line:
                        cline = convert_string(
                            secchipath + line.split('\'')[1], level)
                        cline = correct_path(cline)
                        cdict['imb1'].append(cline)
                    if 'imbprev=sccreadfits(' in line:
                        cline = convert_string(
                            secchipath + line.split('\'')[1], level)
                        cline = correct_path(cline)
                        cdict['imb0'].append(cline)
                    if 'lasco1=readfits' in line:
                        cline = lasco_path + line.split('\'')[1]
                        cline = correct_path(cline)
                        cdict['lasco1'].append(cline)
                        cdate = cline[cline.find(
                            '/preped/')+8:cline.find('/preped/')+16]
                        cpre = [s for s in pre_event if (
                            cdate in s and '/c2/' in s)]
                        if len(cpre) == 0:
                            print(
                                f'Cloud not find pre event image for {cdate}')
                            breakpoint()
                        cdict['pre_lasco'].append(cpre[0])
                    if 'lasco0=readfits' in line:
                        cline = lasco_path + line.split('\'')[1]
                        cline = correct_path(cline)
                        cdict['lasco0'].append(cline)
        cdict['date'] = ev
        cdict['pro_files'] = ok_pro_files
        cdict['sav_files'] = ok_sav_files
        event.append(cdict)
    return event


def main():
    # Configuración de parámetros
    configuration = Configuration(Path(
        "/gehme-gpu/projects/2020_gcs_with_ml/repo_mariano/2020_gcs_with_ml/nn/neural_gcs/sirats_config/sirats_inception_run4.ini"))

    TRAINDIR = configuration.train_dir
    OPATH = configuration.opath
    BATCH_SIZE = configuration.batch_size
    BATCH_LIMIT = configuration.batch_limit
    SEED = configuration.rnd_seed
    IMG_SIZE = configuration.img_size
    DEVICE = configuration.device
    ONLY_MASK = configuration.only_mask
    DO_TRAINING = configuration.do_training
    DO_INFERENCE = configuration.do_inference
    REAL_IMG_INFERENCE = configuration.real_img_inference
    IMAGES_TO_INFER = configuration.images_to_infer
    MODEL_ARQ = configuration.model_arq
    SAVE_MODEL = configuration.save_model
    LOAD_MODEL = configuration.load_model
    EPOCHS = configuration.epochs
    TRAIN_IDX_SIZE = configuration.train_index_size
    LR = configuration.lr
    PAR_RNG = configuration.par_rng
    PAR_LOSS_WEIGHTS = configuration.par_loss_weight
    os.makedirs(OPATH, exist_ok=True)

    # Logging configuration
    LOGF_PATH = os.path.join(OPATH, 'sirats.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(funcName)-5s: %(levelname)-s, %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=LOGF_PATH,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(
        '%(asctime)s  %(funcName)-5s: %(levelname)-s, %(message)s', datefmt='%m-%d %H:%M')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Cargar y procesar datos
    dataset = Cme_MVP_Dataset(root_dir=TRAINDIR,
                              img_size=IMG_SIZE,
                              only_mask=ONLY_MASK)
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
    if MODEL_ARQ == 'inception':
        model = SiratsInception(device=DEVICE,
                                output_size=6,
                                img_shape=IMG_SIZE,
                                loss_weights=PAR_LOSS_WEIGHTS)
    elif MODEL_ARQ == 'distribution':
        model = SiratsDistribution(device=DEVICE,
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
        load_model(model, os.path.join(OPATH, 'models'))

    num_parameters = sum(p.numel() for p in model.parameters())
    logging.info(f'Number of parameters: {num_parameters}\n')

    # Ejecutar entrenamiento o inferencia
    if DO_TRAINING:
        run_training(model, cme_train_dataloader, cme_test_dataloader, BATCH_SIZE, EPOCHS, OPATH, PAR_LOSS_WEIGHTS,
                     SAVE_MODEL)

    if DO_INFERENCE:
        errorVP1 = []
        errorVP2 = []
        errorVP3 = []

        if not REAL_IMG_INFERENCE:
            img_counter = 0
            stop_flag = False
            for iteration, (img, targets, sat_masks, occulter_masks, satpos, plotranges, idx) in enumerate(cme_test_dataloader, 0):
                img = img.to(DEVICE)
                predictions = model.infer(img)

                if MODEL_ARQ == 'distribution':
                    predictions = predictions.mean()

                for i in range(BATCH_SIZE):
                    img_counter += 1
                    logging.info(
                        f"Plotting image {img_counter} of {IMAGES_TO_INFER}")
                    error = plot_mask_MVP(img[i], sat_masks[i], targets[i], predictions[i],
                                          occulter_masks[i], satpos[i], plotranges[i], OPATH, f'img_{img_counter}.png')
                    errorVP1.append(error[0])
                    errorVP2.append(error[1])
                    errorVP3.append(error[2])
                    if img_counter == IMAGES_TO_INFER:
                        stop_flag = True
                        break
                if stop_flag:
                    break

            errors = [errorVP1, errorVP2, errorVP3]
            logging.info("Plotting histogram")
            plot_histogram(errors, OPATH, 'histogram.png')

            # Save errors in a pickle file
            with open(os.path.join(OPATH, 'errors.pkl'), 'wb') as f:
                pickle.dump(errors, f)
                f.close()

        else:
            # Get events
            events = get_paths_cme_exp_sources()
            img_counter = 0
            for ev in events:
                # Get event images
                ima = fits.getdata(ev['ima1'][1]) - fits.getdata(ev['ima0'][1])
                imb = fits.getdata(ev['imb1'][1]) - fits.getdata(ev['imb0'][1])
                lasco = fits.getdata(ev['lasco1'][1]) - \
                    fits.getdata(ev['lasco0'][1])

                # Get event headers
                event_headers = []
                # event_headers.append(fits.getheader(ev['ima0'][0]))
                event_headers.append(fits.getheader(ev['imb1'][0]))
                # event_headers.append(fits.getheader(ev['imb0'][0]))
                event_headers.append(fits.getheader(ev['ima1'][0]))
                # event_headers.append(fits.getheader(ev['lasco0'][0]))
                event_headers.append(fits.getheader(ev['lasco1'][0]))

                satpos, plotranges = pyGCS.processHeaders(event_headers)

                # synth_img_path = '/gehme-gpu/projects/2020_gcs_with_ml/data/gcs_ml_3VP_size_100000_seed_72430'
                # synth_img_list = os.listdir(synth_img_path)
                # random_img = random.choice(synth_img_list)
                # synth_img_list = os.listdir(os.path.join(synth_img_path, random_img))
                # for img in synth_img_list:
                #     if img.find('sat1') != -1:
                #         ima = read_image(os.path.join(synth_img_path, random_img, img), mode=torchvision.io.image.ImageReadMode.GRAY)
                #         ima = ima.squeeze(0)
                #     if img.find('sat2') != -1:
                #         imb = read_image(os.path.join(synth_img_path, random_img, img), mode=torchvision.io.image.ImageReadMode.GRAY)
                #         imb = imb.squeeze(0)
                #     if img.find('sat3') != -1:
                #         lasco = read_image(os.path.join(synth_img_path, random_img, img), mode=torchvision.io.image.ImageReadMode.GRAY)
                #         lasco = lasco.squeeze(0)

                event_list = [imb, ima, lasco]

                # make events as tensors
                event_list = [torch.tensor(ev, dtype=torch.float32)
                              for ev in event_list]

                # Add occulter to images
                center_idxs = [
                    ((ev.shape[0] - 1) // 2, (ev.shape[1] - 1) // 2) for ev in event_list]
                # Occulters size for [sat1, sat2 ,sat3] in [Rsun]
                occulter_size = [2., 2., 4.3]
                # occ_center=[(30,-15),(0,-5),(0,0)] # [(38,-15),(0,-5),(0,0)] # (y,x)
                r_values = [radius_to_px(
                    plotranges[i], event_list[i].shape, event_headers[i], i) for i in range(len(occulter_size))]
                # event_list = [add_occulter(ev, occulter_size[i], center_idxs[i]) for i, ev in enumerate(event_list)]
                for i in range(len(event_list)):
                    ev = event_list[i]
                    ev[r_values[i] <= occulter_size[i]/2] = 0
                    event_list[i] = ev

                # Normalize event images
                event_list = [real_img_normalization(ev) for ev in event_list]

                # Resize event images
                resize = torchvision.transforms.Resize(
                    IMG_SIZE[1:3], torchvision.transforms.InterpolationMode.BILINEAR)
                resize_scale_factor = [
                    event_list[i].shape[1] / IMG_SIZE[1] for i in range(len(event_list))]
                event_list = [resize(ev.unsqueeze(0)) for ev in event_list]
                event_list = [ev.squeeze(0) for ev in event_list]
                for i in range(len(event_headers)):
                    h = event_headers[i]
                    h['CDELT1'] = resize_scale_factor[i] * h['CDELT1']
                    h['CDELT2'] = resize_scale_factor[i] * h['CDELT2']
                    event_headers[i] = h

                # join event images
                event_img = torch.stack(event_list, dim=0)

                # Add batch dimension
                event_img = event_img.unsqueeze(0)

                # Move event images to device
                event_img = event_img.to(DEVICE)

                # Infer event images and save losses
                predictions = model.infer(event_img)
                fixed_satpos = "[[32.8937181611, 7.05123478188, 0.0], [300.081940747, 1.95463511752, 0.0], [274.2910293847356, -4.817368115630504, 0.0]]"
                fixed_plotranges = "[[-16.6431925965469, 16.737728407985518, -16.84856349725838, 16.53235750727404], [-15.00659312775248, 15.050622251843686, -14.988981478115997, 15.068233901480168], [-6.338799715536909, 6.304081179329522, -6.388457593426707, 6.254423301439724]]"

                fixed_satpos = torch.tensor(
                    eval(fixed_satpos), dtype=torch.float32)
                fixed_plotranges = torch.tensor(
                    eval(fixed_plotranges), dtype=torch.float32)

                # Plot infered masks
                plot_real_infer(event_img, predictions, satpos, plotranges, OPATH,
                                f'img_{img_counter}.png', fixed_satpos=fixed_satpos, fixed_plotranges=fixed_plotranges, use_fixed=False)
                img_counter += 1


if __name__ == '__main__':
    main()
