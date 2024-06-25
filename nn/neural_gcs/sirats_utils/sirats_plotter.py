import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))
from pathlib import Path
from astropy.io import fits
from nn.utils.coord_transformation import pnt2arr
from pyGCS_raytrace import pyGCS
from nn.utils.gcs_mask_generator import maskFromCloud
import torch
import torchvision
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class SiratsPlotter:
    def __init__(self) -> None:
        pass

    def calculate_non_overlapping_area(self, mask1, mask2):
        # Combine masks to identify overlapping areas
        non_overlapping_area_err = np.sum(
            np.abs(mask1 - mask2)) / np.sum(mask1)
        return non_overlapping_area_err

    def plot_histogram(self, errors, opath, namefile):
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


    def plot_images(self, img, prediction, satpos, plotranges, opath, namefile):
        images = img.cpu().detach().numpy()
        images = np.squeeze(images)
        prediction = np.squeeze(prediction.cpu().detach().numpy())
        satpos = satpos.cpu().detach().numpy()
        plotranges = plotranges.cpu().detach().numpy()

        # flip images in y axis
        # images = np.flip(images, axis=1)

        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        fig.tight_layout()

        fig.suptitle(prediction)

        color = ['purple', 'k', 'r', 'b']
        cmap = mpl.colors.ListedColormap(color)

        IMG_SIZE = (images.shape[0], images.shape[1], images.shape[2])

        for i in range(IMG_SIZE[0]):
            param_clouds = []
            param_clouds += prediction.tolist()
            param_clouds.append([satpos[i, :]])
            clouds = pyGCS.getGCS(*param_clouds, nleg=50,
                                  ncirc=100, ncross=100)
            x, y = clouds[0, :, 1], clouds[0, :, 2]
            arr_cloud = pnt2arr(x, y, [plotranges[i, :]], images.shape[1:3], 0)

            # flip arr_cloud in y axis
            arr_cloud = np.flip(arr_cloud, axis=0)

            ax[i].imshow(images[i, :, :], cmap="gray", vmin=0, vmax=1)
            ax[i].imshow(arr_cloud, cmap='Greens', alpha=0.6,
                         vmin=0, vmax=1)
            
        opath = os.path.join(opath, 'plots')
            
        os.makedirs(opath, exist_ok=True)
        plt.savefig(os.path.join(opath, namefile), dpi=300)
        plt.close()

    def plot_mask_MVP(self, img, sat_masks, target, prediction, occulter_masks, satpos, plotranges, opath, namefile):
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

        fig, ax = plt.subplots(2, IMG_SIZE[0], figsize=(13, 12))
        fig.tight_layout()
        diff = ((target/prediction) - 1) * 100
        fig.suptitle(
            f'target: {np.around(target, 3)}\nPrediction: {np.around(prediction, 3)}\ndiff(%){np.around(diff, 3)}')

        color = ['purple', 'k', 'r', 'b', 'g']
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
            clouds = pyGCS.getGCS(*param_clouds, nleg=50,
                                  ncirc=100, ncross=100)
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

            non_overlapping_area = self.calculate_non_overlapping_area(
                sat_masks_for_err[i], masks_infered[i])

            error.append(non_overlapping_area)
            ax[0][i].imshow(sat_mask_for_err, vmin=0,
                            vmax=len(color) - 1, cmap=cmap)
            ax[0][i].imshow(nan_mask, cmap=cmap, alpha=0.6,
                            vmin=0, vmax=len(color) - 1)
            # ax[0][i].imshow(nan_occulter, cmap=cmap, alpha=0.25, vmin=0, vmax=len(color) - 1)
            ax[0][i].set_title(
                f'non-overlapping area: {np.around(non_overlapping_area, 3)}')

            img_mean = np.mean(img[i, :, :])
            img_std = np.std(img[i, :, :])
            ax[1][i].imshow(img[i, :, :], cmap="gray", vmin=img_mean -
                            3 * img_std, vmax=img_mean + 3 * img_std)
            # replace 0 for np.nan in arr_cloud
            arr_cloud[arr_cloud == 0] = np.nan
            arr_cloud[arr_cloud == 1] = img[i, :, :].max() + 1
            ax[1][i].imshow(arr_cloud, cmap='Greens', alpha=0.6,
                            vmin=0, vmax=1, interpolation='nearest')

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

        suptitle = f'ima satpos: {", ".join([f"{x:.2f}" for x in satpos[0, :]])} -- fixed_satpos: {", ".join([f"{x:.2f}" for x in fixed_satpos[0, :]])} -- plotranges: {", ".join([f"{x:.2f}" for x in plotranges[0, :]])} -- fixed_plotranges: {", ".join([f"{x:.2f}" for x in fixed_plotranges[0, :]])}\n\n'
        suptitle += f'imb satpos: {", ".join([f"{x:.2f}" for x in satpos[1, :]])} -- fixed_satpos: {", ".join([f"{x:.2f}" for x in fixed_satpos[1, :]])} -- plotranges: {", ".join([f"{x:.2f}" for x in plotranges[1, :]])} -- fixed_plotranges: {", ".join([f"{x:.2f}" for x in fixed_plotranges[1, :]])}\n\n'
        suptitle += f'lasco satpos: {", ".join([f"{x:.2f}" for x in satpos[2, :]])} -- fixed_satpos: {", ".join([f"{x:.2f}" for x in fixed_satpos[2, :]])} -- plotranges: {", ".join([f"{x:.2f}" for x in plotranges[2, :]])} -- fixed_plotranges: {", ".join([f"{x:.2f}" for x in fixed_plotranges[2, :]])}\n\n'
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
                ax[i].imshow(arr_cloud, cmap='Greens',
                             alpha=0.6, vmin=0, vmax=1)
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
                ax[i].imshow(arr_cloud, cmap='Greens',
                             alpha=0.6, vmin=0, vmax=1)

        masks_dir = os.path.join(opath, 'real_img_infer')
        os.makedirs(masks_dir, exist_ok=True)
        plt.savefig(os.path.join(masks_dir, namefile), dpi=300)
        plt.close()

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
