import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torchvision
import torch
from sirats_model import SiratsInception
from nn.utils.gcs_mask_generator import maskFromCloud
from pyGCS_raytrace import pyGCS
from nn.utils.coord_transformation import pnt2arr
from astropy.io import fits
from pathlib import Path
mpl.use('Agg')


class SiratsPlotter:
    def __init__(self) -> None:
        pass

    def calculate_non_overlapping_area(self, real_mask, infered_mask):
        # Combine masks to identify overlapping areas
        union = np.sum(np.logical_or(real_mask, infered_mask))
        symetric_diff = np.sum(np.logical_xor(real_mask, infered_mask))
        non_overlapping_area_err = symetric_diff / union if union > 0 else 0
        return non_overlapping_area_err

    def plot_overlap_err_histogram(self, errors, opath, namefile):
        fig, ax = plt.subplots(1, 4, figsize=(20, 11))
        errors = np.array(errors)
        allvp_errors = np.mean(errors, axis=0)
        ax[0].hist(allvp_errors, bins=30)
        ax[0].set_yscale('log')
        ax[0].set_title(
            f'AllVPs, mean: {np.around(np.mean(allvp_errors), 2)}, std: {np.around(np.std(allvp_errors), 2)}')
        ax[0].set_xlabel('Non-overlapping area error')
        ax[0].set_ylabel('Frequency')
        
        for i in range(errors.shape[0]):
            ax[i+1].hist(errors[i], bins=30)
            ax[i+1].set_yscale('log')
            ax[i+1].set_title(
                f'VP{i+1}, mean: {np.around(np.mean(errors[i]), 2)}, std: {np.around(np.std(errors[i]), 2)}')
            ax[i+1].set_xlabel('Non-overlapping area error')
            ax[i+1].set_ylabel('Frequency')


        plt.subplots_adjust(wspace=0.4)  # increment the space between subplots

        masks_dir = os.path.join(opath, 'infered_synth_imgs')

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
            ax[1][i].imshow(img[i, :, :], cmap="gray", vmin=0, vmax=1)
            # replace 0 for np.nan in arr_cloud
            arr_cloud[arr_cloud == 0] = np.nan
            arr_cloud[arr_cloud == 1] = img[i, :, :].max() + 1
            ax[1][i].imshow(arr_cloud, cmap='Greens', alpha=0.6,
                            vmin=0, vmax=1, interpolation='nearest')

        masks_dir = os.path.join(opath, 'infered_synth_imgs')
        os.makedirs(masks_dir, exist_ok=True)
        plt.savefig(os.path.join(masks_dir, namefile), dpi=300)
        plt.close()

        return error

    def plot_real_infer(self, imgs, prediction, satpos, plotranges, opath, namefile, fixed_satpos=None, fixed_plotranges=None, use_fixed=False):
        """
        Plot the real and inferred images with overlaid cloud contours.

        Args:
            imgs (torch.Tensor): The input images.
            prediction (torch.Tensor): The predicted cloud parameters.
            satpos (np.ndarray): The satellite positions.
            plotranges (np.ndarray): The plot ranges.
            opath (str): The output path for saving the plot.
            namefile (str): The name of the output file.
            fixed_satpos (np.ndarray, optional): The fixed satellite positions. Defaults to None.
            fixed_plotranges (np.ndarray, optional): The fixed plot ranges. Defaults to None.
            use_fixed (bool, optional): Flag indicating whether to use fixed satellite positions and plot ranges. Defaults to False.
        """

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

        suptitle = f'ima satpos: {", ".join([f"{x:.2f}" for x in satpos[0, :]])} -- fixed_satpos: {", ".join([f"{x:.2f}" for x in fixed_satpos[0, :]])} -- plotranges: {
            ", ".join([f"{x:.2f}" for x in plotranges[0, :]])} -- fixed_plotranges: {", ".join([f"{x:.2f}" for x in fixed_plotranges[0, :]])}\n\n'
        suptitle += f'imb satpos: {", ".join([f"{x:.2f}" for x in satpos[1, :]])} -- fixed_satpos: {", ".join([f"{x:.2f}" for x in fixed_satpos[1, :]])} -- plotranges: {
            ", ".join([f"{x:.2f}" for x in plotranges[1, :]])} -- fixed_plotranges: {", ".join([f"{x:.2f}" for x in fixed_plotranges[1, :]])}\n\n'
        suptitle += f'lasco satpos: {", ".join([f"{x:.2f}" for x in satpos[2, :]])} -- fixed_satpos: {", ".join([f"{x:.2f}" for x in fixed_satpos[2, :]])} -- plotranges: {
            ", ".join([f"{x:.2f}" for x in plotranges[2, :]])} -- fixed_plotranges: {", ".join([f"{x:.2f}" for x in fixed_plotranges[2, :]])}\n\n'
        suptitle += f'Using fixed satpos and plotranges = {use_fixed}'
        suptitle += f'\n\nPrediction: {prediction}'

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

                # flip arr_cloud in x
                arr_cloud = np.flip(arr_cloud, axis=0)

                imgs_mean = np.mean(imgs[i, :, :])
                imgs_std = np.std(imgs[i, :, :])

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

                # flip arr_cloud in x
                arr_cloud = np.flip(arr_cloud, axis=0)
                
                cloud_cmap = mpl.colors.ListedColormap(['none', '#49704F'])


                imgs_mean = np.mean(imgs[i, :, :])
                imgs_std = np.std(imgs[i, :, :])

                ax[i].imshow(imgs[i, :, :], cmap="gray", vmin=0, vmax=1)

                ax[i].imshow(arr_cloud, cmap=cloud_cmap,
                             alpha=1, vmin=0, vmax=1)

        masks_dir = os.path.join(opath, 'real_img_infer')
        os.makedirs(masks_dir, exist_ok=True)
        plt.savefig(os.path.join(masks_dir, namefile), dpi=300)
        plt.close()

    def plot_triplet_images(self, img, opath, namefile):
        img = img.cpu().detach().numpy()
        fig, ax = plt.subplots(1, 3, figsize=(18, 8))
        for i in range(3):
            ax[i].imshow(img[i], cmap="gray", vmin=0, vmax=1)
            ax[i].set_title(f'VP {i+1}')
        masks_dir = os.path.join(opath, 'real_img_infer')
        os.makedirs(masks_dir, exist_ok=True)
        plt.savefig(os.path.join(masks_dir, namefile), dpi=300)
        plt.close()

    def plot_img_histogram(self, img, sd_range, opath, namefile):
        img = img.cpu().detach().numpy()
        mean = np.mean(img, axis=(1, 2))
        std = np.std(img, axis=(1, 2))

        fig, ax = plt.subplots(1, 3, figsize=(18, 8))
        for i in range(3):
            ax[i].hist(img[i].flatten(), bins=30)
            ax[i].set_title(
                f'mean: {mean[i]:.2f}, std: {std[i]:.2f}, sd_range: {sd_range}')
            ax[i].set_yscale('log')
            ax[i].set_xlabel('Pixel value')
            ax[i].set_ylabel('Frequency')

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

        img = torch.tensor(
            [sat1_data, sat2_data, sat1_data], dtype=torch.float32)

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


    def plot_params_error_histogram(self, targets: torch.Tensor, predictions: torch.Tensor, par_loss_weight: torch.Tensor, opath: Path, namefile: str):
        # Get target and predictions to cpu
        titles = ["Long", "Lat", "Tilt", "Height", "K", "Ang"]
        targets = targets.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        par_loss_weight = par_loss_weight.cpu().detach().numpy()
        # read shape 0 of predictions and add the same shape to par_loss_weight
        par_loss_weight = np.tile(par_loss_weight, (predictions.shape[0], 1))
        l1_err = predictions - targets
        mean_square_err = np.mean(par_loss_weight * (predictions - targets) ** 2, axis=0)
        mean = np.mean(l1_err, axis=0)
        std = np.std(l1_err, axis=0)
        std3 = 3 * std
        p1 = np.percentile(l1_err, 1, axis=0)
        p10 = np.percentile(l1_err, 10, axis=0)
        p90 = np.percentile(l1_err, 90, axis=0)
        p99 = np.percentile(l1_err, 99, axis=0)

        params_data_path = os.path.join(opath, 'histogram_data/params_histogram_data')
        os.makedirs(params_data_path, exist_ok=True)
        np.savez(os.path.join(params_data_path, "params_histogram_data"), l1_err=l1_err, mean=mean, std=std, std3=std3, mean_square_err=mean_square_err, p1=p1, p10=p10, p90=p90, p99=p99)

        # Generate histograms
        fig, ax = plt.subplots(1, 6, figsize=(23, 9))
        fig.tight_layout(pad=3.0) # Increase the padding between subplots
        plt.subplots_adjust(top=0.85)  # Adjust the top of the plot to give more space for titles

        for i in range(6):
            ax[i].hist(l1_err[:, i], bins=30)
            ax[i].set_title(f'{titles[i]}\nmean={mean[i]:.2f}, std={std[i]:.2f}\n3std={std3[i]:.2f}, MSE={mean_square_err[i]:.2f}\np1={p1[i]:.2f}, p10={p10[i]:.2f}\np90={p90[i]:.2f}, p99={p99[i]:.2f}', fontsize=10)
            ax[i].set_yscale('log')

        # Save the figure
        opath = os.path.join(opath, 'infered_synth_imgs')
        os.makedirs(opath, exist_ok=True)
        plt.savefig(os.path.join(opath, namefile), dpi=300)
        plt.close()
