import torch
import matplotlib.pyplot as plt
import cv2
from scipy import stats
import numpy as np


def binary_mask_normalization(img: torch.Tensor):
    img[img > 1] = 1
    img[img < 0] = 0
    return img

def real_img_normalization(img: torch.Tensor, excl_occulter_level = None):
    """
    Normalize the input image tensor using Sirats normalization method.

    Args:
        img (torch.Tensor): The input image tensor. Should have shape [3, x, y].
        excl_occulter_level (int or str, optional): The value to exclude from normalization. 
            If 'auto', the most frequent integer value in the image will be excluded. 
            If an integer value is provided, that value and its adjacent value will be excluded. 
            Defaults to None.

    Returns:
        torch.Tensor: The normalized image tensor.

    """
    sd_range=1.5
    #exclude occulter values at excl_occulter_level
    for i in range(3):
        if excl_occulter_level is not None:
            if excl_occulter_level == 'auto':
                # finds the most frequent integer value in the image
                occ_indx = (img[i] != stats.mode(img[i].flatten(),keepdims=True)[0][0])
            else:
                occ_indx = (img[i] != excl_occulter_level) & (img[i] != excl_occulter_level-1)
            m = np.nanmean(img[i][occ_indx])
            sd = np.nanstd(img[i][occ_indx])
        else:
            m = np.nanmean(img[i])
            sd = np.nanstd(img[i])
        img[i] = (img[i] - m + sd_range * sd) / (2 * sd_range * sd)
        img[i][img[i] >1]=1
        img[i][img[i] <0]=0
    return img

def normalize(self, image, excl_occulter_level='auto', sd_range=2, norm_limits=[None, None], increase_contrast=False, 
                  median_kernel=3, plot_histograms=False, histogram_names='', path=''):
        '''
        Normalizes the input image to the 0-1 range. 
        Note that only the first channel of the image is used!
        
        sd_range: number of standard deviations around the mean to use for normalization 
        norm_limits: if not None, the image is first truncated to the given limits
        increase_contrast: if True, increases the contrast of the normalized image radially.
        plot_histograms: if True, plots the histograms of the original and normalized images
        histogram_names: name of the histogram to save 
        path: dir path to save the histograms
        excl_occulter_level: if not None, this level is excluded from the mean and sd computation used to normalize
                             Use 'auto' to exclude the most frequent value in the image
        median_kernel: if not None, the image is smoothed using a median kernel of the given size. Min is 3
        '''
        # pre clipping
        if norm_limits[0] is not None:
            image[image < norm_limits[0]] = 0
        if norm_limits[1] is not None:
            image[image > norm_limits[1]] = 1
        
        #using only first channel
        oimage = image[:,:,0].copy()

        #median kernel
        if median_kernel is not None:
            oimage = cv2.medianBlur(oimage, median_kernel)

        #exclude occulter values at excl_occulter_level
        if excl_occulter_level is not None:
            if excl_occulter_level == 'auto':
                # finds the most frequent integer value in the image
                occ_indx = (oimage != stats.mode(oimage.flatten(),keepdims=True)[0][0])
            else:
                occ_indx = (oimage != excl_occulter_level) & (oimage != excl_occulter_level-1)
            m = np.nanmean(oimage[occ_indx])
            sd = np.nanstd(oimage[occ_indx])
        else:
            m = np.nanmean(oimage)
            sd = np.nanstd(oimage)
        
        if plot_histograms:
            plt.hist(oimage, bins=50, range=(np.percentile(oimage,1),np.percentile(oimage,99)), color='blue', alpha=0.7)
            plt.title('Histograma de valores de la imagen')
            plt.xlabel('Valor')
            plt.ylabel('Frecuencia')
            plt.savefig(path+"histograms/orig/"+histogram_names+".png")
            plt.close()

        #normalizing
        oimage = (oimage - m + sd_range * sd) / (2 * sd_range * sd)

        #If True, increase contrast of the Normalized image radialy, above a specific radius.
        if increase_contrast:
            modified_image,radius,distance, multiplier = self._apply_linear_multiplier(oimage[:,:,0])
            m1  = np.mean(oimage)
            sd1 = np.std(oimage)
            binary_image = np.where(oimage[:,:,0] > m1+sd1/15, 1, 0) #0.51
            aaa=np.where(distance > 100., 1, 0)
            oimage[np.logical_and(binary_image == 1, aaa == 1)] = np.multiply(oimage, multiplier)[np.logical_and(binary_image == 1, aaa == 1)]
        
        #clipping to 0-1 range
        oimage[oimage > 1]=1
        oimage[oimage < 0]=0

        #checking for nan values and replace for the mean value
        mean = np.mean(oimage)
        if np.isnan(mean):
            self.logger.warning('Found nan values in the normalized image. Replacing with the mean value')
            non_nan_mean = np.nanmean(oimage) # mean of the non-nan values        
            if str(non_nan_mean).isdigit() == False:
                self.logger.warning('Full nan, se pudre la momia')
                
            oimage = np.nan_to_num(oimage, nan=non_nan_mean)

        if plot_histograms:
            normalized_asd = oimage.flatten()
            plt.hist(normalized_asd, bins=50, range=(np.percentile(normalized_asd, 5),  np.percentile(normalized_asd, 95)), color='blue', alpha=0.7)
            plt.hist(normalized_asd, bins=50, range=(0,1), color='blue', alpha=0.7)
            plt.title('Histograma de valores de la imagen Normalizados')
            plt.xlabel('Valor')
            plt.ylabel('Frecuencia')
            plt.savefig(path+"histograms/resize/"+histogram_names+".png")
            plt.close()
        
        return  np.dstack([oimage,oimage,oimage])