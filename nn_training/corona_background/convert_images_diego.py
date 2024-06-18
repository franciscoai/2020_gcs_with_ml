from astropy.io import fits
import numpy as np
import sys
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import os

def rebin(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.
    Number of output dimensions must match number of input dimensions.
    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)
    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]
    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray

def read_fits(file_path, header=False, imageSize=[],smooth_kernel=[0,0]):
    try:       
        img = fits.open(file_path)[0].data
        img=img.astype("float32")
        img = gaussian_filter(img, sigma=smooth_kernel)
        if len(imageSize) != 0:
            img = rebin(img, imageSize, operation='mean')  
        if header:
            hdr = fits.open(file_path)[0].header
            if len(imageSize) != 0:
                naxis_original = hdr["naxis1"]
                hdr["naxis1"]=imageSize[0]
                hdr["naxis2"]=imageSize[1]
                hdr["crpix1"]=hdr["crpix1"] / (naxis_original/imageSize[0])
                hdr["crpix2"]=hdr["crpix2"] / (naxis_original/imageSize[0])
                hdr["cdelt1"]=hdr["cdelt1"] * (naxis_original/imageSize[0])
                hdr["cdelt2"]=hdr["cdelt2"] * (naxis_original/imageSize[0])
        # flips y axis to match the orientation of the images
        img = np.flip(img, axis=0) 
        if header:
            return img, hdr
        else:
            return img 
    except:
        print(f'WARNING. I could not find file {file_path}')
        return None

def running_difference(img1, img2,dir1='',dir2=''):
    img1, hdr1 = read_fits(dir1+img1,header=True)
    img2, hdr2 = read_fits(dir2+img2,header=True)
    img_diff = img2 - img1
    return img_diff, hdr2

def save_fits(img, hdr, path):
    hdu = fits.PrimaryHDU(img, header=hdr)
    hdu.writeto(path, overwrite=True)

path='/gehme/projects/2020_gcs_with_ml/data/corona_background_affects/lasco/c2/tmp/'
files=[f for f in os.listdir(path) if f.endswith('.fts')]
files.sort()
print(files)
#for loop that goes through all the files and saves the running difference
#the for loop takes values ​​two at a time and without repeating
for i in range(0,len(files)-1,2):
    img_diff, hdr_diff = running_difference(files[i], files[i+1],dir1=path,dir2=path)
    
    save_fits(img_diff, hdr_diff, f'/gehme/projects/2020_gcs_with_ml/data/corona_background_affects/lasco/c2/{files[i]}')
    print(f'Processed {files[i]} and {files[i+1]} ;saved as {files[i]}')

breakpoint()