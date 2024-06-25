import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))))
from ext_libs.rebin import rebin
from scipy.ndimage import gaussian_filter


def read_fits(file_path,smooth_kernel=[0,0]):
    imageSize=[512,512]
    try: 
        img = fits.open(file_path)
        img=(img[0].data).astype("float32")
        img = gaussian_filter(img, sigma=smooth_kernel)
        if smooth_kernel[0]!=0 and smooth_kernel[1]!=0: 
            img = rebin(img, imageSize,operation='mean')
        return img  
    except:
        print(f'WARNING. could not find file {file_path}')
        return None


################################## MAIN ################################################
imsize_nn=[512,512]
imsize=[0,0]
file1 =['/gehme/data/soho/lasco/level_05/c2/20021110/22134067.fts']                
file2 = ['/gehme/data/soho/lasco/level_05/c2/20021110/22134066.fts']
file3 = ['/gehme/data/soho/lasco/level_1/c2/20070108/25243923.fts']


image1 = read_fits(file1[0])# final image
image2 = read_fits(file2[0])# initial image
image3 = read_fits(file3[0])
header1 = fits.getheader(file1[0])
header2 = fits.getheader(file2[0])
header3 = fits.getheader(file3[0])
breakpoint()
if (image1 is not None) and (image2 is not None) and (image3 is not None):
    # Check shape
    if header1['NAXIS1'] != imsize_nn[0]:
        scale=(header1['NAXIS1']/imsize_nn[0])
        plt_scl = header1['CDELT1'] * scale
    else:
        plt_scl = header1['CDELT1']
    # Resize images if necessary
    if (imsize[0]!=0 and imsize[1]!=0) or (image1.shape !=image2.shape):
        image1 = rebin(image1,imsize_nn,operation='mean') 
        image2 = rebin(image2,imsize_nn,operation='mean')
        image3 = rebin(image3,imsize_nn,operation='mean')
        header1['NAXIS1'] = imsize_nn[0]   
        header1['NAXIS2'] = imsize_nn[1]


    img=image1-image2







filename=os.path.basename(file1[0])[:-4]
color=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w']

cmap = mpl.colors.ListedColormap(color)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image1, cmap='gray')
axes[0].set_title('Final time image')
axes[0].axis('off')
axes[1].imshow(image2, cmap='gray')
axes[1].set_title('Initial time image')
axes[1].axis('off')
axes[2].imshow(image3, cmap='gray')
axes[2].set_title('Level 1 image ')
axes[2].axis('off')

plt.savefig("/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat_L1/lasco_extreme_cases/"+filename+".png", dpi=300)
