import torch


def binary_mask_normalization(img: torch.Tensor):
    img[img > 1] = 1
    img[img < 0] = 0
    return img

def real_img_normalization(img: torch.Tensor):
    sd_range=1.5
    m = torch.mean(img)
    sd = torch.std(img)
    img = (img - m + sd_range * sd) / (2 * sd_range * sd)
    img[img >1]=1
    img[img <0]=0
    return img