import warnings
import os
import torchvision
import torch
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import Dataset
from torchvision.io import read_image
mpl.use('Agg')


class Cme_MVP_Dataset(Dataset):
    def __init__(self, root_dir: str, num_vp: int = 3, file_ext: str = '.png', img_size: list = [3, 512, 512], only_mask: bool = False):
        self.root_dir = root_dir
        self.num_vp = num_vp
        self.imgs = self.__get_dirs(self.root_dir)
        self.file_ext = file_ext
        self.img_size = img_size
        self.only_mask = only_mask
        self.transform = self.__define_transforms()
        csv_path = [f for f in os.listdir(
            self.root_dir) if f.endswith('Set_Parameters.csv')][0]
        self.csv_df = pd.read_csv(os.path.join(self.root_dir, csv_path))

    def __get_dirs(self, root_dir):
        imgs = []
        dirs = os.listdir(root_dir)
        dirs = [int(d) for d in dirs if not d.endswith('.csv') and not d.endswith('.back')]
        dirs.sort()
        for d in dirs:
            imgs.append(os.path.join(root_dir, str(d)))
        return imgs

    def __getitem__(self, idx):
        """
        Retrieves the data corresponding to the given index.

        Args:
            idx (int): The index of the data to retrieve.

        Returns:
            tuple: A tuple containing the following elements:
                - img (torch.Tensor): The satellite masks.
                - targets (torch.Tensor): The target values.
                - sat_masks (torch.Tensor): The binary satellite masks.
                - occulter_masks (torch.Tensor): The binary occulter masks.
                - satpos (torch.Tensor): The satellite positions.
                - plotranges (torch.Tensor): The plot ranges.
                - idx (int): The index of the retrieved data.
        """
        try:
            flag = True
            
            mask_dir = os.path.join(self.imgs[idx], 'mask')

            # Read the images
            for i in range(self.num_vp):
                if self.only_mask:
                    sat_mask = read_image(os.path.join(
                        mask_dir, f'sat{i+1}.png'), mode=torchvision.io.image.ImageReadMode.GRAY)
                    if flag:
                        img = torch.zeros((self.num_vp, sat_mask.shape[1], sat_mask.shape[2]))
                        flag = False
                    img[i, :, :] = sat_mask
                else:
                    sat_imgs = [f for f in os.listdir(self.imgs[idx]) if f != 'mask']
                    sat_imgs.sort(key=lambda x: x.split('sat')[1].split('.')[0]) # Sort by satellite number
                    for sat_img in sat_imgs:
                        sat_img = read_image(os.path.join(self.imgs[idx], sat_img), mode=torchvision.io.image.ImageReadMode.GRAY)
                        if flag:
                            img = torch.zeros((self.num_vp, sat_img.shape[1], sat_img.shape[2]))
                            flag = False
                        img[i, :, :] = sat_img
                    
            img = img.float()
            img = self.__normalize(img)
            if self.img_size[0] != None:
                img = self.transform(img)

            # Read the occulter masks
            flag = True
            for i in range(self.num_vp):
                occulter_mask = read_image(os.path.join(
                    mask_dir, f'sat{i+1}_occ.png'), mode=torchvision.io.image.ImageReadMode.GRAY)
                if flag:
                    occulter_masks = torch.zeros(
                        (self.num_vp, occulter_mask.shape[1], occulter_mask.shape[2]))
                    flag = False
                occulter_masks[i, :, :] = occulter_mask

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sat_masks = img.new_tensor(img > 0, dtype=torch.uint8)
                occulter_masks = occulter_masks.new_tensor(
                    occulter_masks > 0, dtype=torch.uint8)

            resize = torchvision.transforms.Resize(self.img_size[1:3], torchvision.transforms.InterpolationMode.BILINEAR)
            sat_masks = resize(sat_masks)
            occulter_masks = resize(occulter_masks)

            labels = ["CMElon", "CMElat", "CMEtilt", "height", "k", "ang"]
            targets = []
            for label in labels:
                targets.append(self.csv_df[label].iloc[idx])
            targets = torch.tensor(targets, dtype=torch.float32)

            satpos = torch.tensor(
                eval(self.csv_df["satpos"].iloc[idx]), dtype=torch.float32)
            plotranges = torch.tensor(
                eval(self.csv_df["plotranges"].iloc[idx]), dtype=torch.float32)

            # Squeeze everything
            # img = torch.squeeze(img)
            # targets = torch.squeeze(targets)
            # mask = torch.squeeze(mask)
            occulter_masks = torch.squeeze(occulter_masks)

            satpos = torch.squeeze(satpos)
            plotranges = torch.squeeze(plotranges)
            return img, targets, sat_masks, occulter_masks, satpos, plotranges, idx
        
        except Exception as e:
            logging.error(f"Error reading data from {self.imgs[idx]}: {str(e)}")
            return self.__getitem__(idx+1)

    def __normalize(self, img):
        if self.only_mask:
            img[img > 1] = 1
            img[img < 0] = 0
        else: # Real image
            sd_range=1.5
            m = torch.mean(img)
            sd = torch.std(img)
            img = (img - m + sd_range * sd) / (2 * sd_range * sd)
            img[img >1]=1
            img[img <0]=0
            
        if torch.isnan(img).any():
            raise ValueError("NaN values found in the image tensor.")
            
        return img

    def __define_transforms(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                self.img_size[1:3], torchvision.transforms.InterpolationMode.BILINEAR),
        ])
        return transform
    
    def __len__(self):
        return len(self.imgs)