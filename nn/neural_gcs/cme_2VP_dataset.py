import os
import torchvision
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import warnings
from torchvision.io import read_image
from torch.utils.data import Dataset

class Cme_2VP_Dataset(Dataset):
    def __init__(self, root_dir:str, file_ext:str='.png', img_size:list=[512, 512]):
        self.root_dir = root_dir
        self.imgs = self.__get_dirs(self.root_dir)
        self.file_ext = file_ext
        self.image_size = img_size
        self.transform = self.__define_transforms()
        csv_path = [f for f in os.listdir(self.root_dir) if f.endswith('.csv')][0]
        self.csv_df = pd.read_csv(os.path.join(self.root_dir, csv_path))
       
    def __get_dirs(self, root_dir):
        imgs = []
        dirs = os.listdir(root_dir)
        dirs = [int(d) for d in dirs if not d.endswith('.csv')]
        dirs.sort()
        for d in dirs:
            imgs.append(os.path.join(root_dir, str(d)))
        return imgs
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # file = next((f for f in os.listdir(self.imgs[idx]) if f.endswith(self.file_ext)), None)
        # if file is None:
        #     raise FileNotFoundError("No file with the specified extension found in directory.")
        try:
            mask_dir = os.path.join(self.imgs[idx], 'mask')
            
            sat1_mask = read_image(os.path.join(mask_dir, 'sat1.png'), mode=torchvision.io.image.ImageReadMode.GRAY)
            sat2_mask = read_image(os.path.join(mask_dir, 'sat2.png'), mode=torchvision.io.image.ImageReadMode.GRAY)
            img = torch.zeros((2, sat1_mask.shape[1], sat1_mask.shape[2]))
            img[0, :, :] = sat1_mask
            img[1, :, :] = sat2_mask

            img = img.float()
            img = self.__normalize(img)
            img = self.transform(img)

            occulter_mask_sat1 = read_image(os.path.join(mask_dir, 'sat1_occ.png'), mode=torchvision.io.image.ImageReadMode.GRAY)
            occulter_mask_sat2 = read_image(os.path.join(mask_dir, 'sat2_occ.png'), mode=torchvision.io.image.ImageReadMode.GRAY)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sat1_mask = sat1_mask.new_tensor(sat1_mask > 0, dtype=torch.uint8)
                sat2_mask = sat2_mask.new_tensor(sat2_mask > 0, dtype=torch.uint8)
                occulter_mask_sat1 = occulter_mask_sat1.new_tensor(occulter_mask_sat1 > 0, dtype=torch.uint8)
                occulter_mask_sat2 = occulter_mask_sat2.new_tensor(occulter_mask_sat2 > 0, dtype=torch.uint8)

            resize = torchvision.transforms.Resize(self.image_size, torchvision.transforms.InterpolationMode.BILINEAR)
            sat1_mask = resize(sat1_mask)
            sat2_mask = resize(sat2_mask)
            occulter_mask_sat1 = resize(occulter_mask_sat1)
            occulter_mask_sat2 = resize(occulter_mask_sat2)

            # CMElon,CMElat,CMEtilt,height,k,ang
            labels = ["CMElon", "CMElat", "CMEtilt", "height", "k", "ang"]
            targets = []
            for label in labels:
                targets.append(self.csv_df[label].iloc[idx])
            targets = torch.tensor(targets, dtype=torch.float32)

            satpos = torch.tensor(eval(self.csv_df["satpos"].iloc[idx]), dtype=torch.float32)
            plotranges = torch.tensor(eval(self.csv_df["plotranges"].iloc[idx]), dtype=torch.float32)

            #Squeeze everything
            #img = torch.squeeze(img)
            #targets = torch.squeeze(targets)
            #mask = torch.squeeze(mask)
            occulter_mask_sat1 = torch.squeeze(occulter_mask_sat1)
            occulter_mask_sat2 = torch.squeeze(occulter_mask_sat2)
            satpos = torch.squeeze(satpos)
            plotranges = torch.squeeze(plotranges)

            return img, targets, sat1_mask, sat2_mask, occulter_mask_sat1, occulter_mask_sat2, satpos, plotranges, idx
        except :
            print(f"Error in {self.imgs[idx]}")
            img, targets, sat1_mask, sat2_mask, occulter_mask_sat1, occulter_mask_sat2, satpos, plotranges, idx = self.__getitem__(idx+1)
            return img, targets, sat1_mask, sat2_mask, occulter_mask_sat1, occulter_mask_sat2, satpos, plotranges, idx



    def __normalize(self, img):
        sd_range = 1
        m = torch.mean(img)
        sd = torch.std(img)
        img = (img - m + sd_range * sd) / (2 * sd_range * sd)
        img[img > 1] = 1
        img[img < 0] = 0
        return img

    def __define_transforms(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.image_size, torchvision.transforms.InterpolationMode.BILINEAR),
        ])
        return transform
