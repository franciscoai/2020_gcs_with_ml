import os
import torchvision
import torch
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class CmeDataset(Dataset):
    def __init__(self, root_dir:str, file_ext:str='.png', img_size:list=[512, 512]):
        self.root_dir = root_dir
        self.imgs = self.__get_dirs(self.root_dir)
        self.file_ext = file_ext
        self.image_size = img_size
        self.transform = self.__define_transforms()
        #find .csv files
        csv_path = [f for f in os.listdir(self.root_dir) if f.endswith('.csv')][0]
        self.csv_df = pd.read_csv(os.path.join(self.root_dir, csv_path))



        
    def __get_dirs(self, root_dir):
        imgs = []
        dirs = os.listdir(root_dir)
        dirs = [d for d in dirs if not d.endswith('.csv')]
        for d in dirs:
            imgs.append(os.path.join(root_dir, d))
        return imgs
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        file = [f for f in os.listdir(self.imgs[idx]) if f.endswith(self.file_ext)]
        mask_dir = os.path.join(self.imgs[idx], 'mask')
        img = read_image(os.path.join(self.imgs[idx], file[0]), mode=torchvision.io.image.ImageReadMode.RGB)
        img = img.float()
        img = self.__normalize(img)
        img = self.transform(img)
        mask = read_image(os.path.join(mask_dir, '2.png'), mode=torchvision.io.image.ImageReadMode.GRAY)
        mask = (mask > 0).float()
        resize = torchvision.transforms.Resize(self.image_size, torchvision.transforms.InterpolationMode.BILINEAR)
        mask = resize(mask)
        parameters = file[0].split('_')
        parameters = parameters[:6]
        parameters = [float(p) for p in parameters]
        parameters = torch.tensor(parameters, dtype=torch.float32)
        satpos = torch.tensor(eval(self.csv_df["satpos"].iloc[idx]), dtype=torch.float32)
        plotranges = torch.tensor(eval(self.csv_df["plotranges"].iloc[idx]), dtype=torch.float32)
        return img, parameters, mask, satpos, plotranges

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
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # transform = torchvision.transforms.Compose([
        #     torchvision.transforms.Resize(self.image_size, torchvision.transforms.InterpolationMode.BILINEAR),
        # ])

    #     transform = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(232, torchvision.transforms.InterpolationMode.BILINEAR),
    #     torchvision.transforms.CenterCrop(224),
    #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
        return transform
    
