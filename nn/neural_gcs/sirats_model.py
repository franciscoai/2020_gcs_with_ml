import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))

# imports for inception model
from torchvision.models.inception import inception_v3, Inception_V3_Weights


class Sirats_net(nn.Module):
    def __init__(self, device, output_size=6, img_shape=[3, 512, 512], loss_weights=[100,100,100,10,1,10]):
        super(Sirats_net, self).__init__()

        self.device = torch.device(
            f'cuda:{device}') if torch.cuda.is_available() else torch.device('cpu')
        print(f'\nUsing device: {self.device}\n')

        self.img_shape = img_shape
        self.loss_weights = loss_weights

        self.block_list = nn.ModuleList([self._generateBlock() for c in range(self.img_shape[0])])

        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(30*30*128, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, output_size),
        )

        # send to device
        self.to(self.device)

    def forward(self, img_stack):
        feature_maps = []
        for i in range(img_stack.shape[1]):
            x = self.block_list[i](img_stack[:,i,:,:].unsqueeze(1))
            x = x.unsqueeze(4)
            feature_maps.append(x)
        feature_maps = torch.cat(feature_maps, dim=4)
        feature_maps = torch.max(feature_maps, dim=4)[0] # max pooling??
        feature_maps = torch.flatten(feature_maps, start_dim=1)
        x = self.fc(feature_maps)
        return x


    def optimize_model(self, img, targets, optimizer, par_loss_weight, scheduler=None):
        img, target, par_loss_weight = img.to(self.device), targets.to(self.device), par_loss_weight.to(self.device)
        self.output_params = self.forward(img)
        loss_value = self.weighted_mse_loss(self.output_params, target, par_loss_weight)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return loss_value
    
    def weighted_mse_loss(self, input, target, weight):
        return torch.mean(weight * (input - target) ** 2)
    
    def test_model(self, img, targets, par_loss_weight):
        img, target, par_loss_weight = img.to(self.device), targets.to(self.device), par_loss_weight.to(self.device)
        self.output_params = self.forward(img)
        loss_value = self.weighted_mse_loss(self.output_params, target, par_loss_weight)
        return loss_value
    
    def infer(self, img):
        self.eval()
        with torch.inference_mode():
            predictions = self.forward(img[None, :, :, :])
        return predictions
    
    def custom_loss(self, predictions, targets, device='cuda:0'):
        self.loss_weights = self.loss_weights.to(device)
        loss = torch.mean((predictions - targets) / self.loss_weights[None,:])**2
        return loss

    def plot_loss(self, losses, epoch_list, batch_size, opath, plot_epoch=True, meanLoss=False):
        if not meanLoss:
            plt.plot(np.arange(len(losses))*batch_size, losses)
            plt.yscale('log')
            plt.xlabel('# Image')
            plt.ylabel('Loss')
            plt.grid("both")
            # add vertical line every epoch
            if plot_epoch:
                for epoch in range(len(epoch_list)):
                    plt.axvline(x=epoch*epoch_list[epoch]
                                * batch_size, color='r', linestyle='--')
            plt.savefig(opath)
            plt.close()
        else:
            plt.plot(losses)
            plt.yscale('log')
            plt.xlabel('# Epoch')
            plt.ylabel('Mean Loss')
            plt.grid("both")
            plt.savefig(opath)
            plt.close()

    def save_model(self, opath):
        models_path = os.path.join(opath, 'models')
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), models_path+'/model.pth')
        return models_path+'/model.pth'

    def load_model(self, opath):
        models_path = os.path.join(opath, 'models')
        if os.path.exists(models_path+'/model.pth'):
            self.load_state_dict(torch.load(models_path+'/model.pth'))
            return models_path+'/model.pth'
        else:
            return None

    def _generateLayer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=True, pool_params=[1, 1, 0]):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            (torch.nn.MaxPool2d(kernel_size=pool_params[0], stride=pool_params[1], padding=pool_params[2]) if pool else torch.nn.Identity())
        )
    
    def _generateBlock(self): # Here we define the conv block structure for every image.
        layer1 = self._generateLayer(
            in_channels=1, out_channels=48, kernel_size=2, stride=2, padding=0, pool=True, pool_params=[3, 1, 1])
        layer2 = self._generateLayer(
            in_channels=48, out_channels=128, kernel_size=5, stride=3, padding=2, pool=True, pool_params=[3, 1, 1])
        layer3 = self._generateLayer(
            in_channels=128, out_channels=192, kernel_size=3, stride=3, padding=2, pool=False)
        layer4 = self._generateLayer(
            in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, pool=False)
        layer5 = self._generateLayer(
            in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1, pool=True, pool_params=[3, 1, 1])
        return torch.nn.Sequential(layer1, layer2, layer3, layer4, layer5)

class Sirats_inception(nn.Module):
    def __init__(self, device, optimizer=None, loss_fn=None, scheduler=None, output_size=6, img_shape=[3, 512, 512], loss_weights=[100,100,100,10,1,10]):
        super(Sirats_inception, self).__init__()

        self.device = torch.device(
            f'cuda:{device}') if torch.cuda.is_available() else torch.device('cpu')
        print(f'\nUsing device: {self.device}\n')

        self.img_shape = img_shape
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.loss_weights = loss_weights

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(512, output_size),
        )

        self.backbone = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.backbone.aux_logits = False
        self.backbone.fc = torch.nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = True

        # send to device
        self.to(self.device)

    def forward(self, x):
        x = self.backbone(x)
        output = self.fc(x)
        return output

    def optimize_model(self, img, targets, par_loss_weight):
        img, target, par_loss_weight = img.to(self.device), targets.to(self.device), par_loss_weight.to(self.device)
        self.output_params = self.forward(img)
        # loss_value = self.loss_fn(self.output_params, target)
        loss_value = self.weighted_mse_loss(self.output_params, target, par_loss_weight)
        self.optimizer.zero_grad()
        loss_value.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss_value
    
    def test_model(self, img, targets, par_loss_weight):
        img, target, par_loss_weight = img.to(self.device), targets.to(self.device), par_loss_weight.to(self.device)
        self.output_params = self.forward(img)
        loss_value = self.weighted_mse_loss(self.output_params, target, par_loss_weight)
        return loss_value
    
    def weighted_mse_loss(self, input, target, weight):
        return torch.mean(weight * (input - target) ** 2)
    
    def infer(self, img):
        self.eval()
        with torch.inference_mode():
            predictions = self.forward(img)
        return predictions
    
    def custom_loss(self, predictions, targets, device='cuda:0'):
        self.loss_weights = self.loss_weights.to(device)
        loss = torch.mean((predictions - targets) / self.loss_weights[None,:])**2
        return loss

    def plot_loss(self, losses, epoch_list, batch_size, opath, plot_epoch=True, meanLoss=False):
        if not meanLoss:
            plt.plot(np.arange(len(losses))*batch_size, losses)
            plt.yscale('log')
            plt.xlabel('# Image')
            plt.ylabel('Loss')
            plt.grid("both")
            # add vertical line every epoch
            if plot_epoch:
                for epoch in range(len(epoch_list)):
                    plt.axvline(x=epoch*epoch_list[epoch]
                                * batch_size, color='r', linestyle='--')
            plt.savefig(opath)
            plt.close()
        else:
            plt.plot(losses)
            plt.yscale('log')
            plt.xlabel('# Epoch')
            plt.ylabel('Mean Loss')
            plt.grid("both")
            plt.savefig(opath)
            plt.close()

    def save_model(self, opath):
        models_path = os.path.join(opath, 'models')
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), models_path+'/model.pth')
        return models_path+'/model.pth'

    def load_model(self, opath):
        models_path = os.path.join(opath, 'models')
        if os.path.exists(models_path+'/model.pth'):
            self.load_state_dict(torch.load(models_path+'/model.pth'))
            return models_path+'/model.pth'
        else:
            return None
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler