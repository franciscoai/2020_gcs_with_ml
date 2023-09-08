import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))


class Sirats_net(torch.nn.Module):
    def __init__(self, device, output_size=6, imsize=[512, 512]):
        super(Sirats_net, self).__init__()

        # general params
        self.device = torch.device(
            f'cuda:{device}') if torch.cuda.is_available() else torch.device('cpu')
        print(f'\nUsing device: {self.device}\n')
        self.imsize = imsize
        self.loss_weights = torch.tensor([100,100,100,10,1,10])

        # conv params
        self.block1 = self._generate_block(
            in_channels=1, out_channels=48, kernel_size=2, stride=2, padding=0, pool=True, pool_params=[3, 1, 1])
        self.block2 = self._generate_block(
            in_channels=48, out_channels=128, kernel_size=5, stride=3, padding=2, pool=True, pool_params=[3, 1, 1])
        self.block3 = self._generate_block(
            in_channels=128, out_channels=192, kernel_size=3, stride=3, padding=2, pool=False)
        self.block4 = self._generate_block(
            in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, pool=False)
        self.block5 = self._generate_block(
            in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1, pool=True, pool_params=[3, 1, 1])
        
        # fc
        self.fc = torch.nn.Sequential(
            #torch.nn.Dropout(),
            torch.nn.Linear(30*30*128, 4096),
            torch.nn.ReLU(inplace=True),
            #torch.nn.Dropout(),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, output_size),
        )
        
        # send to device
        self.to(self.device)

    def forward(self, img):
        x = self.block1(img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), 30*30*128)
        x = self.fc(x)
        return x

    def optimize_model(self, img, targets, loss, optimizer, scheduler):
        img, target = img.to(self.device), targets.to(self.device)
        self.output_params = self.forward(img)
        loss_value = loss(self.output_params, target)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        #scheduler.step()
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

    def plot_loss(self, losses, epoch_list, batch_size, opath, plot_epoch=True):
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

    def _generate_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=True, pool_params=[1, 1, 0]):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            (torch.nn.MaxPool2d(kernel_size=pool_params[0], stride=pool_params[1], padding=pool_params[2]) if pool else torch.nn.Identity())
        )

    def _generateHiddenLayer(self, hidden_layer):
        layers = []
        for i in range(len(hidden_layer)-1):
            layers.append(torch.nn.Linear(hidden_layer[i], hidden_layer[i+1]))
            layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Dropout(p=0.95))
        return torch.nn.Sequential(*layers)