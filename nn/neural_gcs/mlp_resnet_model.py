import torch.nn as nn
import torchvision.models.detection as detection


class Mlp_Resnet(nn.Module):
    def __init__(self, backbone, input_size, hidden_size, output_size) -> None:
        super(Mlp_Resnet, self).__init__()

        self.backbone = backbone
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, 256)
        #self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        #self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, self.output_size)
        self.relu = nn.LeakyReLU()

        # for param in self.backbone.parameters():
        #     param.requires_grad = False # freeze backbone to test if is necessary to train it (it should not be necessary)


    def forward(self, features):
        x = self.fc1(features)
        x = self.relu(x)
        # x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        # x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)


        return x