import torch
import torchvision.models.detection as detection


class Mlp_Resnet(torch.nn.Module):
    def __init__(self, backbone, input_size=1000, hidden_size=512, output_size=6) -> None:
        super(Mlp_Resnet, self).__init__()

        self.backbone = backbone
        # self.backbone = torch.nn.Sequential(*(list(backbone.children())[:-1])) # remove last layer

        # freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 2 conv layers and the linear layer
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # regression head
        self.regression = torch.nn.Sequential(
            torch.nn.Linear(1000, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 6),
            # torch.nn.ReLU(),
            # torch.nn.Linear(512, 256),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 6),
        )

    def forward(self, x):
        freatures = self.backbone(x)
        #freatures = freatures.view(freatures.size(0), -1)
        predictions = self.regression(freatures)
        return predictions
        