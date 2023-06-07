import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models as models


class Resnet_gcs(nn.Module):
    def __init__(self, backbone, input_size, hidden_size, output_size) -> None:
        super(Resnet_gcs, self).__init__()

        self.backbone = backbone
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, image):
        x = self.backbone(image)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x





if __name__ == "__main__":
    backbone = backbone_utils.resnet_fpn_backbone(backbone_name='resnet50', weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = Resnet_gcs(backbone=backbone, input_size=512, hidden_size=256, output_size=6)