import torch
import torchvision.models.detection as detection


class Mlp_Resnet(torch.nn.Module):
    def __init__(self, backbone, input_size=1000, hidden_size=512, output_size=6, gcs_par_rng=None) -> None:
        super(Mlp_Resnet, self).__init__()
        self.gcs_par_rng = gcs_par_rng
        self.backbone = backbone
        # regression head
        self.regression = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size*8),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(output_size*8, output_size)
        )
        #self.backbone = torch.nn.Sequential(*(list(backbone.children())[:-1])) # remove last layer

        # freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 2 conv layers and the linear layer
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        
        

    def forward(self, x):
        freatures = self.backbone(x)
        #freatures = freatures.view(freatures.size(0), -1)
        predictions = self.regression(freatures)
        # for i in range(predictions.shape[0]):
        #     for j in range(predictions.shape[1]):
        #         if j in [0,1,2]:
        #             predictions[i][j] = torch.tanh(predictions[i][j]) * torch.max(torch.abs(self.gcs_par_rng[j,:]))
        #         else:
        #             predictions[i][j] = torch.sigmoid(predictions[i][j]) * torch.max(torch.abs(self.gcs_par_rng[j,:]))
        for i in range(predictions.shape[0]):
            for j in range(predictions.shape[1]):
                if j in [0,1,2]:
                    #predictions[i][j] += 20 #torch.tanh(predictions[i][j]) * torch.max(torch.abs(self.gcs_par_rng[j,:])) 
                    continue
                else:
                    predictions[i][j] += self.gcs_par_rng[j,0]
        return predictions
