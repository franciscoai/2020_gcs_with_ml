import torch
import torchvision.models.detection as detection


class Mlp_Resnet(torch.nn.Module):
    def __init__(self, backbone, input_size=1000, hidden_layer=[512, 256, 128, 16], output_size=6, gcs_par_rng=None) -> None:
        super(Mlp_Resnet, self).__init__()
        self.gcs_par_rng = gcs_par_rng
        #self.backbone = backbone
        self.input_layer = torch.nn.Linear(input_size, hidden_layer[0])
        self.output_layer = torch.nn.Linear(hidden_layer[-1], output_size)
        self.hidden_layer = self._generateHiddenLayer(hidden_layer)        
        #self.backbone = torch.nn.Sequential(*(list(backbone.children())[:-1])) # remove last layer

        # freeze backbone parameters
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # Unfreeze last 2 conv layers and the linear layer
        # for param in self.backbone.layer4.parameters():
        #     param.requires_grad = True

        
        

    def forward(self, x):
        #freatures = self.backbone(x)
        #freatures = freatures.view(freatures.size(0), -1)
        predictions = self.input_layer(x)
        predictions = self.hidden_layer(predictions)
        predictions = self.output_layer(predictions)
        # for i in range(predictions.shape[0]):
        #     for j in range(predictions.shape[1]):
        #         if j in [0,1,2]:
        #             predictions[i][j] = torch.tanh(predictions[i][j]) * torch.max(torch.abs(self.gcs_par_rng[j,:]))
        #         else:
        #             predictions[i][j] = torch.sigmoid(predictions[i][j]) * torch.max(torch.abs(self.gcs_par_rng[j,:]))
        # for i in range(predictions.shape[0]):
        #     for j in range(predictions.shape[1]):
        #         if j in [0,1,2]:
        #             #predictions[i][j] += 20 #torch.tanh(predictions[i][j]) * torch.max(torch.abs(self.gcs_par_rng[j,:])) 
        #             continue
        #         else:
        #             predictions[i][j] += self.gcs_par_rng[j,0]
        return predictions

    def _generateHiddenLayer(self, hidden_layer):
        layers = []
        for i in range(len(hidden_layer)-1):
            layers.append(torch.nn.Linear(hidden_layer[i], hidden_layer[i+1]))
            layers.append(torch.nn.LeakyReLU())
        return torch.nn.Sequential(*layers)