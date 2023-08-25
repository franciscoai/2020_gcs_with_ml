import torch
import torchvision.models.detection as detection


class Mlp_Resnet(torch.nn.Module):
    def __init__(self, backbone, input_size=1000, hidden_layer=[512, 256], output_size=6, gcs_par_rng=None, trainable_layers=3) -> None:
        super(Mlp_Resnet, self).__init__()
        self.gcs_par_rng = gcs_par_rng
        self.trainable_layers  = trainable_layers
        self.backbone = backbone
        self.input_layer = torch.nn.Linear(input_size, hidden_layer[0])
        self.output_layer = torch.nn.Linear(hidden_layer[-1], output_size)
        self.hidden_layer = self._generateHiddenLayer(hidden_layer)
        self.lrelu = torch.nn.LeakyReLU()      
        #self.backbone = torch.nn.Sequential(*(list(backbone.children())[:-1])) # remove last layer

        # select layers that won't be frozen
        if self.trainable_layers < 0 or self.trainable_layers > 5:
            raise ValueError(f"Trainable layers should be in the range [0,5], got {self.trainable_layers}")
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:self.trainable_layers]
        if self.trainable_layers == 5:
            layers_to_train.append("bn1")
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)


    def forward(self, x):
        freatures = self.backbone(x)
        #freatures = freatures.view(freatures.size(0), -1)
        predictions = self.input_layer(freatures)
        predictions = self.lrelu(predictions)
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
            layers.append(torch.nn.Dropout(p=0.5))
        return torch.nn.Sequential(*layers)