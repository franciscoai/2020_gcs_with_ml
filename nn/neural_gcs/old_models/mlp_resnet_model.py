import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


class Mlp_Resnet(torch.nn.Module):
    def __init__(self, device, backbone_model, hidden_layer=[2048, 1024], output_size=6, gcs_par_rng=None, trainable_layers=3, imsize=[512,512]) -> None:
        super(Mlp_Resnet, self).__init__()
        
        self.backbone_model_path = "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/9999.torch"
        self.labels=['Background','Occ','CME','CME','CME','CME','CME','CME'] # labels for the different classes
        self.num_classes = 3 # background, CME, occulter
        self.device = torch.device(f'cuda:{device}') if torch.cuda.is_available(
        ) else torch.device('cpu')  # runing on gpu unles its not available
        self.imsize = imsize
        self.backbone_model = backbone_model
        self.trainable_layers  = trainable_layers
        if self.backbone_model == "maskrcnn":
            self.backbone=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=self.trainable_layers) 
            self.in_features = self.backbone.roi_heads.box_predictor.cls_score.in_features 
            self.backbone.roi_heads.box_predictor=FastRCNNPredictor(self.in_features,num_classes=self.num_classes)
            model_param = torch.load(self.backbone_model_path, map_location=self.device)
            self.backbone.load_state_dict(model_param)      
            self.backbone.to(self.device)
            self.backbone.backbone.fpn.register_forward_hook(self.get_features)
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            self.input_size=36864
            self.maxpool = torch.nn.MaxPool2d(kernel_size=16, stride=16, padding=1)

        self.input_layer = torch.nn.Linear(self.input_size, hidden_layer[0])
        self.output_layer = torch.nn.Linear(hidden_layer[-1], output_size)
        self.hidden_layer = self._generateHiddenLayer(hidden_layer)
        
        self.gcs_par_rng = gcs_par_rng
        self.lrelu = torch.nn.LeakyReLU()
        

        # select layers that won't be frozen
        # if self.backbone_model == "resnet18":
        #     if self.trainable_layers < 0 or self.trainable_layers > 5:
        #         raise ValueError(f"Trainable layers should be in the range [0,5], got {self.trainable_layers}")
        #     layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:self.trainable_layers]
        #     if self.trainable_layers == 5:
        #         layers_to_train.append("bn1")
        #     for name, parameter in self.backbone.named_parameters():
        #         if all([not name.startswith(layer) for layer in layers_to_train]):
        #             parameter.requires_grad_(False)


    def forward(self, x):
        infered_mask = self.backbone(x)
        features = self.maxpool(self.features)
        features = torch.flatten(features, start_dim=1)
        predictions = self.input_layer(features)
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
            #layers.append(torch.nn.Dropout(p=0.5))
        return torch.nn.Sequential(*layers)
    

    def get_features(self, module, inputs, outputs):
        if self.backbone_model == "maskrcnn":
            self.features = outputs["0"]
