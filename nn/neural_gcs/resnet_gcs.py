import os
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import DataLoader
from cme_dataset import CmeDataset
from mlp_resnet_model import Mlp_Resnet

mpl.use('Agg')

EPOCHS = 1
IMAGE_LIMIT = None
BATCH_SIZE = 16
IMG_SiZE = [512, 512]
GPU = 0
TRAINDIR = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_training_mariano'
OPATH = "/gehme-gpu/projects/2020_gcs_with_ml/output/cme_seg_training_mariano"


def compute_loss(predictions, mask):
    pass

def optimize(images_limit=IMAGE_LIMIT):
    losses_per_batch = []
    batch_count = 0
    for epoch in range(EPOCHS):
        stop_flag = False
        for i, (inputs, targets, mask) in enumerate(cme_dataloader, 0):
            inputs, targets = inputs.to(device), targets.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # send inputs to model and get predictions
            predictions = model(inputs)
            # calculate loss
            loss = criterion(predictions, targets)
            # backpropagate loss
            loss.backward()
            # update weights
            optimizer.step()
            # save loss
            losses_per_batch.append(loss.item())
            # print statistics
            if i % 10 == 9:  # print every 10 batches
                print(
                    f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {loss.item():.3f}')
            # add batch to batch count
            batch_count += 1
            # check if we reached the images limit
            if i is not None and i == images_limit:
                stop_flag = True
                break
        if stop_flag:
            break
    # # plot loss
    plt.plot(losses_per_batch)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(OPATH, 'loss.png'))


if __name__ == "__main__":
    cme_dataset = CmeDataset(root_dir=TRAINDIR, img_size=IMG_SiZE)
    cme_dataloader = DataLoader(
        cme_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device(f'cuda:{GPU}') if torch.cuda.is_available(
    ) else torch.device('cpu')  # runing on gpu unles its not available
    print(f'Running on {device}')

    backbone = torchvision.models.resnet101(weights='DEFAULT')
    backbone = backbone.to(device)
    # model = Mlp_Resnet(backbone=backbone, input_size=1000,
    #                    hidden_size=256, output_size=6)
    model = Mlp_Resnet(backbone=backbone)
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([param for param in model.backbone.parameters() if param.requires_grad] + [
                                 param for param in model.regression.parameters() if param.requires_grad], lr=1e-3)

    optimize()
