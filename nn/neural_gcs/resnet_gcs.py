import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('TkAgg')
mpl.use('Agg')
from torch.utils.data import DataLoader
from cme_dataset import CmeDataset
from mlp_resnet_model import Mlp_Resnet, TinyVgg
from nn.utils.gcs_mask_generator import maskFromCloud
from torch.utils.data import random_split

# Train Parameters
SAVE_MODEL = True
LOAD_MODEL = False
EPOCHS = 100
BATCH_LIMIT = None
BATCH_SIZE = 16
IMG_SiZE = [512, 512]
GPU = 0
LR = 1e-2
# CMElon,CMElat,CMEtilt,height,k,ang
GCS_PAR_RNG = torch.tensor([[-180,180],[-70,70],[-90,90],[8,30],[0.2,0.6], [10,60]]) 
LOSS_WEIGHTS = torch.tensor([100,100,100,10,1,10])
TRAINDIR = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_training_mariano'
OPATH = "/gehme-gpu/projects/2020_gcs_with_ml/output/cme_seg_training_mariano_onlyMask"
os.makedirs(OPATH, exist_ok=True)

#Backbone Parameters
TRAINABLE_LAYERS = 3

def save_model(model, namefile):
    models_path = os.path.join(OPATH, 'models')
    os.makedirs(models_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_path, namefile))

def load_model(model, namefile):
    models_path = os.path.join(OPATH, 'models')
    model.load_state_dict(torch.load(os.path.join(models_path, namefile)))

def plot_loss(losses, namefile):
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(OPATH, namefile))
    plt.close()

def plot_masks(img, mask, target, prediction, occulter_mask, satpos, plotranges, namefile):
    '''
    Plots the target and predicted masks
    '''
    img = img.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()
    satpos = satpos.cpu().detach().numpy()
    plotranges = plotranges.cpu().detach().numpy()
    #loss = compute_loss(torch.tensor(prediction), torch.tensor(target), device='cpu')
    loss = criterion(torch.tensor(prediction), torch.tensor(target))
    occulter_mask = occulter_mask.cpu().detach().numpy()
    prediction = np.squeeze(prediction)
    mask_infered = maskFromCloud(prediction, sat=0, satpos=[satpos], imsize=IMG_SiZE, plotranges=[plotranges])
    mask_infered[occulter_mask > 0] = 0 #setting 0 where the occulter is
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'target: {np.around(target,3)}\nPrediction: {np.around(prediction, 3)}')
    ax[0].imshow(img[0,:,:], vmin=0, vmax=1, cmap='gray')
    ax[0].imshow(mask, alpha=0.4)
    ax[0].set_title('Target Mask')
    ax[1].imshow(img[0,:,:], vmin=0, vmax=1, cmap='gray')
    ax[1].imshow(mask_infered, alpha=0.4)
    ax[1].set_title(f'Prediction Mask: {loss}')
    os.makedirs(os.path.join(OPATH, 'masks'), exist_ok=True)
    plt.savefig(os.path.join(OPATH, 'masks', namefile))
    plt.close()

def compute_loss_old(predictions, targets, mask, occulter_mask, satpos, plotranges):
    '''
    Computes mean square error between predicted and target masks

    NOTE: This function is not working properly because it deletes the gradient information
    '''
    losses = torch.zeros(predictions.shape[0])
    # predictions = predictions.cpu().detach().numpy()
    # mask = mask.cpu().detach().numpy()
    # satpos = satpos.cpu().detach().numpy()
    # plotranges = plotranges.cpu().detach().numpy()
    # occulter_mask = occulter_mask.cpu().detach().numpy()
    # targets = targets.cpu().detach().numpy()
    for i in range(predictions.shape[0]):
        # mask_infer = maskFromCloud(predictions[i,:], sat=0, satpos=[satpos[i,:]], imsize=IMG_SiZE, plotranges=[plotranges[i,:]])
        # mask_infer[occulter_mask[i,:,:] > 0] = 0 #setting 0 where the occulter is
        #loss = torch.sum((torch.tensor(mask_infer) - torch.tensor(mask[i,:,:]))**2) / torch.sum(torch.tensor(mask[i,:,:][mask[i,:,:] > 0])) 
        loss = torch.mean((torch.tensor((predictions[i,:]) - targets[i,:])**2)) #/ targets[i,:])**2)
        losses[i] = loss
        #plot_masks(mask[i], mask_infer, predictions[i])
    return (torch.mean(losses))

def compute_loss(predictions, targets, device='cuda:0'):
    loss_weights = LOSS_WEIGHTS.to(device)
    loss = torch.mean((predictions - targets) / loss_weights[None,:])**2
    return loss

def test_model():
    for i, (inputs, targets, mask, occulter_mask, satpos, plotranges) in enumerate(cme_test_dataloader, 0):
        model.eval()
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.inference_mode():
            test_logits = model(inputs)
            test_loss = criterion(test_logits, targets)
        model.train()
        return test_loss.item()
        

def optimize(batch_limit=BATCH_LIMIT):
    train_losses_per_batch = []
    test_losses_per_batch = []
    batch_count = 0
    model.train()
    for epoch in range(EPOCHS):
        stop_flag = False
        for i, (inputs, targets, mask, occulter_mask, satpos, plotranges) in enumerate(cme_train_dataloader, 0):
            # try:
            inputs, targets = inputs.to(device), targets.to(device)
            # send inputs to model and get predictions
            predictions = model(inputs)
            # calculate loss
            # loss = criterion(predictions, targets)
            #loss = compute_loss(predictions, targets, mask, occulter_mask, satpos, plotranges)
            #loss = compute_loss(predictions, targets)
            loss = criterion(predictions, targets)
            # zero the parameter gradients
            optimizer.zero_grad()
            # backpropagate loss
            loss.backward()
            # update weights
            optimizer.step()
            # save loss
            train_losses_per_batch.append(loss.item())
            # print statistics
            if i % 10 == 9:  # print every 10 batches
                print(f'Epoch: {epoch + 1}, Image: {(i+1)*BATCH_SIZE}, Batch: {i + 1}, Loss: {loss.item():.5f}, learning rate: {optimizer.param_groups[0]["lr"]:.5f}')
                plot_loss(train_losses_per_batch, "train_loss.png")
                test_loss = test_model()
                test_losses_per_batch.append(test_loss)
                print(f'Test loss: {test_loss:.5f}')
                plot_loss(test_losses_per_batch, "test_loss.png")            
            # add batch to batch count
            batch_count += 1
            # check if we reached the images limit
            if i is not None and i == batch_limit:
                stop_flag = True
                break
            # except:
            #     print(f'Error in batch {i}, skipping...')
            #     continue
        if stop_flag:
            break
    #save model
    if SAVE_MODEL:
        save_model(model, "model.pth")
    
    #plot target mask and predicted mask
    data_iter = iter(cme_train_dataloader)
    for i in range(10):
        inputs, targets, mask, occulter_mask, satpos, plotranges = next(data_iter)
        inputs, targets, mask, occulter_mask, satpos, plotranges = inputs[0], targets[0], mask[0], occulter_mask[0], satpos[0], plotranges[0]
        inputs = inputs.to(device)
        predictions = model(inputs[None,:,:,:])
        plot_masks(inputs, mask, targets, predictions, occulter_mask, satpos, plotranges, namefile=f'targetVinfered_{i}.png')
    


if __name__ == "__main__":
    cme_dataset = CmeDataset(root_dir=TRAINDIR, img_size=IMG_SiZE)
    train_dataset, test_dataset = random_split(cme_dataset, [int(len(cme_dataset)*0.85), int(len(cme_dataset)*0.15)])
    cme_train_dataloader = DataLoader(cme_dataset, batch_size=BATCH_SIZE, shuffle=True)
    cme_test_dataloader = DataLoader(cme_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device(f'cuda:{GPU}') if torch.cuda.is_available(
    ) else torch.device('cpu')  # runing on gpu unles its not available
    print(f'Running on {device}')

    backbone = torchvision.models.resnet50(weights='DEFAULT', num_classes=1000)
    backbone = backbone.to(device)
    model = Mlp_Resnet(backbone=backbone, gcs_par_rng=GCS_PAR_RNG, trainable_layers=TRAINABLE_LAYERS)
    if LOAD_MODEL:
        load_model(model, "model.pth")
    model.to(device)
    model_params = [param for param in model.backbone.parameters() if param.requires_grad] + [param for param in model.parameters() if param.requires_grad]
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model_params, lr=LR)
    optimize()
