import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
#mpl.use('TkAgg')
from torch.utils.data import DataLoader
from nn.neural_gcs.cme_1VP_dataset import Cme_1VP_Dataset
from nn.neural_gcs.cme_2VP_dataset import Cme_2VP_Dataset
from nn.neural_gcs.sirats_model import Sirats_net
from nn.utils.gcs_mask_generator import maskFromCloud
#from torch.utils.data import random_split

# Train Parameters
DEVICE = 0
TWOVP_MODE = True
INFERENCE_MODE = False
SAVE_MODEL = True
LOAD_MODEL = True
EPOCHS = 25
BATCH_LIMIT = None
BATCH_SIZE = 32
TRAIN_IDX_SIZE = 9500
SEED = 42
IMG_SiZE = [512, 512]
GPU = 0
LR = [1e-3, 1e-5]
# CMElon,CMElat,CMEtilt,height,k,ang
GCS_PAR_RNG = torch.tensor([[-180,180],[-70,70],[-90,90],[8,30],[0.2,0.6], [10,60]]) 
LOSS_WEIGHTS = torch.tensor([100,100,100,10,1,10])
TRAINDIR = '/gehme-gpu/projects/2020_gcs_with_ml/data/gcs_ml_2VP_10k'
OPATH = "/gehme-gpu/projects/2020_gcs_with_ml/output/sirats_v3_2VP_10k_25E"
os.makedirs(OPATH, exist_ok=True)


def plot_masks(img, mask, target, prediction, occulter_mask, satpos, plotranges, opath, namefile):
    img = img.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()
    satpos = satpos.cpu().detach().numpy()
    plotranges = plotranges.cpu().detach().numpy()
    occulter_mask = occulter_mask.cpu().detach().numpy()

    loss = loss_fn(torch.tensor(prediction), torch.tensor(target[None,:]))  # Assuming loss_fn is defined
    prediction = np.squeeze(prediction)
    mask_infered = maskFromCloud(prediction, sat=0, satpos=[satpos], imsize=IMG_SiZE, plotranges=[plotranges])
    mask_infered[occulter_mask > 0] = 0  # Setting 0 where the occulter is
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'target: {np.around(target, 3)}\nPrediction: {np.around(prediction, 3)}')
    
    for i in range(2):
        img = np.squeeze(img)
        ax[i].imshow(img, vmin=0, vmax=1, cmap='gray')
        if i == 0:
            ax[i].imshow(mask, alpha=0.4)
            ax[i].set_title('Target Mask')
        else:
            ax[i].imshow(mask_infered, alpha=0.4)
            ax[i].set_title(f'Prediction Mask: {loss}')
    
    masks_dir = os.path.join(opath, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    plt.savefig(os.path.join(masks_dir, namefile))
    plt.close()

def run_training():
    train_losses_per_batch = []
    mean_train_losses_per_batch = []
    test_losses_per_batch = []
    mean_test_error_in_batch = []
    epoch_list = []
    total_batches_per_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        train_onlyepoch_losses = []
        test_onlyepoch_losses = []
        epoch_list.append(total_batches_per_epoch)  # Store the total number of batches processed
        total_batches_per_epoch = 0

        if not TWOVP_MODE:
            for i, (img, targets, mask, occulter_mask, satpos, plotranges, idx) in enumerate(cme_train_dataloader, 0):
                total_batches_per_epoch += 1
                loss_value = model.optimize_model(img, targets, loss_fn, optimizer, scheduler)
                train_losses_per_batch.append(loss_value.detach().cpu())
                train_onlyepoch_losses.append(loss_value.detach().cpu())

                if i % 10 == 0:            
                    print(f'Epoch: {epoch + 1}, Image: {(i + 1) * BATCH_SIZE}, Batch: {i + 1}, Loss: {loss_value:.5f}, learning rate: {optimizer.param_groups[-1]["lr"]:.7f}')

                if i % 50 == 0:
                    model.plot_loss(train_losses_per_batch, epoch_list, BATCH_SIZE, os.path.join(OPATH, "train_loss.png"), plot_epoch=False)

                if i == BATCH_LIMIT:
                    break
            mean_train_losses_per_batch.append(np.mean(train_onlyepoch_losses))
        else:
            for i, (img, targets, sat1_mask, sat2_mask, occulter_mask_sat1, occulter_mask_sat2, satpos, plotranges, idx) in enumerate(cme_train_dataloader, 0):
                total_batches_per_epoch += 1
                loss_value = model.optimize_model(img, targets, loss_fn, optimizer, scheduler)
                train_losses_per_batch.append(loss_value.detach().cpu())
                train_onlyepoch_losses.append(loss_value.detach().cpu())

                if i % 10 == 0:            
                    print(f'Epoch: {epoch + 1}, Image: {(i + 1) * BATCH_SIZE}, Batch: {i + 1}, Loss: {loss_value:.5f}, learning rate: {optimizer.param_groups[-1]["lr"]:.7f}')

                if i % 50 == 0:
                    model.plot_loss(train_losses_per_batch, epoch_list, BATCH_SIZE, os.path.join(OPATH, "train_loss.png"), plot_epoch=False)

                if i == BATCH_LIMIT:
                    break
            mean_train_losses_per_batch.append(np.mean(train_onlyepoch_losses))

        # Test
        model.eval()
        if not TWOVP_MODE:
            for i, (img, targets, mask, occulter_mask, satpos, plotranges, idx) in enumerate(cme_test_dataloader, 0):
                loss_test = model.test_model(img, targets, loss_fn)
                test_losses_per_batch.append(loss_test.detach().cpu())
                test_onlyepoch_losses.append(loss_test.detach().cpu())
            mean_test_error_in_batch.append(np.mean(test_onlyepoch_losses))
        else:
            for i, (img, targets, sat1_mask, sat2_mask, occulter_mask_sat1, occulter_mask_sat2, satpos, plotranges, idx) in enumerate(cme_test_dataloader, 0):
                loss_test = model.test_model(img, targets, loss_fn)
                test_losses_per_batch.append(loss_test.detach().cpu())
                test_onlyepoch_losses.append(loss_test.detach().cpu())
            mean_test_error_in_batch.append(np.mean(test_onlyepoch_losses))

        print(f'Epoch: {epoch + 1}, Test Loss: {loss_test:.5f}\n')
        model.plot_loss(test_losses_per_batch, epoch_list, BATCH_SIZE, os.path.join(OPATH, "test_loss.png"), plot_epoch=False)
        
        # Plot mean loss per epoch
        model.plot_loss(mean_train_losses_per_batch, epoch_list, BATCH_SIZE, os.path.join(OPATH, "mean_train_loss.png"), plot_epoch=False, meanLoss=True)
        model.plot_loss(mean_test_error_in_batch, epoch_list, BATCH_SIZE, os.path.join(OPATH, "mean_test_loss.png"), plot_epoch=False, meanLoss=True)

        # Save model
        if SAVE_MODEL:
            status = model.save_model(OPATH)
            print(f"Model saved at: {status}\n")

if __name__ == '__main__':
    # train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.90), int(len(dataset) * 0.1)])
    if not TWOVP_MODE:
        dataset = Cme_1VP_Dataset(root_dir=TRAINDIR, img_size=IMG_SiZE)
    else:
        dataset = Cme_2VP_Dataset(root_dir=TRAINDIR, img_size=IMG_SiZE)
    random.seed(SEED)
    total_samples = len(dataset)
    train_size = TRAIN_IDX_SIZE
    train_indices = random.sample(range(total_samples), train_size)
    test_indices = list(set(range(total_samples)) - set(train_indices))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    cme_train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    cme_test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Sirats_net(device=DEVICE, output_size=6, imsize=IMG_SiZE, twoVP_mode=TWOVP_MODE)
    if LOAD_MODEL:
        status = model.load_model(OPATH)
        if status:
            print(f"Model loaded from: {status}\n")
        else:
            print(f"No model found at: {OPATH}, starting from scratch\n")

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_parameters}\n')

    optimizer = torch.optim.Adadelta(model.parameters(), lr=LR[0], weight_decay=0.95)
    #optimizer = torch.optim.Adam(model.parameters(), lr=LR[0])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (len(cme_train_dataloader) / BATCH_SIZE) * EPOCHS, eta_min=LR[1])
    loss_fn = torch.nn.MSELoss()

    if not INFERENCE_MODE:
        run_training()
    else:
        data_iter = iter(cme_test_dataloader)
        if not TWOVP_MODE:
            for i in range(10):
                img, targets, mask, occulter_mask, satpos, plotranges, idx = next(data_iter)
                img, targets, mask, occulter_mask, satpos, plotranges, idx = img[0], targets[0], mask[0], occulter_mask[0], satpos[0], plotranges[0], idx[0]
                img = img.to(DEVICE)
                predictions = model.infer(img)
                plot_masks(img, mask, targets, predictions, occulter_mask, satpos, plotranges, opath=OPATH, namefile=f'targetVinfered_{idx}.png')
        else:
            for i in range(10):
                img, targets, sat1_mask, sat2_mask, occulter_mask_sat1, occulter_mask_sat2, satpos, plotranges, idx = next(data_iter)
                img, targets, sat1_mask, sat2_mask, occulter_mask_sat1, occulter_mask_sat2, satpos, plotranges, idx = img[0], targets[0], sat1_mask[0], sat2_mask[0], occulter_mask_sat1[0], occulter_mask_sat2[0], satpos[0], plotranges[0], idx[0]
                img = img.to(DEVICE)
                predictions = model.infer(img)
                sat1_mask = torch.squeeze(sat1_mask)
                sat2_mask = torch.squeeze(sat2_mask)
                plot_masks(img[0,:,:], sat1_mask, targets, predictions, occulter_mask_sat1, satpos[0,:], plotranges[0,:], opath=OPATH, namefile=f'targetVinfered_{idx}_sat1.png')
                plot_masks(img[1,:,:], sat2_mask, targets, predictions, occulter_mask_sat2, satpos[1,:], plotranges[1,:], opath=OPATH, namefile=f'targetVinfered_{idx}_sat2.png')
