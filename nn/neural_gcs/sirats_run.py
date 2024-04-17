import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from sirats_normalization import *
from pathlib import Path
from pyGCS_raytrace import pyGCS
from astropy.io import fits
from torch.utils.data import DataLoader
from nn.neural_gcs.cme_mvp_dataset import Cme_MVP_Dataset
from nn.neural_gcs.sirats_model import *
from nn.neural_gcs.sirats_config import Configuration
from pyGCS_raytrace import pyGCS
from sirats_utils.sirats_plotter import SiratsPlotter
from sirats_utils.utils_functions import get_paths_cme_exp_sources
import cv2
import torch
import numpy as np
import random
import pickle
import torchvision
import logging


def load_model(model: SiratsNet, model_folder: Path):
    model_path = os.path.join(model_folder, 'model.pth')
    os.makedirs(model_folder, exist_ok=True)  # Ensure directory exists
    if os.path.isfile(model_path):
        status = model.load_model(model_path)  # Load directly
        if status:
            copy_and_rename_existing_model(model_folder)
            logging.info(f"Model loaded from: {model_path}\n")
    else:
        logging.warning(
            f"No model found at: {model_path}, starting from scratch\n")

def copy_and_rename_existing_model(model_folder: Path):
    models_counter = len(os.listdir(model_folder))
    new_path = os.path.join(model_folder, f"model_run{models_counter}")
    os.system(f'cp {os.path.join(model_folder, "model.pth")} {new_path}')

def add_occulter(img, occulter_size, centerpix, repleace_value=None):
    '''
    Replace a circular area of radius occulter_size in input image[h,w,3] with a constant value
    repleace_value: if None, the area is replaced with the image mean. If set to scalar float that value is used
    '''
    if centerpix is not None:
        w = int(round(centerpix[0]))
        h = int(round(centerpix[1]))
    else:
        h, w = img.shape[:2]
        h = int(h/2)
        w = int(w/2)

    mask = np.zeros((img.shape[0], img.shape[0]), dtype=np.uint8)
    cv2.circle(mask, (w, h), occulter_size, 1, -1)

    if repleace_value is None:
        img[mask == 1] = np.mean(img)
    else:
        img[mask == 1] = repleace_value
    return img

def radius_to_px(plotranges, imsize, headers, sat):
    x = np.linspace(plotranges[0], plotranges[1], num=imsize[0])
    y = np.linspace(plotranges[2], plotranges[3], num=imsize[1])
    xx, yy = np.meshgrid(x, y)
    x_cS, y_cS = center_rSun_pixel(headers, plotranges, sat)
    return np.sqrt((xx - x_cS)**2 + (yy - y_cS)**2)

def center_rSun_pixel(headers, plotranges, sat):
    '''
    Gets the location of Suncenter in deg
    '''
    x_cS = (headers['CRPIX1']*plotranges[sat]*2) / \
        headers['NAXIS1'] - plotranges[sat]  # headers['CRPIX1']
    y_cS = (headers['CRPIX2']*plotranges[sat]*2) / \
        headers['NAXIS2'] - plotranges[sat]  # headers['CRPIX2']
    return x_cS, y_cS

def run_training(model, cme_train_dataloader, cme_test_dataloader, batch_size, epochs, opath, par_loss_weights, save_model):
    train_losses_per_batch = []
    median_train_losses_per_batch = []
    test_losses_per_batch = []
    median_test_error_in_batch = []
    epoch_list = []
    total_batches_per_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_onlyepoch_losses = []
        test_onlyepoch_losses = []
        # Store the total number of batches processed
        epoch_list.append(total_batches_per_epoch)
        total_batches_per_epoch = 0
        for i, (img, targets, sat_masks, occulter_masks, satpos, plotranges, idx) in enumerate(cme_train_dataloader, 0):
            total_batches_per_epoch += 1
            loss_value = model.optimize_model(img, targets, par_loss_weights)
            train_losses_per_batch.append(loss_value.detach().cpu())
            train_onlyepoch_losses.append(loss_value.detach().cpu())

            if i % 10 == 0:
                logging.info(
                    f'Epoch: {epoch + 1}, Image: {(i + 1) * batch_size}, Batch: {i + 1}, Loss: {loss_value:.5f}, learning rate: {model.optimizer.param_groups[-1]["lr"]:.7f}')

            if i % 50 == 0:
                model.plot_loss(train_losses_per_batch, epoch_list, batch_size, os.path.join(
                    opath, "train_loss.png"), plot_epoch=False)

        median_train_losses_per_batch.append(np.median(train_onlyepoch_losses))

        # Test
        model.eval()
        with torch.no_grad():
            for i, (img, targets, sat_masks, occulter_masks, satpos, plotranges, idx) in enumerate(cme_test_dataloader, 0):
                loss_test = model.test_model(img, targets, par_loss_weights)
                test_losses_per_batch.append(loss_test.detach().cpu())
                test_onlyepoch_losses.append(loss_test.detach().cpu())
            median_test_error_in_batch.append(np.median(test_onlyepoch_losses))

        logging.info(f'Epoch: {epoch + 1}, Test Loss: {loss_test:.5f}\n')
        model.plot_loss(test_losses_per_batch, epoch_list, batch_size, os.path.join(
            opath, "test_loss.png"), plot_epoch=False)

        # Plot mean loss per epoch
        model.plot_loss(median_train_losses_per_batch, epoch_list, batch_size, os.path.join(
            opath, "mean_train_loss.png"), plot_epoch=False, medianLoss=True)
        model.plot_loss(median_test_error_in_batch, epoch_list, batch_size, os.path.join(
            opath, "mean_test_loss.png"), plot_epoch=False, medianLoss=True)

        # Save model
        if save_model:
            status = model.save_model(opath)
            logging.info(f"Model saved at: {status}\n")

def main():
    # Configuración de parámetros
    configuration = Configuration(Path(
        "/gehme-gpu/projects/2020_gcs_with_ml/repo_mariano/2020_gcs_with_ml/nn/neural_gcs/sirats_config/sirats_inception_run6.ini"))

    TRAINDIR = configuration.train_dir
    OPATH = configuration.opath
    BATCH_SIZE = configuration.batch_size
    BATCH_LIMIT = configuration.batch_limit
    SEED = configuration.rnd_seed
    IMG_SIZE = configuration.img_size
    DEVICE = configuration.device
    ONLY_MASK = configuration.only_mask
    DO_TRAINING = configuration.do_training
    DO_INFERENCE = configuration.do_inference
    REAL_IMG_INFERENCE = configuration.real_img_inference
    IMAGES_TO_INFER = configuration.images_to_infer
    MODEL_ARQ = configuration.model_arq
    SAVE_MODEL = configuration.save_model
    LOAD_MODEL = configuration.load_model
    EPOCHS = configuration.epochs
    TRAIN_IDX_PERCENT = configuration.train_index_percent
    LR = configuration.lr
    PAR_RNG = configuration.par_rng
    PAR_LOSS_WEIGHTS = configuration.par_loss_weight
    os.makedirs(OPATH, exist_ok=True)

    # Logging configuration
    LOGF_PATH = os.path.join(OPATH, 'sirats.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(funcName)-5s: %(levelname)-s, %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=LOGF_PATH,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(
        '%(asctime)s  %(funcName)-5s: %(levelname)-s, %(message)s', datefmt='%m-%d %H:%M')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Cargar y procesar datos
    dataset = Cme_MVP_Dataset(root_dir=TRAINDIR,
                              img_size=IMG_SIZE,
                              only_mask=ONLY_MASK)
    
    # Instanciar plotter

    sirats_plotter = SiratsPlotter()
    
    random.seed(SEED)
    total_samples = len(dataset)
    train_size = int((dataset.len_csv - 1) * TRAIN_IDX_PERCENT)
    train_indices = random.sample(range(total_samples), train_size)
    test_indices = list(set(range(total_samples)) - set(train_indices))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    cme_train_dataloader = DataLoader(train_dataset,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
    cme_test_dataloader = DataLoader(test_dataset,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True)
    # Configurar el modelo
    if MODEL_ARQ == 'inception':
        model = SiratsInception(device=DEVICE,
                                output_size=6,
                                img_shape=IMG_SIZE,
                                loss_weights=PAR_LOSS_WEIGHTS)
    elif MODEL_ARQ == 'distribution':
        model = SiratsDistribution(device=DEVICE,
                                   output_size=6,
                                   img_shape=IMG_SIZE,
                                   loss_weights=PAR_LOSS_WEIGHTS)

    # Configurar optimizer, loss function y scheduler
    optimizer = torch.optim.Adadelta(model.parameters())
    scheduler = None
    loss_fn = None

    # Setear optimizer, loss function y scheduler al modelo
    model.set_optimizer(optimizer)
    model.set_loss_fn(loss_fn)
    model.set_scheduler(scheduler)

    # Cargar o inicializar el modelo
    if LOAD_MODEL:
        load_model(model, os.path.join(OPATH, 'models'))

    num_parameters = sum(p.numel() for p in model.parameters())
    logging.info(f'Number of parameters: {num_parameters}\n')

    # Ejecutar entrenamiento o inferencia
    if DO_TRAINING:
        run_training(model, cme_train_dataloader, cme_test_dataloader, BATCH_SIZE, EPOCHS, OPATH, PAR_LOSS_WEIGHTS,
                     SAVE_MODEL)

    if DO_INFERENCE:
        errorVP1 = []
        errorVP2 = []
        errorVP3 = []

        if not REAL_IMG_INFERENCE:
            img_counter = 0
            stop_flag = False
            for iteration, (img, targets, sat_masks, occulter_masks, satpos, plotranges, idx) in enumerate(cme_test_dataloader, 0):
                img = img.to(DEVICE)
                predictions = model.infer(img)

                if MODEL_ARQ == 'distribution':
                    predictions = predictions.mean()

                for i in range(BATCH_SIZE):
                    img_counter += 1
                    logging.info(
                        f"Plotting image {img_counter} of {IMAGES_TO_INFER}")
                    error = sirats_plotter.plot_mask_MVP(img[i], sat_masks[i], targets[i], predictions[i],
                                          occulter_masks[i], satpos[i], plotranges[i], OPATH, f'img_{img_counter}.png')
                    errorVP1.append(error[0])
                    errorVP2.append(error[1])
                    errorVP3.append(error[2])
                    if img_counter == IMAGES_TO_INFER:
                        stop_flag = True
                        break
                if stop_flag:
                    break

            errors = [errorVP1, errorVP2, errorVP3]
            logging.info("Plotting histogram")
            sirats_plotter.plot_histogram(errors, OPATH, 'histogram.png')

            # Save errors in a pickle file
            with open(os.path.join(OPATH, 'errors.pkl'), 'wb') as f:
                pickle.dump(errors, f)
                f.close()

        else:
            # Get events
            events = get_paths_cme_exp_sources(dates=['20130209'])
            for ev in events:
                # ev_date = ev['date'].split('/')[-1].split('_')[1]
                case_counter = -1 
                for case in range(len(ev['ima1'])):
                    case_counter += 1
                    # Get event images
                    ima = fits.getdata(ev['ima1'][case]) - fits.getdata(ev['ima0'][case])
                    imb = fits.getdata(ev['imb1'][case]) - fits.getdata(ev['imb0'][case])
                    lasco = fits.getdata(ev['lasco1'][case]) - fits.getdata(ev['lasco0'][case])

                    filename = os.path.basename(ev['ima1'][case]).replace('.fts', '')

                    # Get event headers
                    event_headers = []
                    event_headers.append(fits.getheader(ev['imb1'][case]))
                    event_headers.append(fits.getheader(ev['ima1'][case]))
                    event_headers.append(fits.getheader(ev['lasco1'][case]))

                    satpos, plotranges = pyGCS.processHeaders(event_headers)

                    # synth_img_path = '/gehme-gpu/projects/2020_gcs_with_ml/data/gcs_ml_3VP_size_100000_seed_72430'
                    # synth_img_list = os.listdir(synth_img_path)
                    # random_img = random.choice(synth_img_list)
                    # synth_img_list = os.listdir(os.path.join(synth_img_path, random_img))
                    # for img in synth_img_list:
                    #     if img.find('sat1') != -1:
                    #         ima = read_image(os.path.join(synth_img_path, random_img, img), mode=torchvision.io.image.ImageReadMode.GRAY)
                    #         ima = ima.squeeze(0)
                    #     if img.find('sat2') != -1:
                    #         imb = read_image(os.path.join(synth_img_path, random_img, img), mode=torchvision.io.image.ImageReadMode.GRAY)
                    #         imb = imb.squeeze(0)
                    #     if img.find('sat3') != -1:
                    #         lasco = read_image(os.path.join(synth_img_path, random_img, img), mode=torchvision.io.image.ImageReadMode.GRAY)
                    #         lasco = lasco.squeeze(0)

                    case_list = [imb, ima, lasco]

                    # make events as tensors
                    case_list = [torch.tensor(case, dtype=torch.float32)
                                  for case in case_list]

                    # Add occulter to images
                    center_idxs = [
                        ((case.shape[0] - 1) // 2, (case.shape[1] - 1) // 2) for case in case_list]
                    # Occulters size for [sat1, sat2 ,sat3] in [Rsun]
                    occulter_size = [2., 2., 4.3]
                    # occ_center=[(30,-15),(0,-5),(0,0)] # [(38,-15),(0,-5),(0,0)] # (y,x)
                    r_values = [radius_to_px(
                        plotranges[i], case_list[i].shape, event_headers[i], i) for i in range(len(occulter_size))]
                    # case_list = [add_occulter(case, occulter_size[i], center_idxs[i]) for i, case in enumerate(case_list)]
                    for i in range(len(case_list)):
                        case = case_list[i]
                        case[r_values[i] <= occulter_size[i]/2] = 0
                        case_list[i] = case

                    # Normalize event images
                    case_list = [real_img_normalization(case) for case in case_list]

                    # Resize event images
                    if case_list[0].shape[0] != IMG_SIZE[0] or case_list[0].shape[1] != IMG_SIZE[1]:
                        resize = torchvision.transforms.Resize(
                            IMG_SIZE[1:3], torchvision.transforms.InterpolationMode.BILINEAR)
                        resize_scale_factor = [
                            case_list[i].shape[1] / IMG_SIZE[1] for i in range(len(case_list))]
                        case_list = [resize(case.unsqueeze(0)) for case in case_list]
                        case_list = [case.squeeze(0) for case in case_list]
                        for i in range(len(event_headers)):
                            h = event_headers[i]
                            h['CDELT1'] = resize_scale_factor[i] * h['CDELT1']
                            h['CDELT2'] = resize_scale_factor[i] * h['CDELT2']
                            event_headers[i] = h

                        satpos, plotranges = pyGCS.processHeaders(event_headers)

                    # join event images
                    event_img = torch.stack(case_list, dim=0)

                    # Add batch dimension
                    event_img = event_img.unsqueeze(0)

                    # Move event images to device
                    event_img = event_img.to(DEVICE)

                    # Infer event images and save losses
                    predictions = model.infer(event_img)
                    fixed_satpos = "[[32.8937181611, 7.05123478188, 0.0], [300.081940747, 1.95463511752, 0.0], [274.2910293847356, -4.817368115630504, 0.0]]"
                    fixed_plotranges = "[[-16.6431925965469, 16.737728407985518, -16.84856349725838, 16.53235750727404], [-15.00659312775248, 15.050622251843686, -14.988981478115997, 15.068233901480168], [-6.338799715536909, 6.304081179329522, -6.388457593426707, 6.254423301439724]]"

                    fixed_satpos = torch.tensor(
                        eval(fixed_satpos), dtype=torch.float32)
                    fixed_plotranges = torch.tensor(
                        eval(fixed_plotranges), dtype=torch.float32)

                    # Plot infered masks
                    sirats_plotter.plot_real_infer(event_img, predictions, satpos, plotranges, OPATH,
                                    f'{filename}.png', fixed_satpos=fixed_satpos, fixed_plotranges=fixed_plotranges, use_fixed=True)

if __name__ == '__main__':
    main()
