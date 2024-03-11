import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import numpy as np
import matplotlib.pyplot as plt
from nn_training.corona_background.get_corona_gcs_ml import get_corona
from datetime import datetime


def plot_3VP(imgs_data, dates, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, img in enumerate(imgs_data):
        mean = np.mean(img)
        std = np.std(img)
        axes[i].imshow(img, cmap='gray', vmin=mean-3*std, vmax=mean+3*std)
        axes[i].axis('off')
        axes[i].set_title(dates[i])  # Add date as title
    plt.savefig(path)
    plt.close(fig)


def main():
    imsize=np.array([512, 512], dtype='int32')
    opath = "/gehme/projects/2020_gcs_with_ml/data/corona_background_affects/3VP_backgrounds"

    for i in range(20):
        # Get the corona images
        imgs_data, headers, size_occ, dates = get_corona()

        # Plot the 3VP images
        plot_3VP(imgs_data, dates, os.path.join(opath, f"3VP_backgrounds_{i}.png"))


if __name__ == "__main__":
    main()