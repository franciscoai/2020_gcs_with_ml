import os
import shutil

origin_path = "/gehme-gpu/projects/2020_gcs_wiht_ml/data/neural_cme_seg/LabPicsChemistry/Train"
output_path = "/gehme-gpu/projects/2020_gcs_wiht_ml/data/neural_cme_seg/LabPicsChemistry/Train_filtered"

# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

imgs = []
mask = []
folder = []

# Collect paths of image files and mask directories
for pth in os.listdir(origin_path):
    img_path = os.path.join(origin_path, pth, "Image.jpg")
    mask_path = os.path.join(origin_path, pth, "Vessels")
    imgs.append(img_path)
    mask.append(mask_path)
    
    # Create the corresponding directory in the output path
    out_folder = os.path.join(output_path, pth)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    folder.append(out_folder)

# Copy the files
for i in range(len(imgs)):
    shutil.copy2(imgs[i], os.path.join(folder[i], "Image.jpg"))
    mask_dst = os.path.join(folder[i], "Vessels")
    if os.path.exists(mask_dst):
        shutil.rmtree(mask_dst)
    shutil.copytree(mask[i], mask_dst)

print("all done...")