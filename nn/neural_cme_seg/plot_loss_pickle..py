
import pickle
import matplotlib.pyplot as plt

pkl_path="/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/all_loss"

with open(pkl_path, 'rb') as pkl_file:
     data= pickle.load(pkl_file)
fig, ax = plt.subplots()
ax.loglog(data)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)
fig.savefig(pkl_path+".png")