import pickle
import matplotlib.pyplot as plt

training_loss_file = "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_fran/all_loss"
with open(training_loss_file, 'rb') as file:
    # Load the pickled object
    training_loss = pickle.load(file)
plt.plot(training_loss)
plt.yscale("log")
plt.show()