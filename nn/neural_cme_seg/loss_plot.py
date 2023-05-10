import pickle
import matplotlib.pyplot as plt

training_loss_file = "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_fran/all_loss"
testing_loss_file = "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_fran/test_loss"
with open(training_loss_file, 'rb') as file:
    training_loss = pickle.load(file)
with open(testing_loss_file, 'rb') as file:
    testing_loss = pickle.load(file)



fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(training_loss)  
axs[0].set_title('Training loss')
axs[0].set_yscale("log")
axs[1].hist(testing_loss,bins=50) 
axs[1].set_title('Testing loss')


plt.tight_layout()
plt.show()