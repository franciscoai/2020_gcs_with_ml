import os
from astropy.io import fits
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

data_path="/gehme/projects/2020_gcs_with_ml/data/corona_back_database"
exec_path ="/gehme/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn_training/corona_background/catalogues"
cor2_opath=data_path+"/cor2/analysis"

sat="cor2_a"#"cor2_b""lasco_c2"




def process(path,sat):
   if sat=="cor2_a":
      path=path+"/cor2/cor2_a"
   elif sat=="cor2_b":
      path=path+"/cor2/cor2_b"
   elif sat=="lasco_c2":
      path=path+"/lasco/c2"
   df_analyzed=pd.DataFrame(columns=["paths","mean","std","contraste"])
   files= os.listdir(path)
   for i in files:
      file_path=path+"/"+i
      if os.path.exists(file_path):
         img = fits.open(file_path)
         data=img[0].data
         mean= np.mean(data)
         std = np.std(data)
         contraste=std/mean
         df_analyzed = df_analyzed.append({"paths": file_path, "mean": mean, "std": std, "contraste":contraste}, ignore_index=True)
      else:
         print(i,"does not exist")
   df_analyzed.to_csv(exec_path+"/"+sat+"_path_list_analyzed.csv", index=False)
   return df_analyzed



#df=process(data_path,sat)


df= pd.read_csv(exec_path+"/"+sat+"_path_list_analyzed.csv", sep=",")

for i in range(len(df["contraste"])):
   if df.loc[i,"contraste"]>1:
      im= fits.open(df.loc[i,"paths"])
      img=im[0].data
      vmin = df.loc[i,"mean"]-3*(df.loc[i,"std"])
      vmax = df.loc[i,"mean"]+3*(df.loc[i,"std"])
      fig0, ax0 = plt.subplots()
      imagen = ax0.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
      filename = os.path.basename(df.loc[i,"paths"])
      filename = os.path.splitext(filename)[0]
      plt.savefig(cor2_opath+"/images/"+filename+".png", format='png')
      im.close()

#grafica la media y el contaste
fig1, ax1 = plt.subplots()
ax1.plot(df["mean"], '.k')
ax1.set_title('Gráfico de Media')
fig1.savefig(cor2_opath+"/plots/"+sat+'_mean_plot_analyzed.png')

fig2, ax2 = plt.subplots()
ax2.hist(df["mean"], bins=50, color='k', alpha=0.7)
ax2.set_title('Histograma de Media')
ax2.set_yscale('log')
fig2.savefig(cor2_opath+"/plots/"+sat+'_mean_hist_analyzed.png')

fig3, ax3 = plt.subplots()
ax3.plot(df["contraste"], '.k')
ax3.set_title('Std/Media')
fig3.savefig(cor2_opath+"/plots/"+sat+'_std_plot_analyzed.png')

fig4, ax4 = plt.subplots()
ax4.hist(df["contraste"], bins=50, color='k', alpha=0.7)
ax4.set_title('Std/Media')
ax4.set_yscale('log')
fig4.savefig(cor2_opath+"/plots/"+sat+'_std_hist_analyzed.png')
      










