import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from datetime import datetime, timedelta



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
   df_analyzed=pd.DataFrame(columns=["paths","date","mean","std","contraste"])
   files= os.listdir(path)

   for i in files:
      file_path=path+"/"+i
      if sat=="cor2_a" or sat=="cor2_b":
         formato = '%Y%m%d_%H%M%S'
         date = datetime.strptime(i[0:-10], formato)

      if os.path.exists(file_path):
         breakpoint()
         img = fits.open(file_path)
         data=img[0].data
         mean= np.mean(data)
         std = np.std(data)
         contraste=std/mean
         df_analyzed = df_analyzed.append({"paths": file_path,"date":date, "mean": mean, "std": std, "contraste":contraste}, ignore_index=True)
      else:
         print(i,"does not exist")
   df_analyzed.to_csv(exec_path+"/"+sat+"_path_list_analyzed.csv", index=False)
   return df_analyzed




def filtered(sat):
   df= pd.read_csv(exec_path+"/"+sat+"_path_list_analyzed.csv", sep=",")
   if sat=="cor2_a" or sat=="cor2_b":
      df['date'] = pd.to_datetime(df['date'])
      df_sorted = df.sort_values('date')

      resultados = []

      # Iterar sobre cada fecha en la columna 'date'
      for i, date in enumerate(df_sorted['date']):
         
         # Calcular la fecha límite 12 horas antes de la fecha actual
         prev_date = date - timedelta(hours=24)
         #for i in range(len(df["date"])):
         if not(df.loc[i,"date"]<date and df.loc[i,"date"]>=prev_date):
            resultados.append((df.loc[i,"paths"],i))

      
      m=0
      for lista in resultados:
         path=lista[0]
         index=lista[1]
         print("image "+str(m)+" of "+str(len(resultados)))
         img = fits.open(path)
         data=img[0].data
         header=img[0].header
         vmin = df.loc[index,"mean"]-3*(df.loc[index,"std"])
         vmax = df.loc[index,"mean"]+3*(df.loc[index,"std"])
         
         fig0, ax0 = plt.subplots()
         imagen = ax0.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
         filename = os.path.basename(df.loc[index,"paths"])
         filename = os.path.splitext(filename)[0]
         m=m+1
         fits_img = fits.PrimaryHDU(data, header=header)
         plt.savefig(cor2_opath+"/images/"+"/"+sat+"/"+filename+".png", format='png')
         fits_img.writeto(cor2_opath+"/images/"+"/"+sat+"/"+filename+".fits",overwrite=True) 
         img.close()
                    
         
         

   elif sat=="lasco_c2":
      print("lasco")
     

      
   

#df=process(data_path,sat)
filtered(sat)













# for i in range(len(df["contraste"])):
#    if df.loc[i,"contraste"]<500:
#       im= fits.open(df.loc[i,"paths"])
#       img=im[0].data
#       vmin = df.loc[i,"mean"]-3*(df.loc[i,"std"])
#       vmax = df.loc[i,"mean"]+3*(df.loc[i,"std"])
#       fig0, ax0 = plt.subplots()
#       imagen = ax0.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
#       filename = os.path.basename(df.loc[i,"paths"])
#       filename = os.path.splitext(filename)[0]
#       plt.savefig(cor2_opath+"/images/"+filename+".png", format='png')
#       im.close()

#graphics mean, std an contrast
# fig1, ax1 = plt.subplots()
# ax1.plot(np.abs(df["mean"]), '.k')
# ax1.set_title('Gráfico de Media')
# ax1.set_yscale('log')
# fig1.savefig(cor2_opath+"/plots/"+sat+'_mean_plot_analyzed.png')

# fig2, ax2 = plt.subplots()
# ax2.hist(np.abs(df["mean"]), bins=50, color='k', alpha=0.7)
# ax2.set_title('Histograma de Media')
# ax2.set_yscale('log')
# fig2.savefig(cor2_opath+"/plots/"+sat+'_mean_hist_analyzed.png')

# fig3, ax3 = plt.subplots()
# ax3.plot(np.abs(df["contraste"]), '.k')
# ax3.set_title('Contrast')
# ax3.set_yscale('log')
# fig3.savefig(cor2_opath+"/plots/"+sat+'_contrast_plot_analyzed.png')

# fig4, ax4 = plt.subplots()
# ax4.hist(np.abs(df["contraste"]), bins=50, color='k', alpha=0.7)
# ax4.set_title('Contrast')
# ax4.set_yscale('log')
# fig4.savefig(cor2_opath+"/plots/"+sat+'_contrast_hist_analyzed.png')

# fig3, ax3 = plt.subplots()
# ax3.plot(np.abs(df["std"]), '.k')
# ax3.set_title('Std')
# ax3.set_yscale('log')
# fig3.savefig(cor2_opath+"/plots/"+sat+'_std_plot_analyzed.png')

# fig4, ax4 = plt.subplots()
# ax4.hist(np.abs(df["std"]), bins=50, color='k', alpha=0.7)
# ax4.set_title('Std')
# ax4.set_yscale('log')
# fig4.savefig(cor2_opath+"/plots/"+sat+'_std_hist_analyzed.png')
      










