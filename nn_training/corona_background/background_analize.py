import os
from astropy.io import fits
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

exec_path = os.getcwd()+"/catalogues"
cor2_path=exec_path+"/"+"cor2_path_list.csv"
lasco_path=exec_path+"/"+"lasco_path_list.csv"
sat="lasco"#"cor2"

def process(path,sat):
   df= pd.read_csv(path , sep="\t")
   df_analyzed=pd.DataFrame(columns=["paths","mean","std"])
   for i in df["paths"]:
      if os.path.exists(i):
         img = fits.open(i)
         data=img[0].data
         mean= np.mean(data)
         std = np.std(data)
         df_analyzed = df_analyzed.append({"paths": i, "mean": mean, "std": std}, ignore_index=True)
      else:
         print(i,"does not exist")
   df_analyzed.to_csv(exec_path+"/"+sat+"_path_list_analyzed.csv", index=False)
   return df_analyzed

df=process(cor2_path,sat)
x = range(len(df))
plt.errorbar(x, df['mean'], yerr=df['std'], fmt='o', capsize=4)
plt.title('Media y Desviación Estándar')
plt.xticks(x)
plt.show()


      