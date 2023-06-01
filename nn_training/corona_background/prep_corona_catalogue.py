import pandas as pd
import os
from astropy.io import fits

"""
Reads pairs of LVL1 coronograph images from various instruments and saves a differential corona for each pair.
Images are resized
"""

exec_path = os.getcwd()+"/catalogues"
path=exec_path+'/Lista_Final_CMEs_downloads_lascoc2.csv' #'/catalogues/Lista_Final_CMEs.csv' # file with the list of cor files
downloads=["pre_a_1h_download_c2",	"pre_b_1h_download_c2",	"pre_a_2h_download_c2",	"pre_b_2h_download_c2"]
df= pd.read_csv(path , sep="\t")

# convertir las columnas de fecha y hora en objetos de fecha y hora de Pandas
df['evento_a'] = pd.to_datetime(df['date_a'] + ' ' + df['time_a'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df['evento_b'] = pd.to_datetime(df['date_b'] + ' ' + df['time_b'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# crear una nueva columna con la fecha y hora concatenadas
#df['fecha_hora_a'] = df['fecha_hora_a'].dt.strftime('%Y-%m-%d %H:%M:%S')
#df['fecha_hora_b'] = df['fecha_hora_b'].dt.strftime('%Y-%m-%d %H:%M:%S')

# crear columnas nuevas con las horas restadas
df['preevento_a_1h'] = (df['evento_a'] - pd.to_timedelta('1 hour')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
df['preevento_b_1h'] = (df['evento_b'] - pd.to_timedelta('1 hour')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')

df['preevento_a_3h'] = (df['evento_a'] - pd.to_timedelta('2 hours')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
df['preevento_b_3h'] = (df['evento_b'] - pd.to_timedelta('2 hours')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')


df.to_csv(exec_path+'/Lista_Final_CMEs_downloads_lascoc2.csv',sep='\t', header=True,index=False)


paths=[]
for i in range(len(df.index)):
    for j in downloads:
        if df.loc[i,j] != "No data" and df.loc[i,j] != "*":
            element=df.loc[i,j]
            paths.append(element)


    for k in range(0,2):        #repetir para evento b
        if (df.loc[i,downloads[k]] != "No data" and df.loc[i,downloads[0]] != "*") and (df.loc[i,downloads[k+2]] != "No data" and df.loc[i,downloads[2]] != "*"):
            path_1h=(df.loc[i,downloads[k]]).replace("level_05","level_1")
            path_2h=(df.loc[i,downloads[k+2]]).replace("level_05","level_1")
            im1= fits.open(path_1h[0])
            im2= fits.open(path_2h[0])
            im=im1-im2
            header= fits.getheader(path_1h)
            final_img = fits.PrimaryHDU(im, header=header)
            final_img.writeto('gehme/projects/2020_gcs_with_ml/data/corona_back_database'+path_1h)
   


paths=pd.DataFrame(paths,columns=['paths'])
paths.to_csv(exec_path+'/path_list.csv',sep='\t', header=True,index=False)



