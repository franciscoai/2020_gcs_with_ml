import pandas as pd
import os
from astropy.io import fits

"""
Reads pairs of LVL1 coronograph images from various instruments and saves a differential corona for each pair.
Images are resized
"""

exec_path = os.getcwd()+"/catalogues"
lasco_path = exec_path+'/Lista_Final_CMEs_downloads_lascoc2.csv' #'/catalogues/Lista_Final_CMEs.csv' # file with the list of cor files
cor2_path = exec_path+'/Lista_Final_CMEs_downloads_cor2.csv'
lasco_downloads=["pre_a_1h_download_c2","pre_b_1h_download_c2","pre_a_2h_download_c2","pre_b_2h_download_c2"]
cor2_downloads=["pre_a_1h_download","pre_b_1h_download","pre_a_2h_download","pre_b_2h_download"]

lasco= pd.read_csv(lasco_path , sep="\t")
lasco.name='lasco'
cor2= pd.read_csv(cor2_path , sep="\t")
cor2.name='cor2'

def pathlist(df):
    paths=[]
    if df.name=='cor2':
        downloads=cor2_downloads
    else:
        downloads=lasco_downloads
    for i in range(len(df.index)):
        for j in downloads:
            if df.loc[i,j] != "No data" and df.loc[i,j] != "*":
                element= df.loc[i,j]
                paths.append(element)  

    paths=pd.DataFrame(paths,columns=['paths'])
    paths.to_csv(exec_path+'/'+df.name+'_path_list.csv',sep='\t', header=True,index=False)







# convertir las columnas de fecha y hora en objetos de fecha y hora de Pandas
lasco['evento_a'] = pd.to_datetime(lasco['date_a'] + ' ' + lasco['time_a'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
lasco['evento_b'] = pd.to_datetime(lasco['date_b'] + ' ' + lasco['time_b'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# crear una nueva columna con la fecha y hora concatenadas
#lasco['fecha_hora_a'] = lasco['fecha_hora_a'].dt.strftime('%Y-%m-%d %H:%M:%S')
#lasco['fecha_hora_b'] = lasco['fecha_hora_b'].dt.strftime('%Y-%m-%d %H:%M:%S')

# crear columnas nuevas con las horas restadas
lasco['preevento_a_1h'] = (lasco['evento_a'] - pd.to_timedelta('1 hour')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
lasco['preevento_b_1h'] = (lasco['evento_b'] - pd.to_timedelta('1 hour')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')

lasco['preevento_a_3h'] = (lasco['evento_a'] - pd.to_timedelta('2 hours')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
lasco['preevento_b_3h'] = (lasco['evento_b'] - pd.to_timedelta('2 hours')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')


lasco.to_csv(exec_path+'/Lista_Final_CMEs_downloads_lascoc2.csv',sep='\t', header=True,index=False)

pathlist(lasco)





# paths=[]
# for i in range(len(lasco.index)):
#     for j in lasco_downloads:
#         if lasco.loc[i,j] != "No data" and lasco.loc[i,j] != "*":
#             element=lasco.loc[i,j]
#             paths.append(element)


    # for k in range(0,2):        #repetir para evento b
    #     if (lasco.loc[i,lasco_downloads[k]] != "No data" and lasco.loc[i,lasco_downloads[0]] != "*") and (lasco.loc[i,lasco_downloads[k+2]] != "No data" and lasco.loc[i,lasco_downloads[2]] != "*"):
    #         path_1h=(lasco.loc[i,lasco_downloads[k]]).replace("level_05","level_1")
    #         path_2h=(lasco.loc[i,lasco_downloads[k+2]]).replace("level_05","level_1")
    #         im1= fits.open(path_1h[0])
    #         im2= fits.open(path_2h[0])
    #         im=im1-im2
    #         header= fits.getheader(path_1h)
    #         final_img = fits.PrimaryHDU(im, header=header)
    #         final_img.writeto('gehme/projects/2020_gcs_with_ml/data/corona_back_database'+path_1h)
   


# paths=pd.DataFrame(paths,columns=['paths'])
# paths.to_csv(exec_path+'/path_list.csv',sep='\t', header=True,index=False)



