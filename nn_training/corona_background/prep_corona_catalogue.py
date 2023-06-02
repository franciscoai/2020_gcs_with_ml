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
    name=df.name
    if name=='cor2':
        downloads=cor2_downloads
    else:
        downloads=lasco_downloads
    for i in range(len(df.index)):
        for j in downloads:
            if df.loc[i,j] != "No data" and df.loc[i,j] != "*" and df.loc[i,j] != "No img/double data":
                element= df.loc[i,j]
                paths.append(element)

        # for k in range(0,2):        #repetir para evento b
        #     if (df.loc[i,downloads[k]] != "No data" and df.loc[i,downloads[0]] != "*") and (df.loc[i,downloads[k+2]] != "No data" and df.loc[i,downloads[2]] != "*"):
        #         path_1h=(df.loc[i,downloads[k]])#.replace("level_05","level_1")
        #         path_2h=(df.loc[i,downloads[k+2]])#.replace("level_05","level_1")
        #         im1= fits.open(path_1h)
        #         im2= fits.open(path_2h)
        #         im=im1[0].data-im2[0].data
        #         header= fits.getheader(path_1h)
        #         final_img = fits.PrimaryHDU(im, header=header[0:-3])

        #         final_img.writeto('gehme/projects/2020_gcs_with_ml/data/corona_back_database/'+df.name+'/'+os.path.basename(path_1h)) 

    paths=pd.DataFrame(paths,columns=['paths'])
    paths = paths.drop_duplicates()
    paths.to_csv(exec_path+'/'+df.name+'_path_list.csv',sep='\t', header=True,index=False)



pathlist(cor2)





