import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from ext_libs.rebin import rebin
from astropy.io import fits
import pandas as pd
import glob
"""
Reads pairs of LVL1 coronograph images from various instruments and saves a differential corona for each pair.
Images are resized
"""

exec_path = os.getcwd()+"/catalogues"
lasco_path = exec_path+'/Lista_Final_CMEs_downloads_lascoc2.csv' #'/catalogues/Lista_Final_CMEs.csv' # file with the list of cor files
cor2_path = exec_path+'/Lista_Final_CMEs_downloads_cor2.csv'
lasco_downloads=["pre_a_1h_download_c2","pre_b_1h_download_c2","pre_a_2h_download_c2","pre_b_2h_download_c2"]
cor2_downloads=["pre_a_1h_download","pre_b_1h_download","pre_a_2h_download","pre_b_2h_download"]
imsize=[512,512]


lasco= pd.read_csv(lasco_path , sep="\t")
lasco.name='lasco'

cor2= pd.read_csv(cor2_path , sep="\t")
cor2.name='cor2'
for i in range(len(cor2.index)):
    for j in cor2_downloads:
        if cor2.loc[i,j] != "No data" and cor2.loc[i,j] != "*" and cor2.loc[i,j] != "No img/double data":
            cor2.at[i, j] = "/gehme/data/stereo/secchi/"+ cor2.at[i, j]
            if j== cor2_downloads[0] or j==cor2_downloads[2]:
                cor2.at[i,j]=cor2.at[i,j].replace("A","a")
            else:  
                cor2.at[i,j]=cor2.at[i,j].replace("B","b")

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

        for k in range(0,2):        #repetir para evento b
            if (df.loc[i,downloads[k]] != "No data" and df.loc[i,downloads[k]] != "*" and df.loc[i,downloads[k]] != "No img/double data") and (df.loc[i,downloads[k+2]] != "No data" and df.loc[i,downloads[k+2]] != "*" and df.loc[i,downloads[k+2]] != "No img/double data"): 
                try:
                        file1=glob.glob(df.loc[i,downloads[k]][0:-5]+"*")
                        file2=glob.glob(df.loc[i,downloads[k+2]][0:-5]+"*")
                        path_1h=(file1[0])#.replace("level_05","level_1")
                        path_2h=(file2[0])#.replace("level_05","level_1")
                        im1= fits.open(path_1h)
                        im2= fits.open(path_2h)
                        image1 = rebin(im1[0].data,imsize,operation='mean') 
                        image2 = rebin(im2[0].data,imsize,operation='mean') 
                        im=image1-image2
                        header= fits.getheader(path_1h)
                        header['NAXIS1'] = imsize[0]   
                        header['NAXIS2'] = imsize[1]
                        final_img = fits.PrimaryHDU(im, header=header[0:-3])
                        final_img.writeto('/gehme/projects/2020_gcs_with_ml/data/corona_back_database/'+df.name+'/'+os.path.basename(path_1h),overwrite=True) 


                except ValueError as e:
                    print("error "+str(e)+" on "+path_1h+"  or  "+path_2h)

    paths=pd.DataFrame(paths,columns=['paths'])
    paths = paths.drop_duplicates()
    paths.to_csv(exec_path+"/"+name+"_path_list.csv", index=False)



pathlist(cor2)





