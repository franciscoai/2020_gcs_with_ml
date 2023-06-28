import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from ext_libs.rebin import rebin
from astropy.io import fits
import pandas as pd
import glob
from datetime import datetime, timedelta


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
do_write=True




lasco= pd.read_csv(lasco_path , sep="\t")
lasco.name='lasco'

cor2= pd.read_csv(cor2_path , sep="\t")
cor2.name='cor2'
for i in range(len(cor2.index)):
    for j in cor2_downloads:
        if cor2.loc[i,j] != "No data" and cor2.loc[i,j] != "*" and cor2.loc[i,j] != "No img/double data":
            cor2.at[i, j] = "/gehme/data/stereo/secchi/"+ cor2.at[i, j]


def pathlist(df,column_list, do_write=True):
    '''
    df=dataframe
    column_list: list of columns to use 
    '''
    paths=[]
    name=df.name
    for i in range(len(df.index)):
        for k in range(0,2):        #repetir para evento b
            if (df.loc[i,column_list[k]] != "No data" and df.loc[i,column_list[k]] != "*" and df.loc[i,column_list[k]] != "No img/double data") and (df.loc[i,column_list[k+2]] != "No data" and df.loc[i,column_list[k+2]] != "*" and df.loc[i,column_list[k+2]] != "No img/double data"): 
                
                if name=="cor2":
                    file1=glob.glob(df.loc[i,column_list[k]][0:-5]+"*")
                    file2=glob.glob(df.loc[i,column_list[k+2]][0:-5]+"*")
                    formato = '%Y%m%d_%H%M%S'
                elif name=="lasco":
                    file1=glob.glob(df.loc[i,column_list[k]])
                    file2=glob.glob(df.loc[i,column_list[k+2]])
                paths.append(file1)
                paths.append(file2)
                

    
    paths=pd.DataFrame(paths,columns=['paths'])
    paths = paths.drop_duplicates()
    paths.to_csv(exec_path+"/"+name+"_path_list.csv", index=False)

    if name=="cor2":
        cor2_a=pd.DataFrame(columns=['paths',"date"])
        cor2_b=pd.DataFrame(columns=['paths',"date"])
        for i in paths["paths"]:
            if i.endswith("a.fts") or i.endswith("A.fts"):
                basename=os.path.basename(i)
                date = datetime.strptime(basename[0:-10], formato)
                cor2_a=cor2_a.append({"paths":i,"date":date}, ignore_index=True)
               
            elif i.endswith("b.fts") or i.endswith("B.fts"):
                basename=os.path.basename(i)
                date = datetime.strptime(basename[0:-10], formato)
                cor2_b=cor2_b.append({"paths":i,"date":date}, ignore_index=True)
               
        for i, date in enumerate(cor2_a["date"]):
            
            prev_date = date - timedelta(hours=12)
            count = ((cor2_a["date"] < date) & (cor2_a["date"] >= prev_date)).sum()
            if count==1:
                print(i)
                #generar imagen diferencia
                try:
                    if do_write==True:
                        file1=glob.glob(cor2_a.loc[i,"paths"][0:-5]+"*")
                        file2=glob.glob(cor2_a.loc[i+1,"paths"][0:-5]+"*")
                        
                        path_1h=(file1[0]).replace("L0","L1")
                        path_2h=(file2[0]).replace("L0","L1")
                        im1= fits.open(path_1h)
                        im2= fits.open(path_2h)
                        image1 = rebin(im1[0].data,imsize,operation='mean') 
                        image2 = rebin(im2[0].data,imsize,operation='mean') 
                        im=image1-image2
                        header= fits.getheader(path_1h)
                        header['NAXIS1'] = imsize[0]   
                        header['NAXIS2'] = imsize[1]
                        final_img = fits.PrimaryHDU(im, header=header[0:-3])
                        final_img.writeto('/gehme/projects/2020_gcs_with_ml/data/corona_back_database/'+df.name+'/'+"cor2_a"+"/"+os.path.basename(path_1h),overwrite=True)
                except ValueError as e:
                    print("error "+str(e)+" on "+path_1h+"  or  "+path_2h)

        for i, date in enumerate(cor2_b["date"]):
            prev_date = date - timedelta(hours=12)
            count = ((cor2_b["date"] < date) & (cor2_b["date"] >= prev_date)).sum()
            if count==1:
                #generar imagen diferencia
                try:
                    if do_write==True:
                        file1=glob.glob(cor2_b.loc[i,"paths"][0:-5]+"*")
                        file2=glob.glob(cor2_b.loc[i+1,"paths"][0:-5]+"*")
                        
                        path_1h=(file1[0]).replace("L0","L1")
                        path_2h=(file2[0]).replace("L0","L1")
                        im1= fits.open(path_1h)
                        im2= fits.open(path_2h)
                        image1 = rebin(im1[0].data,imsize,operation='mean') 
                        image2 = rebin(im2[0].data,imsize,operation='mean') 
                        im=image1-image2
                        header= fits.getheader(path_1h)
                        header['NAXIS1'] = imsize[0]   
                        header['NAXIS2'] = imsize[1]
                        final_img = fits.PrimaryHDU(im, header=header[0:-3])
                        final_img.writeto('/gehme/projects/2020_gcs_with_ml/data/corona_back_database/'+df.name+'/'+"cor2_b"+"/"+os.path.basename(path_1h),overwrite=True)                  
                except ValueError as e:
                    print("error "+str(e)+" on "+path_1h+"  or  "+path_2h)
                        

#function to create corona background 
pathlist(cor2,cor2_downloads,do_write=do_write) 




