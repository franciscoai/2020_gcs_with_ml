import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from ext_libs.rebin import rebin
from astropy.io import fits
import pandas as pd
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np


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
do_write=True # if set to True it saves the diff images




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
        
        cor2_a=pd.DataFrame(columns=['paths',"date"])#,"header_contrast"])
        cor2_b=pd.DataFrame(columns=['paths',"date"])#,"header_contrast"])
        
        for i in paths["paths"]:
            i=i.replace("L0","L1")
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
                
                #generar imagen diferencia
                try:
                    print("reading image "+str(i)+ " on cor2_a")
                    file1=glob.glob(cor2_a.loc[i,"paths"][0:-10]+"*")
                    file2=glob.glob(cor2_a.loc[i+1,"paths"][0:-10]+"*")
                    if len(file1)!=0 or len(file2)!=0:
                        
                        path_1h=(file1[0])
                        
                        path_2h=(file2[0])
                        
                        im1= fits.open(path_1h)
                        im2= fits.open(path_2h)
                        
                        image1 = rebin(im1[0].data,imsize,operation='mean') 
                        image2 = rebin(im2[0].data,imsize,operation='mean') 
                        im=image1-image2
                        header= fits.getheader(path_1h)
                        header['NAXIS1'] = imsize[0]   
                        header['NAXIS2'] = imsize[1]
                        sigma=header["DATASIG"]
                        avg=header["DATAAVG"]
                        header_contrast= sigma/avg
                        
                        cor2_a.loc[i,"header_contrast"]=header_contrast                           
                        final_img = fits.PrimaryHDU(im, header=header[0:-3])
                        filename = os.path.basename(path_1h)
                        fig0, ax0 = plt.subplots()
                        mean= np.mean(im)
                        std = np.std(im)
                        vmin=mean-3*std
                        vmax=mean+3*std
                        imagen = ax0.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
                        if do_write==True:
                            if header_contrast<6.2:
                                print("saving image "+str(i)+" from cor2_a")
                                plt.savefig('/gehme/projects/2020_gcs_with_ml/data/corona_back_database/'+df.name+'/'+"cor2_a"+"/"+filename+".png", format='png')
                                final_img.writeto('/gehme/projects/2020_gcs_with_ml/data/corona_back_database/'+df.name+'/'+"cor2_a"+"/"+filename+".fits",overwrite=True)
                                im1.close()
                                im2.close()
                            else:
                                im1.close()
                                im2.close()
                    else:
                        print("path not found")
                            
                except:
                    print("error on "+path_1h+"  or  "+path_2h)

        for i, date in enumerate(cor2_b["date"]):
            prev_date = date - timedelta(hours=12)
            count = ((cor2_b["date"] < date) & (cor2_b["date"] >= prev_date)).sum()
            if count==1:
                #generar imagen diferencia
                try:
                    print("reading image "+str(i)+ " on cor2_b")
                    file1=glob.glob(cor2_b.loc[i,"paths"][0:-10]+"*")
                    file2=glob.glob(cor2_b.loc[i+1,"paths"][0:-10]+"*")
                    if len(file1)!=0 or len(file2)!=0:
                        path_1h=(file1[0])
                        path_2h=(file2[0])
                        im1= fits.open(path_1h)
                        im2= fits.open(path_2h)
                        image1 = rebin(im1[0].data,imsize,operation='mean') 
                        image2 = rebin(im2[0].data,imsize,operation='mean') 
                        im=image1-image2
                        header= fits.getheader(path_1h)
                        header['NAXIS1'] = imsize[0]   
                        header['NAXIS2'] = imsize[1]
                        final_img = fits.PrimaryHDU(im, header=header[0:-3])
                        sigma=header["DATASIG"]
                        avg=header["DATAAVG"]
                        header_contrast= sigma/avg
                        cor2_b.loc[i,"header_contrast"]=header_contrast   
                        filename = os.path.basename(path_1h)
                        fig0, ax0 = plt.subplots()
                        mean= np.mean(im)
                        std = np.std(im)
                        vmin=mean-3*std
                        vmax=mean+3*std
                        imagen = ax0.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
                        if do_write==True:    
                            if header_contrast<4.7:
                                print("saving image "+str(i)+" from cor2_b")
                                plt.savefig('/gehme/projects/2020_gcs_with_ml/data/corona_back_database/'+df.name+'/'+"cor2_b"+"/"+filename+".png", format='png')
                                final_img.writeto('/gehme/projects/2020_gcs_with_ml/data/corona_back_database/'+df.name+'/'+"cor2_b"+"/"+filename+".fits",overwrite=True)                  
                                im1.close()
                                im2.close()
                            else:
                                im1.close()
                                im2.close()
                    else:
                        print("path not found")
                except :
                    print("error on "+path_1h+"  or  "+path_2h)
    return cor2_a,cor2_b                    

#function to create corona background
#pathlist(cor2,cor2_downloads,do_write=do_write)




data=pathlist(cor2,cor2_downloads,do_write=do_write)
cor2_a=data[0]
cor2_b=data[1]
cor2_a.to_csv("/gehme/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn_training/corona_background/catalogues/analysis/cor2_a.csv")
cor2_b.to_csv("/gehme/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn_training/corona_background/catalogues/analysis/cor2_b.csv")

#cor2_a = pd.read_csv("/gehme/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn_training/corona_background/catalogues/analysis/cor2_a.csv")
#cor2_b=pd.read_csv("/gehme/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn_training/corona_background/catalogues/analysis/cor2_b.csv")

fig1, ax1 = plt.subplots()
ax1.plot(np.abs(cor2_a["header_contrast"]), '.k')
ax1.set_title('Header Contrast')
ax1.set_yscale('log')
fig1.savefig("/gehme/projects/2020_gcs_with_ml/data/plots/"+"cor2_a_contrast_plot_analyzed.png")

fig2, ax2 = plt.subplots()
ax2.hist(np.abs(cor2_a["header_contrast"]), bins=50, color='k', alpha=0.7)
ax2.set_title('Header Contrast')
ax2.set_yscale('log')
fig2.savefig("/gehme/projects/2020_gcs_with_ml/data/plots/"+'cor2_a_contrast_hist_analyzed.png')


fig3, ax3 = plt.subplots()
ax3.plot(np.abs(cor2_b["header_contrast"]), '.k')
ax3.set_title('Header Contrast')
ax3.set_yscale('log')
fig3.savefig("/gehme/projects/2020_gcs_with_ml/data/plots/"+"cor2_b_contrast_plot_analyzed.png")

fig4, ax4 = plt.subplots()
ax4.hist(np.abs(cor2_b["header_contrast"]), bins=50, color='k', alpha=0.7)
ax4.set_title('Header Contrast')
ax4.set_yscale('log')
fig4.savefig("/gehme/projects/2020_gcs_with_ml/data/plots/"+'cor2_b_contrast_hist_analyzed.png')

