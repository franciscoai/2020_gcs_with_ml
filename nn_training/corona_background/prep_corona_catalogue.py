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
from nn.neural_cme_seg.neural_cme_seg import neural_cme_segmentation
import torch
"""
Reads pairs of LVL1 coronograph images from various instruments and saves a differential corona for each pair.
Images are resized to 512,512 pixels.
Only images with low contrast and no CMEs detected by our NN are saved.
"""
exec_path = os.getcwd()+"/catalogues"
lasco_path = exec_path+'/Lista_Final_CMEs_downloads_lascoc2.csv' #'/catalogues/Lista_Final_CMEs.csv' # file with the list of cor files
cor2_path = exec_path+'/Lista_Final_CMEs_downloads_cor2.csv'
lasco_downloads=["pre_a_1h_download_c2","pre_b_1h_download_c2","pre_a_2h_download_c2","pre_b_2h_download_c2"]
cor2_downloads=["pre_a_1h_download","pre_b_1h_download","pre_a_2h_download","pre_b_2h_download"]
opath = '/gehme/projects/2020_gcs_with_ml/data/corona_back_database/'
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

def prep_catalogue(df,column_list, do_write=True, model_param=None, device=None):
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
        #creates odir
        odir = opath+df.name+'/'+"cor2_a"
        odirb= opath+df.name+'/'+"cor2_b"
        os.makedirs(odir, exist_ok=True)
        os.makedirs(odirb, exist_ok=True)

        for i in paths["paths"]:
            i=i.replace("L0","L1")
            if i.endswith("a.fts") or i.endswith("A.fts"):
                basename=os.path.basename(i)
                date = datetime.strptime(basename[0:-10], formato)
                cor2_a=pd.concat([cor2_a,pd.DataFrame({"paths":[i],"date":[date]})], ignore_index=True)
                #cor2_a=cor2_a.append({"paths":i,"date":date}, ignore_index=True)
               
            elif i.endswith("b.fts") or i.endswith("B.fts"):
                basename=os.path.basename(i)
                date = datetime.strptime(basename[0:-10], formato)
                cor2_b=pd.concat([cor2_b,pd.DataFrame({"paths":[i],"date":[date]})], ignore_index=True)
                #cor2_b=cor2_b.append({"paths":i,"date":date}, ignore_index=True)
               
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
                        filename=filename[0:-4]
                        fig0, ax0 = plt.subplots()
                        mean= np.mean(im)
                        std = np.std(im)
                        vmin=mean-1*std
                        vmax=mean+1*std
                        imagen = ax0.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
                        if do_write==True:
                            imgs, masks, scrs, labels, boxes  = neural_cme_segmentation(model_param, im, device)
                            if not np.max(scrs)>SCR_THRESHOLD:  # header_contrast<6.2 and 
                                print("saving image "+str(i)+" from cor2_a")
                                plt.savefig(odir+"/"+filename+".png", format='png')
                                final_img.writeto(odir+"/"+filename+".fits",overwrite=True)
                                im1.close()
                                im2.close()
                            else:
                                print("image "+str(i)+" from cor2_a has a CME")
                                im1.close()
                                im2.close()
                    else:
                        print("files not found")
                            
                except:
                    im1.close()
                    im2.close()                    
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
                        filename=filename[0:-4]
                        fig0, ax0 = plt.subplots()
                        mean= np.mean(im)
                        std = np.std(im)
                        vmin=mean-1*std
                        vmax=mean+1*std
                        imagen = ax0.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
                        if do_write==True:    
                            imgs, masks, scrs, labels, boxes  = neural_cme_segmentation(model_param, im, device)
                            if not np.max(scrs)>SCR_THRESHOLD: # header_contrast<4.7 and
                                print("saving image "+str(i)+" from cor2_b")
                                plt.savefig(odirb+"/"+filename+".png", format='png')
                                final_img.writeto(odirb+"/"+filename+".fits",overwrite=True)                  
                                im1.close()
                                im2.close()
                            else:
                                print("image "+str(i)+" from cor2_a has a CME")
                                im1.close()
                                im2.close()
                    else:
                        print("files not found")
                except :
                    im1.close()
                    im2.close()
                    print("error on "+path_1h+"  or  "+path_2h)
    return cor2_a,cor2_b                    

#### main
#nn inference
model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v3"
trained_model = '3999.torch'
SCR_THRESHOLD=0.3 # only save images with score below this threshold (i.e., No CME is present)
gpu=0 # GPU to use
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unless its not available
print(f'Using device:  {device}')
#model_param = torch.load(model_path + "/"+ trained_model)
model_param=torch.load(model_path + "/" + trained_model, map_location=device)

#tasks
#cor2 a and b
data=prep_catalogue(cor2,cor2_downloads,do_write=do_write,model_param=model_param, device=device) # get paths of ok files
# lasco
#data=prep_catalogue(lasco,lasco_downloads,do_write=do_write,model_param=model_param, device=device) # get paths of ok files ??

# saves data to csv and plots
# cor2_a=data[0]
# cor2_b=data[1]
# cor2_a.to_csv("/gehme/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn_training/corona_background/catalogues/analysis/cor2_a.csv")
# cor2_b.to_csv("/gehme/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn_training/corona_background/catalogues/analysis/cor2_b.csv")

# # plots contrast stats
# fig1, ax1 = plt.subplots()
# ax1.plot(np.abs(cor2_a["header_contrast"]), '.k')
# ax1.set_title('Header Contrast')
# ax1.set_yscale('log')
# fig1.savefig("/gehme/projects/2020_gcs_with_ml/data/plots/"+"cor2_a_contrast_plot_analyzed.png")

# fig2, ax2 = plt.subplots()
# ax2.hist(np.abs(cor2_a["header_contrast"]), bins=50, color='k', alpha=0.7)
# ax2.set_title('Header Contrast')
# ax2.set_yscale('log')
# fig2.savefig("/gehme/projects/2020_gcs_with_ml/data/plots/"+'cor2_a_contrast_hist_analyzed.png')

# fig3, ax3 = plt.subplots()
# ax3.plot(np.abs(cor2_b["header_contrast"]), '.k')
# ax3.set_title('Header Contrast')
# ax3.set_yscale('log')
# fig3.savefig("/gehme/projects/2020_gcs_with_ml/data/plots/"+"cor2_b_contrast_plot_analyzed.png")

# fig4, ax4 = plt.subplots()
# ax4.hist(np.abs(cor2_b["header_contrast"]), bins=50, color='k', alpha=0.7)
# ax4.set_title('Header Contrast')
# ax4.set_yscale('log')
# fig4.savefig("/gehme/projects/2020_gcs_with_ml/data/plots/"+'cor2_b_contrast_hist_analyzed.png')

