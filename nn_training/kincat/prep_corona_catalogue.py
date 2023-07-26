import pandas as pd
import os
from datetime import datetime, timedelta
from astropy.io import fits
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from ext_libs.rebin import rebin
import glob

units= ["-","-","yyyymmdd","hhmm","yyyymmdd","hhmm","deg","lon","old","deg","deg","deg", "yyyymmdd","hhmm","km/s","g","km/s","LON","LAT","km/s","LON","LAT","km/s","LON","LAT"]
col_names=["HEL","CME","PRE_DATE","PRE_TIME","LAST_DATE","LAST_TIME","CARLON","STONEY","LAT","TILT", "ASP_RATIO","H_ANGLE","DATE","TIME","APEX_SPEED","CME_MASS","SPEED_FPF","FPF_LON","FPF_LAT","SPEED_SSEF","SSEF_LON","SSEF_LAT","SPEED_HMF","HMF_LON","HMF_LAT"]
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
helcat_db=repo_dir + "/nn_training/kincat/helcatslist_20160601.txt" # kincat database
opath= "/gehme/projects/2020_gcs_with_ml/data/corona_background_kincat/cor2/cor2_b"
imsize=[512,512]

# read csv with paths and dates of the downloaded files
downloaded=pd.read_csv('/gehme/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn_training/kincat/helcatslist_20160601_stb_downloaded.csv')
downloaded['DATE_TIME'] = pd.to_datetime(downloaded['DATE_TIME'])
downloaded= downloaded.sort_values('DATE_TIME')
downloaded = downloaded.reset_index(drop=True)



# read helcat database and changes the column names
catalogue = pd.read_csv(helcat_db, sep = "\t")
catalogue=catalogue.drop([0,1])
catalogue.columns=col_names
catalogue = catalogue.reset_index(drop=True)

indice=0

for i in range(len(catalogue.index)):
    print("Reading path "+str(i))
    date_helcat = datetime.strptime((catalogue["PRE_DATE"][i]+" "+catalogue["PRE_TIME"][i]),'%Y-%m-%d %H:%M') #forms datetime object
    time_range= date_helcat-timedelta(hours=6)

    #calculates the closest dates to the original one
    for j in range(len(downloaded.index)):
        date=downloaded["DATE_TIME"][j]
        if date <= date_helcat:
                before_date = date
        elif date >= date_helcat:
                after_date = date
    if date_helcat-before_date>after_date-date_helcat:
          date_helcat=after_date
    else:
          date_helcat=before_date
    
    #if finds dates before date_helcat it takes the closest one and calculate the diff image between them
    dates=[]
    for j in range(len(downloaded.index)):
        path=downloaded.loc[j,"PATH"]
        #if os.path.exists(path):
        date=downloaded["DATE_TIME"][j]
        if date<date_helcat and date>time_range:
            dates.append(date)
        dates=sorted(dates)
        if len(dates)>0:
            pre_date= dates[0]
            idx1=downloaded.index[downloaded['DATE_TIME'] == date_helcat].tolist()
            idx2 = downloaded.index[downloaded['DATE_TIME'] == pre_date].tolist()
            #breakpoint()
            file1=glob.glob(downloaded.loc[idx1[0],"PATH"][0:-5]+"*")
            file2=glob.glob(downloaded.loc[idx2[0],"PATH"][0:-5]+"*")
            if len(file1) ==0 or len(file2)==0: 
                breakpoint() 
            if len(file1)!=0 or len(file2)!=0:
                print(indice)
                indice+=1
                # path1=file1[0]
                # path2=file2[0]
                # im1= fits.open(path1)
                # im2= fits.open(path2)
                # image1 = rebin(im1[0].data,imsize,operation='mean') 
                # image2 = rebin(im2[0].data,imsize,operation='mean') 
                # im=image1-image2
                # header= fits.getheader(path1)
                # filename = os.path.basename(path1)
                # header['NAXIS1'] = imsize[0]   
                # header['NAXIS2'] = imsize[1]
                # #breakpoint()
                # final_img = fits.PrimaryHDU(im, header=header[0:-3])
                # final_img.writeto(opath+"/"+filename,overwrite=True)
                # im1.close()
                # im2.close()
                

                        

                        
