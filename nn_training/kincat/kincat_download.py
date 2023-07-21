
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from data_download.descargar_imagenes_clases import cor2_downloader
import pandas as pd
from datetime import datetime, timedelta

#current dir
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
units= ["-","-","yyyymmdd","hhmm","yyyymmdd","hhmm","deg","lon","old","deg","deg","deg", "yyyymmdd","hhmm","km/s","g","km/s","LON","LAT","km/s","LON","LAT","km/s","LON","LAT"]
col_names=["HEL","CME","PRE_DATE","PRE_TIME","LAST_DATE","LAST_TIME","CARLON","STONEY","LAT","TILT", "ASP_RATIO","H_ANGLE","DATE","TIME","APEX_SPEED","CME_MASS","SPEED_FPF","FPF_LON","FPF_LAT","SPEED_SSEF","SSEF_LON","SSEF_LAT","SPEED_HMF","HMF_LON","HMF_LAT"]
suffix = "/gehme/data/stereo/"
helcat_db =  repo_dir + "/nn_training/kincat/helcatslist_20160601.txt" # kincat database
downloaded_files_list = repo_dir + '/nn_training/kincat/helcatslist_20160601_downloaded.csv' # list of downloaded files

# read helcats database
df = pd.read_csv(helcat_db, sep = "\t")
df=df.drop([0,1])
df.columns=col_names
df = df.reset_index(drop=True)

data=[]
for i in range(22,len(df.index)):
#for i in range(11,12):
    print("Downloading row number  {}".format(i))
    pre_date = df["PRE_DATE"][i]
    pre_time= df["PRE_TIME"][i]
    last_date = df["LAST_DATE"][i]
    last_time= df["LAST_TIME"][i]
    if (last_date+" "+last_time)>(pre_date+" "+pre_time):
        pre_datetime=datetime.strptime((pre_date+" "+pre_time),'%Y-%m-%d %H:%M')
        pre_diff = pre_datetime - timedelta(hours=1)
        start=pre_diff.strftime('%Y/%m/%d %H:%M:%S')
        last_datetime=datetime.strptime((last_date+" "+last_time),'%Y-%m-%d %H:%M')
        last_diff = last_datetime - timedelta(hours=1)
        end=last_diff.strftime('%Y/%m/%d %H:%M:%S')
        # download COR2 images
        asd = cor2_downloader(start_time=pre_diff,end_time=last_diff,nave='STEREO_A',nivel='double',image_type='img',size=2)
        asd.search()
        #breakpoint()
        if len(asd.search_cor2) >= 1:   
            asd.download()
            for p in range(len(asd.search_cor2)):
                path=asd.search_cor2[p][9]
                date_time = datetime.strptime(path[30:-10], '%Y%m%d_%H%M%S')
                data.append([suffix+path,date_time])
    #breakpoint()
data = pd.DataFrame(data, columns=["PATH","DATE_TIME"])    
data.to_csv(downloaded_files_list,  index=False)
