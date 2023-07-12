
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from data_download.descargar_imagenes_clases import cor2_downloader
import pandas as pd
from datetime import datetime, timedelta

units= ["-","-","yyyymmdd","hhmm","yyyymmdd","hhmm","deg","lon","old","deg","deg","deg", "yyyymmdd","hhmm","km/s","g","km/s","LON","LAT","km/s","LON","LAT","km/s","LON","LAT"]
col_names=["HEL","CME","PRE_DATE","PRE_TIME","LAST_DATE","LAST_TIME","CARLON","STONEY","LAT","TILT", "ASP_RATIO","H_ANGLE","DATE","TIME","APEX_SPEED","CME_MASS","SPEED_FPF","FPF_LON","FPF_LAT","SPEED_SSEF","SSEF_LON","SSEF_LAT","SPEED_HMF","HMF_LON","HMF_LAT"]
suffix = "/gehme/data/stereo/secchi/"


df = pd.read_csv("/gehme/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn_training/kincat/helcatslist_20160601.txt", sep = "\t")
df=df.drop([0,1])
df.columns=["HEL","CME","PRE_DATE","PRE_TIME","LAST_DATE","LAST_TIME","CARLON","STONEY","LAT","TILT", "ASP_RATIO","H_ANGLE","DATE","TIME","APEX_SPEED","CME_MASS","SPEED_FPF","FPF_LON","FPF_LAT","SPEED_SSEF","SSEF_LON","SSEF_LAT","SPEED_HMF","HMF_LON","HMF_LAT"]
df = df.reset_index(drop=True)

data=pd.DataFrame()
for i in range(len(df.index)):
    print("chequeando elemento Numero {}".format(i))
    pre_date = df["PRE_DATE"][i]
    pre_time= df["PRE_TIME"][i]
    last_date = df["LAST_DATE"][i]
    last_time= df["LAST_TIME"][i]
    pre_datetime=datetime.strptime((pre_date+" "+pre_time),'%Y-%m-%d %H:%M')
    pre_diff = pre_datetime - timedelta(hours=1)
    start=pre_diff.strftime('%Y/%m/%d %H:%M:%S')
    last_datetime=datetime.strptime((last_date+" "+last_time),'%Y-%m-%d %H:%M')
    last_diff = last_datetime - timedelta(hours=1)
    end=last_diff.strftime('%Y/%m/%d %H:%M:%S')

    asd = cor2_downloader(start_time=pre_diff,end_time=last_diff,nave='STEREO_A',nivel='double',image_type='img',size=2)
    asd.search()

    if len(asd.search_cor2) >= 1:   
        asd.download()
        breakpoint()
        data.append(suffix+"/".join(str(asd.search_cor2[asd.indices_descarga]['fileid']).split('/')[1:]))
        
    
data.to_csv('/gehme/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn_training/kincat/helcatslist_20160601_downloaded.csv', sep='\t', index=False)
