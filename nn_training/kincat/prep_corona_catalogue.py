import pandas as pd

units= ["-","-","yyyymmdd","hhmm","yyyymmdd","hhmm","deg","lon","old","deg","deg","deg", "yyyymmdd","hhmm","km/s","g","km/s","LON","LAT","km/s","LON","LAT","km/s","LON","LAT"]
col_names=["HEL","CME","PRE_DATE","PRE_TIME","LAST_DATE","LAST_TIME","CARLON","STONEY","LAT","TILT", "ASP_RATIO","H_ANGLE","DATE","TIME","APEX_SPEED","CME_MASS","SPEED_FPF","FPF_LON","FPF_LAT","SPEED_SSEF","SSEF_LON","SSEF_LAT","SPEED_HMF","HMF_LON","HMF_LAT"]

catalogue = pd.read_csv("/gehme/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn_training/kincat/helcatslist_20160601.txt", sep = "\t")
downloaded=pd.read_csv('/gehme/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn_training/kincat/helcatslist_20160601_downloaded.csv')