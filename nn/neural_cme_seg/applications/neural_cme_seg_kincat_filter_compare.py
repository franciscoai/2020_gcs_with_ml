def get_kincat(odir,sat):
    df_list=[]
    odir=odir+sat+"/"
    ext_folders = os.listdir(odir)
    for ext_folder in ext_folders:
        odir_filter=odir+ext_folder+"/filtered"
        if os.path.exists(odir_filter):
            csv_path=odir_filter+"/"+ext_folder+".csv"
            df=pd.read_csv(csv_path)
            df_list.append(df)
    df_full = pd.concat(df_list, ignore_index=True)

    return df_full

def get_seeds(folder,sat):
    '''
    This function downloads the seeds catalogue and saves it in the folder specified
    '''
    url= 'http://spaceweather.gmu.edu/seeds/secchi/detection_cor2/monthly/'
    col_names=["DATE","TIME","EVENT_MOVIE","CPA_ANG","WIDE_ANG","LINEAR_FIT","2_LINEAR_FIT"]  

    if os.path.exists(folder) and os.path.isdir(folder):
        files=os.listdir(folder)
        df_list=[]
        for i in files:    
            if sat=="cor2_a":
                if i.endswith("A_monthly.txt"):
                    path=folder+"/"+i
                    df=pd.read_csv(path, header=None, delim_whitespace=True)
                    df.columns=col_names
                    df_list.append(df)
            elif sat=="cor2_b":
                 if i.endswith("B_monthly.txt"):
                    df=pd.read_csv(path, header=None, names=col_names, delimiter="/t")
                    df_list.append(df)
        df_full = pd.concat(df_list, ignore_index=True)
        df_full['DATE_TIME'] = pd.to_datetime(df_full['DATE'] + ' ' + df_full['TIME'])
             
    else:
        os.makedirs(folder)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('monthly.txt'):
                file_url = f"{url}/{href}"
                file_name = href.split('/')[-1]
                file_response = requests.get(file_url)
                if file_response.status_code == 200:
                    with open(folder+'/'+file_name, 'wb') as archivo:
                        archivo.write(file_response.content)

    return df_full

def comparator(kincat,seeds):
    seeds['DATE_TIME'] = pd.to_datetime(seeds['DATE_TIME'], format="YYYY/MM/DD HH:MM:SS")
    kincat['DATE_TIME'] = pd.to_datetime(kincat['DATE_TIME'])
    kincat.sort_values(by='DATE_TIME', inplace=True)
    seeds.sort_values(by='DATE_TIME', inplace=True)
    kincat['DATE'] = kincat['DATE_TIME'].dt.date
    kincat['TIME'] = pd.to_datetime(kincat['DATE_TIME']).dt.time

    df = kincat.groupby('DATE').agg({'WIDE_ANG': ['mean', 'std'],'CPA_ANG': ['median', 'std'],})
    df = df.reset_index()
    unique_dates_kincat = kincat['DATE_TIME'].dt.date.unique()
    seeds = seeds[seeds['DATE_TIME'].dt.date.isin(unique_dates_kincat)].reset_index()
    
    kincat_data= kincat.groupby('DATE')['TIME'].min().reset_index()
    for i in range(len(kincat_data)):
        
        
        if kincat_data['DATE'][i] == (pd.to_datetime(seeds['DATE'][i])).date():





            breakpoint()
            return idx

################################################################################### MAIN ######################################################################################

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import timedelta, datetime

odir="/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat_L1/"
folder="/gehme-gpu/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn/neural_cme_seg/applications/seeds_catalogue"
sat="cor2_a"#cor2_b

kincat=get_kincat(odir,sat)
seeds=get_seeds(folder,sat)
final_df=comparator(kincat,seeds)
