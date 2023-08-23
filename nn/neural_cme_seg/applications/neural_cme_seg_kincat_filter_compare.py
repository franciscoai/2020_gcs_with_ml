def get_kincat(odir):
    df_list=[]
    ext_folders = os.listdir(odir)
    for ext_folder in ext_folders:
        odir_filter=odir+ext_folder+"/filtered"
        if os.path.exists(odir_filter):
            csv_path=odir_filter+"/"+ext_folder+".csv"
            df=pd.read_csv(csv_path)
            columns=df.columns
            df_list.append(df)
    df_full = pd.concat(df_list, ignore_index=True)

    return df_full

def get_seeds(folder):
    '''
    This function downloads the seeds catalogue and saves it in the folder specified
    '''
    url= 'http://spaceweather.gmu.edu/seeds/secchi/detection_cor2/monthly/'  

    if os.path.exists(folder) and os.path.isdir(folder):
        files=os.listdir(folder)
        
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

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

odir="/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat_L1/cor2_a/"
folder="/gehme-gpu/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn/neural_cme_seg/applications/seeds_catalogue"
kincat=get_kincat(odir)
sedds=get_seeds(folder)
print(kincat)