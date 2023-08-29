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
    columns=["PATH","KINCAT_DATE_TIME","SEEDS_DATE_TIME","KINCAT_CPA_ANG_MEDIAN","KINCAT_CPA_ANG_STD","SEEDS_CPA_ANG","KINCAT_WIDE_ANG_MEDIAN","KINCAT_WIDE_ANG_STD","SEEDS_WIDE_ANG"]
    kincat_ang_col=['MIN_ANG', 'MAX_ANG','CPA_ANG', 'WIDE_ANG', 'MASS_CENTER_ANG', 'APEX_ANG']
       
    compare=[]
    #converts to datetime objs and sort the df
    seeds['DATE_TIME'] = pd.to_datetime(seeds['DATE_TIME'], format="YYYY/MM/DD HH:MM:SS")
    kincat['DATE_TIME'] = pd.to_datetime(kincat['DATE_TIME'])
    kincat.sort_values(by='DATE_TIME', inplace=True)
    seeds.sort_values(by='DATE_TIME', inplace=True)
    kincat['DATE'] = kincat['DATE_TIME'].dt.date
    kincat['TIME'] = pd.to_datetime(kincat['DATE_TIME']).dt.time
    #adjust the 0 degree to the nort
    for i in kincat_ang_col:
        kincat[i]=kincat[i]-np.radians(90)
  

    #goups all the hours in each day and calculates medain a std of cpa_ang and wide_ang
    df = kincat.groupby('DATE').agg({'WIDE_ANG': ['median', 'std'],'CPA_ANG': ['median', 'std'],})
    df = df.reset_index()
    #calculate the unique dates in kincat and conservs only those ones in seeds
    unique_dates_kincat = kincat['DATE_TIME'].dt.date.unique()
    seeds = seeds[seeds['DATE_TIME'].dt.date.isin(unique_dates_kincat)].reset_index()
    #keeps the first hour of everey day
    kincat_min= kincat.groupby('DATE')['TIME'].min().reset_index()
    
    #creates the list to compare the events in kincat and seeds
    for i in range(len(kincat_min)):
        kincat_min['DATE_TIME'] = kincat_min.apply(lambda row: datetime.combine(row['DATE'], row['TIME']), axis=1)
        kincat_date=kincat_min["DATE_TIME"][i]
        path=kincat.loc[kincat["DATE_TIME"]==kincat_date, "PATH"].values[0]
        seeds['TIME_DIFF'] = seeds['DATE_TIME'] - pd.to_datetime(kincat_date)
        positive_time_diff = seeds[seeds['TIME_DIFF'] >= pd.Timedelta(0)]
        min_positive_idx = positive_time_diff['TIME_DIFF'].idxmin()
        seeds_data = seeds.loc[min_positive_idx]
        cpa_ang_seeds=np.radians(seeds_data["CPA_ANG"])
        wide_ang_seeds=np.radians(seeds_data["WIDE_ANG"])
        compare.append([path,kincat_date,seeds_data["DATE_TIME"],df["CPA_ANG"]["median"][i],df["CPA_ANG"]["std"][i],cpa_ang_seeds,df["WIDE_ANG"]["median"][i],df["WIDE_ANG"]["std"][i],wide_ang_seeds])
    
    compare = pd.DataFrame(compare, columns=columns)
    return compare

################################################################################### MAIN ######################################################################################

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import timedelta, datetime

odir="/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat_L1/"
folder="/gehme-gpu/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn/neural_cme_seg/applications/seeds_catalogue"
sat="cor2_a"#cor2_b

kincat=get_kincat(odir,sat)
seeds=get_seeds(folder,sat)
df=comparator(kincat,seeds)


fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(df["KINCAT_CPA_ANG_MEDIAN"], df["SEEDS_CPA_ANG"], color='blue', label='CPA')
ax.errorbar(df["KINCAT_CPA_ANG_MEDIAN"], df["SEEDS_CPA_ANG"], yerr=df["KINCAT_CPA_ANG_STD"], fmt='o', color='blue', ecolor='gray', capsize=5, label='STD')
ax.plot([0, 7.5], [0, 7.5], color='red', linestyle='--')
ax.set_xlabel('Kincat')
ax.set_ylabel('Seeds')
ax.set_title('CPA ANG')
ax.legend()
ax.grid(True)

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.scatter(df["KINCAT_WIDE_ANG_MEDIAN"], df["SEEDS_WIDE_ANG"], color='blue', label='CPA')
ax2.errorbar(df["KINCAT_WIDE_ANG_MEDIAN"], df["SEEDS_WIDE_ANG"], yerr=df["KINCAT_WIDE_ANG_STD"], fmt='o', color='blue', ecolor='gray', capsize=5, label='STD')
ax2.plot([0, 7.5], [0, 7.5], color='red', linestyle='--')
ax2.set_xlabel('Kincat')
ax2.set_ylabel('Seeds')
ax2.set_title('WIDE ANG')
ax2.legend()
ax2.grid(True)
plt.show()
plt.close()
