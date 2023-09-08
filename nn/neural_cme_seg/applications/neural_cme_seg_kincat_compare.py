def get_NN(odir,sat):
    df_list=[]
    odir=odir+sat+"/"
    ext_folders = os.listdir(odir)
    for ext_folder in ext_folders:
        odir_filter=odir+ext_folder+"/filtered"
        if os.path.exists(odir_filter):
            csv_path=odir_filter+"/"+ext_folder+"_filtered_stats"
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
    folder=folder+"/seeds_catalogue"
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

def get_vourlidas(folder,sat):
    '''
    This function downloads the Vourlidas catalogue and saves it in the folder specified
    '''
    url= "http://sd-www.jhuapl.edu/COR2CMECatalog/"
    cols=['CPA', 'Width', 'Vrad','Radial Accel.', 'Vexp', 'Expansion Accel.','Sep. Angle']

    folder=folder+"/Vourlidas_catalogue"
    if os.path.exists(folder) and os.path.isdir(folder):
        files=os.listdir(folder)
        df_list=[]
        for i in files:
    
            path=folder+"/"+i
            df=pd.read_csv(path, header=None)

            new_header = df.iloc[0]  
            df = df[1:]  
            df.columns = new_header
            df = df[df["CPA"] != '    -999']
            if sat=="cor2_a":
                df = df[df["S/C"] != "B"]
            else:
                df = df[df["S/C"] != "A"]
            df_list.append(df)
        
        df_full = pd.concat(df_list, ignore_index=True)
        for i in cols:
            df_full[i] = df_full[i].astype(float)


    else:
        os.makedirs(folder)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('csv'):
                file_url = f"{url}/{href}"
                file_name = href.split('/')[-1]
                file_response = requests.get(file_url)
                if file_response.status_code == 200:
                    with open(folder+'/'+file_name, 'wb') as archivo:
                        archivo.write(file_response.content)

    return df_full


def comparator(NN,seeds,vourlidas):
    columns=["NN_DATE_TIME","SEEDS_DATE_TIME","VOURLIDAS_DATE_TIME","NN_CPA_ANG_MEDIAN","NN_CPA_ANG_STD","SEEDS_CPA_ANG","VOURLIDAS_CPA_ANG","NN_WIDE_ANG_MEDIAN","NN_WIDE_ANG_STD","SEEDS_WIDE_ANG","VOURLIDAS_WIDE_ANG"]
    NN_ang_col=['CPA_ANG', 'WIDE_ANG']
       
    compare=[]
    #converts to datetime objs and sort the df
    seeds['DATE_TIME'] = pd.to_datetime(seeds['DATE_TIME'], format="YYYY/MM/DD HH:MM:SS")
    NN['DATE_TIME'] = pd.to_datetime(NN['DATE_TIME'])
    vourlidas['Date_Time'] =vourlidas['Date']+" "+vourlidas['Time']
    vourlidas['Date_Time'] = pd.to_datetime(vourlidas['Date_Time'])
    vourlidas = vourlidas.sort_values(by='Date_Time')
    NN.sort_values(by='DATE_TIME', inplace=True)
    seeds.sort_values(by='DATE_TIME', inplace=True)
    NN['DATE'] = NN['DATE_TIME'].dt.date
    NN['TIME'] = pd.to_datetime(NN['DATE_TIME']).dt.time
    
    #adjust the 0 degree to the nort
    for i in NN_ang_col:
        
        if i=="WIDE_ANG":
            NN[i]=np.degrees(NN[i])
        
        else:
            NN[i]= np.degrees(NN[i])-90#-np.degrees(NN[i])+270
        #NN.loc[NN[i] < 0, i] += 360 
    
    

    #goups all the hours in each day and calculates medain a std of cpa_ang and wide_ang
    df = NN.groupby('DATE').agg({'WIDE_ANG': ['median', 'std'],'CPA_ANG': ['median', 'std'],})
    df = df.reset_index()
    #calculate the unique dates in NN and conservs only those ones in seeds
    unique_dates_NN = NN['DATE_TIME'].dt.date.unique()
    seeds = seeds[seeds['DATE_TIME'].dt.date.isin(unique_dates_NN)].reset_index()
    vourlidas = vourlidas[vourlidas['Date_Time'].dt.date.isin(unique_dates_NN)].reset_index()
    
    #keeps the first hour of everey day
    NN_min= NN.groupby('DATE')['TIME'].min().reset_index()
    NN_max= NN.groupby('DATE')['TIME'].max().reset_index()
    #creates the list to compare the events in NN and seeds
    for i in range(len(NN_min)):
        NN_min['DATE_TIME'] = NN_min.apply(lambda row: datetime.combine(row['DATE'], row['TIME']), axis=1)
        NN_max['DATE_TIME'] = NN_max.apply(lambda row: datetime.combine(row['DATE'], row['TIME']), axis=1)
        NN_date=NN_min["DATE_TIME"][i]
        #path=NN.loc[NN["DATE_TIME"]==NN_date, "PATH"].values[0]
        seeds['TIME_DIFF'] = seeds['DATE_TIME'] - pd.to_datetime(NN_date)
        time_diff_seeds = seeds[seeds['TIME_DIFF'] >= pd.Timedelta(0)]
        idx_seeds = time_diff_seeds['TIME_DIFF'].idxmin()
        seeds_data = seeds.loc[idx_seeds]
        cpa_ang_seeds=seeds_data["CPA_ANG"]
        wide_ang_seeds=seeds_data["WIDE_ANG"]

        vourlidas['Time_Diff'] = vourlidas['Date_Time'] - pd.to_datetime(NN_date)
        time_diff_vourlidas = vourlidas[vourlidas['Time_Diff'] >= pd.Timedelta(0)]
        idx_vourlidas = time_diff_vourlidas['Time_Diff'].idxmin()
        vourlidas_data = vourlidas.loc[idx_vourlidas]
        cpa_ang_vourlidas=vourlidas_data["CPA"]
        wide_ang_vourlidas=vourlidas_data["Width"]
        if (NN_min["DATE_TIME"][i]<=vourlidas_data["Date_Time"]<=NN_max["DATE_TIME"][i])and(NN_min["DATE_TIME"][i]<=seeds_data["DATE_TIME"]<=NN_max["DATE_TIME"][i]):
            compare.append([NN_date,seeds_data["DATE_TIME"],vourlidas_data["Date_Time"],df["CPA_ANG"]["median"][i],df["CPA_ANG"]["std"][i],cpa_ang_seeds,cpa_ang_vourlidas,df["WIDE_ANG"]["median"][i],df["WIDE_ANG"]["std"][i],wide_ang_seeds,wide_ang_vourlidas])
        elif (NN_min["DATE_TIME"][i]<=vourlidas_data["Date_Time"]<=NN_max["DATE_TIME"][i]) and not(NN_min["DATE_TIME"][i]<=seeds_data["DATE_TIME"]<=NN_max["DATE_TIME"][i]):
            compare.append([NN_date,np.nan,vourlidas_data["Date_Time"],df["CPA_ANG"]["median"][i],df["CPA_ANG"]["std"][i],np.nan,cpa_ang_vourlidas,df["WIDE_ANG"]["median"][i],df["WIDE_ANG"]["std"][i],np.nan,wide_ang_vourlidas])
        elif not(NN_min["DATE_TIME"][i]<=vourlidas_data["Date_Time"]<=NN_max["DATE_TIME"][i])and(NN_min["DATE_TIME"][i]<=seeds_data["DATE_TIME"]<=NN_max["DATE_TIME"][i]):
            compare.append([NN_date,seeds_data["DATE_TIME"],np.nan,df["CPA_ANG"]["median"][i],df["CPA_ANG"]["std"][i],cpa_ang_seeds,np.nan,df["WIDE_ANG"]["median"][i],df["WIDE_ANG"]["std"][i],wide_ang_seeds,np.nan])
    compare = pd.DataFrame(compare, columns=columns)
    return compare

################################################################################### MAIN ######################################################################################

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import timedelta, datetime, time

odir="/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat_L1/"
folder="/gehme-gpu/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn/neural_cme_seg/applications"
sat="cor2_a"#cor2_b


NN=get_NN(odir,sat)
seeds=get_seeds(folder,sat)
vourlidas= get_vourlidas(folder,sat)
df=comparator(NN,seeds,vourlidas)

#breakpoint()
fig, ax = plt.subplots(figsize=(6, 6))
ax.errorbar(df["NN_CPA_ANG_MEDIAN"], df["VOURLIDAS_CPA_ANG"], xerr=df["NN_CPA_ANG_STD"], fmt='o', color='green', ecolor='gray', capsize=5, label='VOURLIDAS')
ax.errorbar(df["NN_CPA_ANG_MEDIAN"], df["SEEDS_CPA_ANG"], xerr=df["NN_CPA_ANG_STD"], fmt='o', color='blue', ecolor='gray', capsize=5, label='SEEDS')
ax.plot([0, 450], [0, 450], color='red', linestyle='--')
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_xlabel('Neural Segmentation')
ax.set_title('CPA ANG')
ax.legend()
ax.grid(True)
fig.savefig(folder+"/plots/"'CPA_ANG_all.png', dpi=300, bbox_inches='tight')


fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.errorbar(df["NN_WIDE_ANG_MEDIAN"], df["VOURLIDAS_WIDE_ANG"], xerr=df["NN_WIDE_ANG_STD"], fmt='o', color='green', ecolor='gray', capsize=5, label='VOURLIDAS')
ax2.errorbar(df["NN_WIDE_ANG_MEDIAN"], df["SEEDS_WIDE_ANG"], xerr=df["NN_WIDE_ANG_STD"], fmt='o', color='blue', ecolor='gray', capsize=5, label='SEEDS')
ax2.plot([0, 350], [0, 350], color='red', linestyle='--')
ax2.set_xlim(0, ax2.get_xlim()[1])
ax2.set_xlabel('Neural Segmentation')
ax2.set_title('WIDE ANG')
ax2.legend()
ax2.grid(True)
fig2.savefig(folder+"/plots/"'WIDE_ANG_all.png', dpi=300, bbox_inches='tight')





fig3, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(18, 6))

ax3.errorbar(df["SEEDS_WIDE_ANG"], df["VOURLIDAS_WIDE_ANG"], fmt='o', color='purple', ecolor='gray', capsize=5, label='VOURLIDAS vs SEEDS')
ax3.plot([0, 350], [0, 350], color='red', linestyle='--')
ax3.set_xlim(0, ax3.get_xlim()[1])
ax3.set_xlabel('Neural Segmentation')
ax3.set_title('WIDE ANG')
ax3.legend()
ax3.grid(True)
ax4.errorbar(df["SEEDS_WIDE_ANG"], df["NN_WIDE_ANG_MEDIAN"], fmt='o', color='purple', ecolor='gray', capsize=5, label='NN vs SEEDS')
ax4.plot([0, 450], [0, 450], color='red', linestyle='--')
ax4.set_xlim(0, ax4.get_xlim()[1])
ax4.set_xlabel('Neural Segmentation')
ax4.set_title('WIDE ANG')
ax4.legend()
ax4.grid(True)
ax5.errorbar(df["VOURLIDAS_WIDE_ANG"], df["NN_CPA_WIDE_MEDIAN"], fmt='o', color='purple', ecolor='gray', capsize=5, label='NN vs VOURLIDAS')
ax5.plot([0, 550], [0, 550], color='red', linestyle='--')
ax5.set_xlim(0, ax5.get_xlim()[1])
ax5.set_xlabel('Neural Segmentation')
ax5.set_title('WIDE ANG')
ax5.legend()
ax5.grid(True)
fig3.savefig(folder+"/plots/"+'WIDE_ANG.png', dpi=300, bbox_inches='tight')
             
fig4, (ax6, ax7, ax8) = plt.subplots(1, 3, figsize=(18, 6))
ax6.errorbar(df["SEEDS_CPA_ANG"], df["VOURLIDAS_CPA_ANG"], fmt='o', color='purple', ecolor='gray', capsize=5, label='VOURLIDAS vs SEEDS')
ax6.plot([0, 650], [0, 650], color='red', linestyle='--')
ax6.set_xlim(0, ax6.get_xlim()[1])
ax6.set_xlabel('Neural Segmentation')
ax6.set_title('CPA ANG')
ax6.legend()
ax6.grid(True)
ax7.errorbar(df["SEEDS_CPA_ANG"], df["NN_CPA_ANG_MEDIAN"], fmt='o', color='purple', ecolor='gray', capsize=5, label='NN vs SEEDS')
ax7.plot([0, 750], [0, 750], color='red', linestyle='--')
ax7.set_xlim(0, ax7.get_xlim()[1])
ax7.set_xlabel('Neural Segmentation')
ax7.set_title('CPA ANG')
ax7.legend()
ax7.grid(True)
ax8.errorbar(df["VOURLIDAS_CPA_ANG"], df["NN_CPA_ANG_MEDIAN"], fmt='o', color='purple', ecolor='gray', capsize=8, label='NN vs VOURLIDAS')
ax8.plot([0, 880], [0, 880], color='red', linestyle='--')
ax8.set_xlim(0, ax8.get_xlim()[1])
ax8.set_xlabel('Neural Segmentation')
ax8.set_title('CPA ANG')
ax8.legend()
ax8.grid(True)
fig3.savefig(folder+"/plots/"'CPA_ANG.png', dpi=300, bbox_inches='tight')
             

plt.show()
plt.close()



#df.loc[df["SEEDS_WIDE_ANG"]>200]