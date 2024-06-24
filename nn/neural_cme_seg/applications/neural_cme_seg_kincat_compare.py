
import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import timedelta, datetime, time
from scipy.optimize import least_squares
from scipy.io import readsav

def get_NN(iodir,sat):
    column_names=["DATE_TIME","MASK","SCORE","CPA_ANG","WIDE_ANG","APEX",'LABEL', 'BOX', 'CME_ID', 'APEX_DIST', 'CPA_DIST', 'WA_DIST', 'ERROR']
    df_list=[]
    odir=iodir+'/'+sat+"/"
    ext_folders = os.listdir(odir)
    for ext_folder in ext_folders:
        odir_filter=odir+ext_folder+"/filtered"
        if os.path.exists(odir_filter):
            csv_path=odir_filter+"/"+ext_folder+"_filtered_stats"
            if  os.path.exists(csv_path):
                df=pd.read_csv(csv_path, names=column_names)
                if len(df["CME_ID"].unique())>1:
                    predominant_cme = df["CME_ID"].value_counts().idxmax()
                    event= df.loc[df["CME_ID"]==predominant_cme]
                    df_list.append(event)

                else:
                    df_list.append(df)
    df_full = pd.concat(df_list, ignore_index=True)
    return df_full

def get_GCS(iodir,sat):
    df_list=[]
    odir=iodir+'/'+sat+"/"
    ext_folders = os.listdir(odir)
    for ext_folder in ext_folders:
        odir_filter=odir+ext_folder+"/gcs_masks"
        if os.path.exists(odir_filter):
            csv_path=odir_filter+"/GCS_mask_stats"
            try:
                df=pd.read_csv(csv_path)
                df_list.append(df)
            except:
                continue
    df_full = pd.concat(df_list, ignore_index=True)
    return df_full


def get_seeds(folder,sat):
    '''
    This function downloads the seeds catalogue and saves it in the folder specified
    '''
    url= 'http://spaceweather.gmu.edu/seeds/secchi/detection_cor2/monthly/'
    col_names=["DATE","TIME","EVENT_MOVIE","CPA_ANG","WIDE_ANG","LINEAR_FIT","2_LINEAR_FIT"]  
    folder=folder+"/seeds_catalogue"
    dt_to_delete = [datetime(2010, 3, 30,3,24,0), datetime(2010, 3, 1,3,24,0),datetime(2010, 3, 1,5,24,0),datetime(2008, 6, 2,2,37,30)]
    if os.path.exists(folder) and os.path.isdir(folder):
        files=os.listdir(folder)
        df_list=[]
        for i in files:
            path=folder+"/"+i
               
            if sat=="cor2_a":
                if i.endswith("A_monthly.txt"):
                    
                    df=pd.read_csv(path, header=None, delim_whitespace=True)
                    df.columns=col_names
                    df_list.append(df)
            elif sat=="cor2_b":
                 if i.endswith("B_monthly.txt"):
                    df=pd.read_csv(path,header=None, delim_whitespace=True)
                    df.columns= col_names
                    df_list.append(df)
        df_full = pd.concat(df_list, ignore_index=True)
        df_full['DATE_TIME'] = pd.to_datetime(df_full['DATE'] + ' ' + df_full['TIME'])
        #delete some cases that are false or multiple cme detections
        df_full = df_full[~df_full['DATE_TIME'].isin(dt_to_delete)]
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

    # correct angles that are > 360
    df_full.loc[df_full['CPA_ANG'] > 350, 'CPA_ANG'] -= 360

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
                df = df[df["S/C"] == "A"]
            elif sat=="cor2_b":
                df = df[df["S/C"] == "B"]
            else:
                print("satellite not recognized")
        
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

def get_cdaw(folder,sat):
    '''
    This function downloads the cdaw catalogue and saves it in the folder specified
    '''
    columns=['Date','Time','CPA','WA','Linear_speed','initial_speed','final_speed','20R_speed','Accel','Mass','Kinetic_Energy','MPA','Remearks']
    folder=folder+"/cdaw_catalogue"
    if os.path.exists(folder) and os.path.isdir(folder):
        path=folder+'/'+'20240223_univ_all.txt'
        df=pd.read_fwf(path, sep='\t' ,skiprows=2)
        df = df.drop(df.columns[-1], axis=1)
        df.columns = columns
    return df

def get_cactus(folder,sat):
    general_path= folder+'/cactus_catalogue/'
    if sat=='lasco_c2':
        columns=['CME_ID','DATE','TIME','UNKNOWN','UNKNOWN_1','DATE_2','TIME_2','UNKNOWN_2','CPA','WA','MEDIAN_VEL','VEL_VARIATION','MIN_VEL','MAX_VEL','UNKNOWN_3','UNKNOWN_4','SATELITE','INSTRUMENT','DETECTOR']
        path = general_path+'cmecat_combo.sav'
        data_dict= readsav(path, idict=None)
        df_list=[]
        for c in range(len(data_dict['combined'][0])):
            for b in range(len(data_dict['combined'][0][0][0])):
                for a in range(len(data_dict['combined'][0][0][0][0])):
                    data_row = data_dict['combined'][0][0][0][0][a]
                    expanded_records = [item for item in data_row]
                    df_list.append(expanded_records)
        df = pd.DataFrame(df_list, columns=columns)
    else:
        path= general_path+'secchi_cmecat_combo.sav'
        data_dict = readsav(path, python_dict=True)
        if sat == 'cor2_a':
            df = pd.DataFrame(data_dict['secchia_combo'])
        elif sat=='cor2_b':
            df = pd.DataFrame(data_dict['secchib_combo'])

    for col in df.columns:
                if df[col].dtype == 'O' and any(isinstance(val, bytes) for val in df[col]):
                    df[col] = df[col].apply(lambda x: x.decode('utf-8').strip('b') if isinstance(x, bytes) else x)
    
    df['DATE_TIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    df_sorted = df.sort_values(by='DATE')
    
    return df_sorted

def comparator(NN,seeds,vourlidas,gcs,cactus):
    columns=["NN_DATE_TIME","SEEDS_DATE_TIME","VOURLIDAS_DATE_TIME","CACTUS_DATE_TIME","NN_CPA_ANG_MEDIAN","NN_CPA_ANG_STD","SEEDS_CPA_ANG","VOURLIDAS_CPA_ANG","CACTUS_CPA_ANG","NN_WIDE_ANG_MEDIAN","NN_WIDE_ANG_STD","SEEDS_WIDE_ANG","VOURLIDAS_WIDE_ANG","CACTUS_WIDE_ANG","GCS_CPA_ANG","GCS_WIDE_ANG"]
    NN_ang_col=['CPA_ANG', 'WIDE_ANG']

    compare=[]
    #converts to datetime objs and sort the df
    seeds['DATE_TIME'] = pd.to_datetime(seeds['DATE_TIME'], format="YYYY/MM/DD HH:MM:SS")
    seeds.sort_values(by='DATE_TIME', inplace=True)
    NN['DATE_TIME'] = pd.to_datetime(NN['DATE_TIME'])
    NN.sort_values(by='DATE_TIME', inplace=True)
    NN['DATE'] = NN['DATE_TIME'].dt.date
    NN['TIME'] = pd.to_datetime(NN['DATE_TIME']).dt.time

    vourlidas['Date_Time'] =vourlidas['Date']+" "+vourlidas['Time']
    vourlidas['Date_Time'] = pd.to_datetime(vourlidas['Date_Time'], format="mixed")
    vourlidas = vourlidas.sort_values(by='Date_Time')
    
    gcs['DATE_TIME'] = pd.to_datetime(gcs['DATE_TIME'])
    gcs = gcs.sort_values(by='DATE_TIME')

    cactus['DATE_TIME'] = pd.to_datetime(cactus['DATE_TIME'])
    cactus = cactus.sort_values(by='DATE_TIME')

    #adjust the 0 degree to the nort
    for i in NN_ang_col:
        if i=="WIDE_ANG":

            NN[i]=np.degrees([float(num) for num in NN[i]])
            gcs[i]=np.degrees(gcs[i])
        else:
            NN[i]= np.degrees([float(num) for num in NN[i]])-90#-np.degrees(NN[i])+270
            gcs[i]= np.degrees(gcs[i])-90
        NN.loc[NN[i] < 0, i] += 360  
        gcs.loc[gcs[i] < 0, i] += 360  

    #goups all the hours in each day and calculates medain a std of cpa_ang and wide_ang
    df = NN.groupby('DATE').agg({'WIDE_ANG': ['min', 'std'],'CPA_ANG': ['median', 'std'],})
    df = df.reset_index()
    #calculate the unique dates in NN and conservs only those ones in other catalogues
    unique_dates_NN = NN['DATE_TIME'].dt.date.unique()
    seeds = seeds[seeds['DATE_TIME'].dt.date.isin(unique_dates_NN)].reset_index()
    vourlidas = vourlidas[vourlidas['Date_Time'].dt.date.isin(unique_dates_NN)].reset_index()
    cactus = cactus[cactus['DATE_TIME'].dt.date.isin(unique_dates_NN)].reset_index()

    #keeps the first hour of everey day
    NN_min= NN.groupby('DATE')['TIME'].min().reset_index()
    NN_max= NN.groupby('DATE')['TIME'].max().reset_index()
    #creates the list to compare the events in NN and seeds
    for i in range(2,len(NN_min)-1):
        NN_min['DATE_TIME'] = NN_min.apply(lambda row: datetime.combine(row['DATE'], row['TIME']), axis=1)
        NN_max['DATE_TIME'] = NN_max.apply(lambda row: datetime.combine(row['DATE'], row['TIME']), axis=1)
        NN_date=NN_min["DATE_TIME"][i]
        #path=NN.loc[NN["DATE_TIME"]==NN_date, "PATH"].values[0]
        seeds['TIME_DIFF'] = seeds['DATE_TIME'] - pd.to_datetime(NN_date)
        time_diff_seeds = seeds[seeds['TIME_DIFF'] >= pd.Timedelta(0)]
        vourlidas['Time_Diff'] = vourlidas['Date_Time'] - pd.to_datetime(NN_date)
        time_diff_vourlidas = vourlidas[vourlidas['Time_Diff'] >= pd.Timedelta(0)]
        cactus['TIME_DIFF'] = cactus['DATE_TIME'] - pd.to_datetime(NN_date)
        time_diff_cactus = cactus[cactus['TIME_DIFF'] >= pd.Timedelta(0)]

        if not time_diff_seeds.empty and not time_diff_vourlidas.empty and not time_diff_cactus.empty :
            
            idx_seeds = time_diff_seeds['TIME_DIFF'].idxmin()
            seeds_data = seeds.loc[idx_seeds]
            cpa_ang_seeds=seeds_data["CPA_ANG"]
            wide_ang_seeds=seeds_data["WIDE_ANG"]
            idx_vourlidas = time_diff_vourlidas['Time_Diff'].idxmin()
            vourlidas_data = vourlidas.loc[idx_vourlidas]
            cpa_ang_vourlidas=vourlidas_data["CPA"]
            wide_ang_vourlidas=vourlidas_data["Width"]
            idx_cactus = time_diff_cactus['TIME_DIFF'].idxmin()
            cactus_data = cactus.loc[idx_cactus]
            cpa_ang_cactus=cactus_data["ANGLE"]
            wide_ang_cactus=cactus_data["WIDTH"]

            NN_prev=NN_min["DATE_TIME"][i]-pd.Timedelta(hours=4)
            NN_post=NN_max["DATE_TIME"][i]+pd.Timedelta(hours=4)
            gcs_data=gcs.loc[gcs["DATE_TIME"]==NN_date]
            if len(gcs_data)>0:
                cpa_ang_gcs = (gcs_data["CPA_ANG"].values)[0]
                wide_ang_gcs = (gcs_data["WIDE_ANG"].values)[0]
            else:
                cpa_ang_gcs = np.nan
                wide_ang_gcs = np.nan
            
            if (NN_prev<=vourlidas_data["Date_Time"]<=NN_post)and(NN_prev<=seeds_data["DATE_TIME"]<=NN_post)and(NN_prev<=cactus_data["DATE_TIME"]<=NN_post):
                compare.append([NN_date,seeds_data["DATE_TIME"],vourlidas_data["Date_Time"],cactus_data["DATE_TIME"],df["CPA_ANG"]["median"][i],df["CPA_ANG"]["std"][i],cpa_ang_seeds,cpa_ang_vourlidas,cpa_ang_cactus,df["WIDE_ANG"]["min"][i],df["WIDE_ANG"]["std"][i],wide_ang_seeds,wide_ang_vourlidas,wide_ang_cactus,cpa_ang_gcs,wide_ang_gcs])
            elif (NN_prev<=vourlidas_data["Date_Time"]<=NN_post)and(NN_prev<=cactus_data["DATE_TIME"]<=NN_post) and not(NN_prev<=seeds_data["DATE_TIME"]<=NN_post):
                compare.append([NN_date,np.nan,vourlidas_data["Date_Time"],cactus_data["DATE_TIME"],df["CPA_ANG"]["median"][i],df["CPA_ANG"]["std"][i],np.nan,cpa_ang_vourlidas,cpa_ang_cactus,df["WIDE_ANG"]["min"][i],df["WIDE_ANG"]["std"][i],np.nan,wide_ang_vourlidas,wide_ang_cactus,cpa_ang_gcs,wide_ang_gcs])
            elif not(NN_prev<=vourlidas_data["Date_Time"]<=NN_post)and(NN_prev<=seeds_data["DATE_TIME"]<=NN_post)and(NN_prev<=cactus_data["DATE_TIME"]<=NN_post):
                compare.append([NN_date,seeds_data["DATE_TIME"],np.nan,cactus_data["DATE_TIME"],df["CPA_ANG"]["median"][i],df["CPA_ANG"]["std"][i],cpa_ang_seeds,np.nan,cpa_ang_cactus,df["WIDE_ANG"]["min"][i],df["WIDE_ANG"]["std"][i],wide_ang_seeds,np.nan,wide_ang_cactus,cpa_ang_gcs,wide_ang_gcs])
            elif (NN_prev<=vourlidas_data["Date_Time"]<=NN_post)and(NN_prev<=seeds_data["DATE_TIME"]<=NN_post)and not(NN_prev<=cactus_data["DATE_TIME"]<=NN_post):
                compare.append([NN_date,seeds_data["DATE_TIME"],vourlidas_data["Date_Time"],np.nan,df["CPA_ANG"]["median"][i],df["CPA_ANG"]["std"][i],cpa_ang_seeds,cpa_ang_vourlidas,np.nan,df["WIDE_ANG"]["min"][i],df["WIDE_ANG"]["std"][i],wide_ang_seeds,wide_ang_vourlidas,np.nan,cpa_ang_gcs,wide_ang_gcs])
    compare = pd.DataFrame(compare, columns=columns)
    return compare

def linear(t,a,b):
    return a*t + b

def linear_error(p, x, y):
    return (linear(x, *p) - y)

def plot_fit(axis,xdf, ydf):
    '''
    This function adss a plot of a linear fit of the input axis
    and prints in the leyend the fit line equation and the pearson corr coeficient
    '''
    x = xdf.values
    y = ydf.values
    # remove nans in both arrays keeping the same indexes
    ok_ind = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x = x[ok_ind]
    y = y[ok_ind]
    #fit and plot with initial conditions [1,0] using least squares
    fit=least_squares(linear_error, [1,0], loss='soft_l1', kwargs={'x': x, 'y': y}) # fit that ingnore outliers   
    label = f"y={fit.x[0]:.2f}x+{fit.x[1]:.2f}\nR={np.corrcoef(x, y)[0,1]:.2f}"
    axis.plot(x, linear(x, *fit.x), color='red', linestyle='--', label=label, linewidth=1.5)
    return axis

def get_r(xdf,ydf):
    x = xdf.values
    y = ydf.values
    # remove nans in both arrays keeping the same indexes
    ok_ind = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x = x[ok_ind]
    y = y[ok_ind]   
    return f'r={np.corrcoef(x, y)[0,1]:.2f} ({len(x)})'  

def plot_all_in_one(df,plot_dir,x_axis,y_axis,line):
    '''
    Saves a comparission plot of x_axis catalogue vs all the catalogues in y_axis in one graphic.
    x_axis: one value for the x axis
    y_axis: a list of catalogues for y axis
    line: coords for correlation line
    '''
    colors=["blue","green","red","orange","yellow"]
    fig, ax = plt.subplots(figsize=(6, 6))
    if x_axis.endswith("CPA_ANG"):
        for i in range(len(y_axis)):
            if y_axis[i] =="NN":
                cpa_name=y_axis[i]+'_CPA_ANG_MEDIAN'
            else:
                cpa_name=y_axis[i]+'_CPA_ANG'
            
            label = y_axis[i]+';' +get_r(df[x_axis], df[cpa_name])
            ax.scatter(df[x_axis],df[cpa_name], color = colors[i], label=label)
            ax.plot(line[0], line[1], color='black', linestyle='-',linewidth=0.5)
            ax.set_xlim(0, ax.get_xlim()[1])
            ax.set_xlabel('GCS')
            ax.set_title('CPA [deg]')
            ax.legend()
            ax.grid(True)
        fig.savefig(plot_dir + '/CPA_ANG_all2.png', dpi=300, bbox_inches='tight')
        
    elif x_axis.endswith("WIDE_ANG"):
        for i in range(len(y_axis)):
            if y_axis[i] =="NN":
                wa_name=y_axis[i]+'_WIDE_ANG_MEDIAN'
            else:
                wa_name=y_axis[i]+'_WIDE_ANG'
            
            label = y_axis[i]+';' +get_r(df[x_axis], df[wa_name])
            ax.scatter(df[x_axis],df[wa_name], color=colors[i], label=label)
            ax.plot(line[0], line[1], color='black', linestyle='-',linewidth=0.5)
            ax.set_xlim(0, ax.get_xlim()[1])
            ax.set_xlabel('GCS')
            ax.set_title('WA [deg]')
            ax.legend()
            ax.grid(True)
        fig.savefig(plot_dir + '/WA_ANG_all2.png', dpi=300, bbox_inches='tight')

def plot_one_per_one(df,plot_dir,x_axis,y_axis,line):
    '''
    Saves a comparission plot of x_axis catalogue vs the catalogues in y_axis, with as much graphics as catalogues in y_axis.
    x_axis: one value for the x axis
    y_axis: a list of catalogues for y axis
    line: coords for correlation line
    '''
    amount=len(y_axis)
    fig3, ax= plt.subplots(1, amount, figsize=(18, 6))

    if x_axis.endswith("CPA_ANG"):
        for i in range(len(y_axis)):
            if y_axis[i] =="NN":
                cpa_name=y_axis[i]+'_CPA_ANG_MEDIAN'
            else:
                cpa_name=y_axis[i]+'_CPA_ANG'
            ax[i].errorbar(df[x_axis], df[cpa_name], fmt='o', color='blue', ecolor='gray', capsize=5, label=x_axis+' vs '+y_axis[i])
            plot_fit(ax[i],df[x_axis], df[cpa_name])
            ax[i].plot(line[0], line[1], color='black', linestyle='-',linewidth=0.5)
            ax[i].set_xlim(0, ax[i].get_xlim()[1])
            ax[i].set_xlabel(x_axis)
            ax[i].set_ylabel(y_axis[i])
            ax[i].set_title('CPA [deg]')
            ax[i].legend()
            ax[i].grid(True)
        fig3.savefig(plot_dir + '/CPA_ANG2.png', dpi=300, bbox_inches='tight')

    elif x_axis.endswith("WIDE_ANG"):
        for i in range(len(y_axis)):
            if y_axis[i] =="NN":
                wa_name=y_axis[i]+'_WIDE_ANG_MEDIAN'
            else:
                wa_name=y_axis[i]+'_WIDE_ANG'
              
            ax[i].errorbar(df[x_axis], df[wa_name], fmt='o', color='blue', ecolor='gray', capsize=5, label=x_axis+' vs '+y_axis[i])
            plot_fit(ax[i],df[x_axis], df[wa_name])
            ax[i].plot(line[0], line[1], color='black', linestyle='-',linewidth=0.5)
            ax[i].set_xlim(0, ax[i].get_xlim()[1])
            ax[i].set_xlabel(x_axis)
            ax[i].set_ylabel(y_axis[i])
            ax[i].set_title('WA [deg]')
            ax[i].legend()
            ax[i].grid(True)
        fig3.savefig(plot_dir + '/WIDE_ANG2.png', dpi=300, bbox_inches='tight')




################################################################################### MAIN ######################################################################################
odir="/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat_L1"
folder="/gehme-gpu/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn/neural_cme_seg/applications"
sat="cor2_b"#cor2_a/cor2_b/lasco_c2
correlation_cords=[[[[0,350],[0,350]],[[0,150],[0,150]]],[[[0,350],[0,350]],[[0,250],[0,250]]]]#[cor2_A[CPA,WA],cor2_b[CPA,WA]]

#-----------------
plot_dir=odir+'/'+sat+'_comparison'
os.makedirs(plot_dir, exist_ok=True)

if sat=='cor2_a':
    cactus = get_cactus(folder,sat)
    NN = get_NN(odir,sat)
    seeds = get_seeds(folder,sat)
    vourlidas = get_vourlidas(folder,sat)
    gcs = get_GCS(odir,sat)
    df=comparator(NN,seeds,vourlidas,gcs,cactus)
    cpa_corr=correlation_cords[0][0]
    wa_corr=correlation_cords[0][1]
    
elif sat=='cor2_b':
    cactus = get_cactus(folder,sat)
    NN = get_NN(odir,sat)
    seeds = get_seeds(folder,sat)
    vourlidas = get_vourlidas(folder,sat)
    gcs = get_GCS(odir,sat)
    df=comparator(NN,seeds,vourlidas,gcs,cactus)
    cpa_corr=correlation_cords[1][0]
    wa_corr=correlation_cords[1][1]

plot_all_in_one(df,plot_dir,"GCS_CPA_ANG",["VOURLIDAS","SEEDS","CACTUS","NN"],cpa_corr)
plot_all_in_one(df,plot_dir,"GCS_WIDE_ANG",["VOURLIDAS","SEEDS","CACTUS","NN"],wa_corr)
plot_one_per_one(df,plot_dir,"GCS_CPA_ANG",["VOURLIDAS","SEEDS","CACTUS","NN"],cpa_corr)
plot_one_per_one(df,plot_dir,"GCS_WIDE_ANG",["VOURLIDAS","SEEDS","CACTUS","NN"],wa_corr)
