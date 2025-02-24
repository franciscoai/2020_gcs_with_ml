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
    column_names=["DATE_TIME","MASK","SCORE","CPA","WA","APEX",'LABEL', 'BOX', 'CME_ID', 'APEX_DIST', 'CPA_DIST', 'WA_DIST', 'ERROR']
    df_list=[]
    odir=iodir+'/'+sat+"/"
    ext_folders = os.listdir(odir)
    for ext_folder in ext_folders:
        event_folder = os.listdir(odir+ext_folder)
        for event in event_folder:
            odir_filter=odir+ext_folder+"/"+event+"/filtered"
            if os.path.exists(odir_filter):
                csv_path=odir_filter+"/"+ext_folder+"_filtered_stats"
                if  os.path.exists(csv_path):
                    df=pd.read_csv(csv_path)
                    if len(df["CME_ID"].unique())>1:
                        predominant_cme = df["CME_ID"].value_counts().idxmax()
                        cme_event= df.loc[df["CME_ID"]==predominant_cme]
                        cme_event = cme_event.copy()
                        cme_event["FOLDER_NAME"] = event+".yht"
                        
                        df_list.append(cme_event)
                    else:
                        df["FOLDER_NAME"] = event+".yht"
                        df_list.append(df)
    
    df_full = pd.concat(df_list, ignore_index=True)
    breakpoint()
    return df_full


def get_cdaw(path,sat):
    '''
    This function downloads the cdaw catalogue and saves it in the specified folder 
    '''
    columns=['HEIGHT', 'DATE', 'TIME', 'ANGLE', 'TEL', 'FC', 'COL', 'ROW', 'VERSION',
       'DATE-OBS', 'TIME-OBS', 'DETECTOR', 'FILTER', 'OBSERVER', 'FEAT_CODE',
       'IMAGE_TYPE', 'YHT_ID', 'ORIG_HTFILE', 'ORIG_WDFILE', 'UNIVERSAL',
       'WDATA', 'HALO', 'ONSET1', 'ONSET2', 'ONSET2_RSUN', 'CPA', 'WA',
       'SPEED', 'ACCEL', 'FEAT_PA', 'FEAT_QUAL', 'QUALITY_INDEX', 'REMARK',
       'FOLDER_NAME', 'DATE_TIME']
    
    df = pd.read_csv(path+"full_cdaw_catalogue.csv", low_memory=False)
    df.columns=columns
    df = df.drop(df[df["CPA"] == "HALO"].index)
    df["WA"] = df["WA"].str.replace(">", "")
    
   
    return df

def get_cactus(folder,sat):
    general_path= folder+'/cactus_catalogue/'
    if sat=='lasco':
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

    for col in df.columns:
                if df[col].dtype == 'O' and any(isinstance(val, bytes) for val in df[col]):
                    df[col] = df[col].apply(lambda x: x.decode('utf-8').strip('b') if isinstance(x, bytes) else x)
    
    df['DATE_TIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    df_sorted = df.sort_values(by='DATE')
    
    return df_sorted

def comparator(NN,cdaw,cactus=None):
    columns=["NN_DATE_TIME","CDAW_DATE_TIME","CACTUS_DATE_TIME","NN_MEDIAN_CPA_ANG","NN_STD_CPA_ANG","CDAW_CPA_MEDIAN","CDAW_WA_MEDIAN","CACTUS_CPA_ANG","NN_MEDIAN_WA","NN_STD_WA","CDAW_CPA_STD","CDAW_WA_STD","CACTUS_WA"]
    NN_ang_col=['CPA', 'AW']
    compare=[]

    #converts to datetime objs and sort the df
    NN['DATE_TIME'] = pd.to_datetime(NN['DATE_TIME'])
    breakpoint()
    NN.sort_values(by='FOLDER_NAME', inplace=True)
    NN['DATE'] = NN['DATE_TIME'].dt.date
    NN['TIME'] = pd.to_datetime(NN['DATE_TIME']).dt.time
    cdaw['DATE_TIME'] = pd.to_datetime(cdaw['DATE_TIME'], format="%Y-%m-%d %H:%M:%S")
    cdaw.sort_values(by='FOLDER_NAME', inplace=True)

    #adjust the 0 degree to the nort
    for i in NN_ang_col:
        if i=="AW":
            NN[i]=np.degrees([float(num) for num in NN[i]])
        else:
            NN[i]= np.degrees([float(num) for num in NN[i]])-90
        NN.loc[NN[i] < 0, i] += 360  
    
    #goups all the hours in each day and calculates medain a std of cpa_ang and wide_ang
    NN_stats = NN.groupby(['FOLDER_NAME','DATE']).agg({'AW': ['min', 'std'],'CPA': ['median', 'std'],})
    NN_stats = NN_stats.reset_index()
    #calculate the unique dates in NN and conservs only those ones in other catalogues
    unique_dates_NN = NN['DATE_TIME'].dt.date.unique()
    cdaw = cdaw[cdaw['DATE_TIME'].dt.date.isin(unique_dates_NN)].reset_index()
    
    #keeps the first hour of everey day
    NN_min= NN.groupby(['FOLDER_NAME','DATE'])['TIME'].min().reset_index()
    NN_max= NN.groupby(['FOLDER_NAME','DATE'])['TIME'].max().reset_index()
    
    if cactus != None:
        cactus['DATE_TIME'] = pd.to_datetime(cactus['DATE_TIME'])
        cactus = cactus.sort_values(by='DATE_TIME')
        cactus = cactus[cactus['DATE_TIME'].dt.date.isin(unique_dates_NN)].reset_index()
    
    #creates the list to compare the events in NN and cdaw
    for i in range(len(NN_min)-1):
        NN_min['DATE_TIME'] = NN_min.apply(lambda row: datetime.combine(row['DATE'], row['TIME']), axis=1)
        NN_max['DATE_TIME'] = NN_max.apply(lambda row: datetime.combine(row['DATE'], row['TIME']), axis=1)
        NN_date=NN_min["DATE_TIME"][i]
        cme_event=NN_min["FOLDER_NAME"].loc[i]
        cdaw_cme_event = cdaw.loc[cdaw["FOLDER_NAME"] == cme_event].copy()
        

        if cactus != None:
            cactus['TIME_DIFF'] = cactus['DATE_TIME'] - pd.to_datetime(NN_date)
            time_diff_cactus = cactus[cactus['TIME_DIFF'] >= pd.Timedelta(0)]
    
        cdaw_cme_event['CPA'] = pd.to_numeric(cdaw_cme_event['CPA'], errors='coerce')
        cdaw_cme_event['WA'] = pd.to_numeric(cdaw_cme_event['WA'], errors='coerce')
        
        cpa_median_cdaw=cdaw_cme_event["CPA"].median()
        wa_median_cdaw=cdaw_cme_event["WA"].median()
        cpa_std_cdaw=cdaw_cme_event["CPA"].std()
        wa_std_cdaw=cdaw_cme_event["WA"].std()
        
        NN_prev=NN_min["DATE_TIME"][i]-pd.Timedelta(hours=4)
        NN_post=NN_max["DATE_TIME"][i]+pd.Timedelta(hours=4)
        if cactus != None:
            if not time_diff_cactus.empty: 
                idx_cactus = time_diff_cactus['TIME_DIFF'].idxmin()
                cactus_data = cactus.loc[idx_cactus]
                cpa_ang_cactus=cactus_data["CPA"]
                wide_ang_cactus=cactus_data["WA"]
                if (NN_prev<=cdaw_data["DATE_TIME"]<=NN_post)and(NN_prev<=cactus_data["DATE_TIME"]<=NN_post):
                    compare.append([NN_date,cdaw_data["DATE_TIME"],cactus_data["DATE_TIME"],NN_stats["CPA"]["median"][i],NN_stats["CPA"]["std"][i],cpa_ang_cdaw,cpa_ang_cactus,NN_stats["AW"]["min"][i],NN_stats["AW"]["std"][i],wide_ang_cdaw,wide_ang_cactus])
                elif not(NN_prev<=cdaw_data["DATE_TIME"]<=NN_post)and(NN_prev<=cactus_data["DATE_TIME"]<=NN_post):
                    compare.append([NN_date,np.nan,cactus_data["DATE_TIME"],NN_stats["CPA"]["median"][i],NN_stats["CPA"]["std"][i],np.nan,cpa_ang_cactus,NN_stats["AW"]["min"][i],NN_stats["AW"]["std"][i],np.nan,wide_ang_cactus])
                elif (NN_prev<=cdaw_data["DATE_TIME"]<=NN_post)and not(NN_prev<=cactus_data["DATE_TIME"]<=NN_post):
                    compare.append([NN_date,NN_date,np.nan,NN_stats["CPA"]["median"][i],NN_stats["CPA"]["std"][i],cpa_ang_cdaw,np.nan,NN_stats["AW"]["min"][i],NN_stats["AW"]["std"][i],wide_ang_cdaw,np.nan])
        else:
            
            compare.append([NN_date,NN_date,np.nan,NN_stats["CPA"]["median"][i],NN_stats["CPA"]["std"][i],cpa_median_cdaw,wa_median_cdaw,np.nan,NN_stats["AW"]["min"][i],NN_stats["AW"]["std"][i],cpa_std_cdaw,wa_std_cdaw,np.nan])
    
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
                cpa_name=y_axis[i]+'_MEDIAN_CPA_ANG'
            else:
                cpa_name=y_axis[i]+'_CPA_MEDIAN'

            df[cpa_name] = df[cpa_name].astype(float)
            label = y_axis[i]+';' +get_r(df[x_axis], df[cpa_name])
            ax.scatter(df[x_axis],df[cpa_name], color = colors[i], label=label)
            ax.plot(line[0], line[1], color='black', linestyle='-',linewidth=0.5)
            ax.set_xlim(0, ax.get_xlim()[1])
            ax.set_xlabel('NN')
            ax.set_title('CPA [deg]')
            ax.legend()
            ax.grid(True)
        fig.savefig(plot_dir + '/CPA_ANG_all.png', dpi=300, bbox_inches='tight')
        
    elif x_axis.endswith("WA"):
        for i in range(len(y_axis)):
            if y_axis[i] =="NN":
                wa_name=y_axis[i]+'_MEDIAN_WA'
            else:
                wa_name=y_axis[i]+'_WA_MEDIAN'
  
            df[wa_name] = df[wa_name].astype(float)
        
            label = y_axis[i]+';' +get_r(df[x_axis], df[wa_name])
            ax.scatter(df[x_axis],df[wa_name], color=colors[i], label=label)
            ax.plot(line[0], line[1], color='black', linestyle='-',linewidth=0.5)
            ax.set_xlim(0, ax.get_xlim()[1])
            ax.set_xlabel('NN')
            ax.set_title('WA [deg]')
            ax.legend()
            ax.grid(True)
        fig.savefig(plot_dir + '/WIDE_ANG_all.png', dpi=300, bbox_inches='tight')

def plot_one_per_one(df,plot_dir,x_axis,y_axis,line):
    '''
    Saves a comparission plot of x_axis catalogue vs the catalogues in y_axis, with as much graphics as catalogues in y_axis.
    x_axis: one value for the x axis
    y_axis: a list of catalogues for y axis
    line: coords for correlation line
    '''
    amount=len(y_axis)
    if amount>1:
        fig3, ax= plt.subplots(1, amount, figsize=(18, 6))
    else:
        fig3, ax= plt.subplots(1, figsize=(6, 6))

    if x_axis.endswith("CPA_ANG"):
        for i in range(len(y_axis)):
            if y_axis[i] =="NN":
                cpa_name=y_axis[i]+'_MEDIAN_CPA_ANG'
            else:
                cpa_name=y_axis[i]+'_CPA_MEDIAN'
            df[cpa_name] = df[cpa_name].astype(float)
            if amount>1:
                ax[i].errorbar(df[x_axis], df[cpa_name], fmt='o', color='blue', ecolor='gray', capsize=5, label=x_axis+' vs '+y_axis[i])
                plot_fit(ax[i],df[x_axis], df[cpa_name])
                ax[i].plot(line[0], line[1], color='black', linestyle='-',linewidth=0.5)
                ax[i].set_xlim(0, ax[i].get_xlim()[1])
                ax[i].set_xlabel(x_axis)
                ax[i].set_ylabel(y_axis[i])
                ax[i].set_title('CPA [deg]')
                ax[i].legend()
                ax[i].grid(True)
            else:
                ax.errorbar(df[x_axis], df[cpa_name], fmt='o', color='blue', ecolor='gray', capsize=5, label=x_axis+' vs '+y_axis[i])
                plot_fit(ax,df[x_axis], df[cpa_name])
                ax.plot(line[0], line[1], color='black', linestyle='-',linewidth=0.5)
                ax.set_xlim(0, ax.get_xlim()[1])
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis[i])
                ax.set_title('CPA [deg]')
                ax.legend()
                ax.grid(True)    
        fig3.savefig(plot_dir + '/CPA_ANG.png', dpi=300, bbox_inches='tight')

    elif x_axis.endswith("WA"):
        for i in range(len(y_axis)):
            if y_axis[i] =="NN":
                wa_name=y_axis[i]+'_MEDIAN_WA'
            else:
                wa_name=y_axis[i]+'_WA_MEDIAN'
            df[wa_name] = df[wa_name].astype(float)  
            if amount>1:
                ax[i].errorbar(df[x_axis], df[wa_name], fmt='o', color='blue', ecolor='gray', capsize=5, label=x_axis+' vs '+y_axis[i])
                plot_fit(ax[i],df[x_axis], df[wa_name])
                ax[i].plot(line[0], line[1], color='black', linestyle='-',linewidth=0.5)
                ax[i].set_xlim(0, ax[i].get_xlim()[1])
                ax[i].set_xlabel(x_axis)
                ax[i].set_ylabel(y_axis[i])
                ax[i].set_title('WA [deg]')
                ax[i].legend()
                ax[i].grid(True)
            else:
                ax.errorbar(df[x_axis], df[wa_name], fmt='o', color='blue', ecolor='gray', capsize=5, label=x_axis+' vs '+y_axis[i])
                plot_fit(ax,df[x_axis], df[wa_name])
                ax.plot(line[0], line[1], color='black', linestyle='-',linewidth=0.5)
                ax.set_xlim(0, ax.get_xlim()[1])
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis[i])
                ax.set_title('WA [deg]')
                ax.legend()
                ax.grid(True)    
        fig3.savefig(plot_dir + '/WIDE_ANG.png', dpi=300, bbox_inches='tight')


################################################################################### MAIN ######################################################################################
odir="/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v5/infer_neural_cme_seg_kincat_L1"
folder="/gehme-gpu/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn/neural_cme_seg/applications"
cdaw_path='/gehme/data/catalogues/soho/lasco/'

sat="lasco"
correlation_cords=[[[0,350],[0,350]],[[0,250],[0,250]]]#[lasco[CPA,WA]]

#-----------------
plot_dir=odir+'/'+sat+'_comparison'
os.makedirs(plot_dir, exist_ok=True)

NN = get_NN(odir,sat)
cdaw =get_cdaw(cdaw_path,sat)
#cactus = get_cactus(folder,sat)

df = comparator(NN,cdaw)#cactus is an optional arg
cpa_corr=correlation_cords[0]
wa_corr=correlation_cords[1]
plot_all_in_one(df,plot_dir,"NN_MEDIAN_CPA_ANG",["CDAW"],cpa_corr)# plot_all_in_one(df,plot_dir,"NN_MEDIAN_CPA_ANG",["CDAW","CACTUS"],cpa_corr)
plot_all_in_one(df,plot_dir,"NN_MEDIAN_WA",["CDAW"],wa_corr)
plot_one_per_one(df,plot_dir,"NN_MEDIAN_CPA_ANG",["CDAW"],cpa_corr)
plot_one_per_one(df,plot_dir,"NN_MEDIAN_WA",["CDAW"],wa_corr)