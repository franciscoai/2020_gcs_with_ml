def plot_mask_prop(x,a,x_title='Date and hour',parameters=None):
        fig2, ax2 = plt.subplots()
        ax2.scatter(x, a, color='red', label=str(i.lower()))
        ax2.set_xlabel(x_title)
        ax2.set_title('Filtered dispersion graphic of '+str(i.lower()))
        ax2.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig2.savefig(data_dir+"/"+ext_folder+"/stats/"+str(i.lower())+"_filtered.png") 

def graphics(df):
    df["DATE_TIME"]= pd.to_datetime(df["DATE_TIME"])
    df = df.sort_values(by='DATE_TIME')
    #max,cpa and min angles graphic
    parameters=['WIDE_ANG','MASS_CENTER_RADIUS','MASS_CENTER_ANG','APEX_RADIUS','APEX_ANG']
    x = []  
    max_ang_y = []  
    min_ang_y = []
    cpa_ang_y = []  

    for idx, row in df.iterrows():
        date_time = row['DATE_TIME']
        max_ang = np.degrees(row['MAX_ANG'])
        min_ang = np.degrees(row['MIN_ANG'])
        cpa_ang = np.degrees(row['CPA_ANG'])
        x.append(date_time)
        max_ang_y.append(max_ang)
        min_ang_y.append(min_ang)
        cpa_ang_y.append(cpa_ang)

    fig, ax = plt.subplots()
    ax.scatter(x, max_ang_y, color='red', label='max_ang')
    ax.scatter(x, min_ang_y, color='blue', label='min_ang')
    ax.scatter(x, cpa_ang_y, color='green', label='cpa_ang')
    ax.set_xlabel('Date and hour')
    ax.set_ylabel('Angles')
    ax.set_title('Filtered dispersion graphic of max, cpa and min angles')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    fig.savefig(data_dir+"/"+ext_folder+"/stats/"+"max_cpa_min_angles_filtered.png") 

    #mass center coordinates graphic
    cm_y_list = []
    cm_x_list = []  

    for idx, row in df.iterrows():
        
        cm_x = row['MASS_CENTER_X']
        cm_y = row['MASS_CENTER_Y']
        cm_x_list.append(cm_x)
        cm_y_list.append(cm_y)

    fig1, ax1 = plt.subplots()
    ax1.scatter(cm_x_list, cm_y_list, color='red', label='max_ang')
    ax1.set_xlabel('x coordinates')
    ax1.set_ylabel('y coordinates')
    ax1.set_title('Filtered dispersion graphic of mass center coordinates')
    ax1.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig1.savefig(data_dir+"/"+ext_folder+"/stats/"+"mass_center_coordinates_filtered.png") 
    
    #generates graphics for the parameters
    for i in parameters:
        
        a=[]
        for idx, row in df.iterrows():
            if i.endswith("ANG"):        
                b = np.degrees(row[i])
            else:
                b = row[i]
            a.append(b)



def filter_mask(df,output_dir,folder):

    ofile=[]
    cpa_ang_y = []  

    for idx, row in df.iterrows():
        cpa_ang = row['CPA_ANG']
        cpa_ang_y.append(cpa_ang)
    cpa_median=np.median(cpa_ang_y)
    for i in cpa_ang_y:
        if np.degrees(cpa_median)-10 <= np.degrees(i) and np.degrees(i)<= np.degrees(cpa_median)+10:
            
            idx_df = df[df['CPA_ANG'] == i].index
            row_df= df.iloc[idx_df].values
            ofile.append(row_df[0])
    columnas = df.columns
    dataframe = pd.DataFrame(ofile, columns=columnas)
    dataframe.to_csv(output_dir+str(folder)+"/stats/"+str(folder)+"_filterd", index=False)

    graphics(dataframe)
    return dataframe


def quadratic(t,a,b,c):
    return a*t**2 + b*t + c


def sqrt_func(t,a,b,c):
    return np.sqrt(a*t + b) + c 




######################################################### MAIN ###################################################################
import pandas as pd
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data_dir='/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat_L1/cor2_a/'
a=0
b=0
c=0

ext_folders = os.listdir(data_dir)
for ext_folder in ext_folders:
    if ext_folder=="49":
        csv_path=data_dir+"/"+ext_folder+"/stats/"+ext_folder+"_stats"
        df=pd.read_csv(csv_path)
        df_filtered=filter_mask(df,data_dir,ext_folder)
        #timestamps = df_filtered['DATE_TIME'].apply(lambda x: x.timestamp())
        df_filtered['DATE_TIME'] = df_filtered['DATE_TIME'].apply(lambda x: x.timestamp())

        vel=df_filtered["APEX_RADIUS"].diff() / df_filtered['DATE_TIME'].diff()
        vel = [v for v in vel if not math.isinf(v)]
        b=np.median(vel)
        
        x_points=df_filtered["DATE_TIME"]
        y_points=df_filtered["APEX_RADIUS"]
        
        optp, _ = curve_fit(quadratic, x_points, y_points, p0=[a,b,c])
        a, b, t = optp
        chi2 = np.sqrt(np.mean((quadratic(x_points, a, b, t)-y_points)**2))
        x_line = np.arange(min(x_points), max(x_points), 1)
        y_line = quadratic(x_line, a, b, t)

        df["DATE_TIME"]= pd.to_datetime(df["DATE_TIME"])
      