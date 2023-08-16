import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_dir='/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat_L1/cor2_a'
parameters=['WIDE_ANG','MASS_CENTER_RADIUS','MASS_CENTER_ANG','APEX_RADIUS','APEX_ANG']
fa=0
ext_folders = os.listdir(data_dir)
for ext_folder in ext_folders:
    if ext_folder=="35":
        print("Working on folder "+ext_folder+", folder "+str(fa)+" of "+str(len(ext_folder)-1))
        csv_path=data_dir+"/"+ext_folder+"/stats/"+ext_folder+"_stats"
        df=pd.read_csv(csv_path)
        df["DATE_TIME"]= pd.to_datetime(df["DATE_TIME"])
        
        #max,cpa and min angles graphic
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
        ax.set_title('Dispersion graphic of max, cpa and min angles')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(data_dir+"/"+ext_folder+"/stats/"+"max_cpa_min_angles.png") 

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
        ax1.set_title('Dispersion graphic of mass center coordinates')
        ax1.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig1.savefig(data_dir+"/"+ext_folder+"/stats/"+"mass_center_coordinates.png") 
        
        #generates graphics for the parameters
        for i in parameters:
            
            a=[]
            for idx, row in df.iterrows():
                if i.endswith("ANG"):        
                    b = np.degrees(row[i])
                else:
                    b = row[i]
                a.append(b)
            fig2, ax2 = plt.subplots()
            ax2.scatter(x, a, color='red', label=str(i.lower()))
            ax2.set_xlabel('Date and hour')
            ax2.set_title('Dispersion graphic of '+str(i.lower()))
            ax2.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            fig2.savefig(data_dir+"/"+ext_folder+"/stats/"+str(i.lower())+".png") 


