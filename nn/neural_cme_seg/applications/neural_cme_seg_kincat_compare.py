
import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import timedelta, datetime, time
from scipy.optimize import least_squares

def get_NN(iodir,sat):
    column_names=["DATE_TIME","MASK","SCORE","CPA_ANG","WIDE_ANG","APEX",'LABEL', 'BOX', 'CME_ID', 'APEX_DIST', 'CPA_DIST', 'WA_DIST', 'ERROR']
    df_list=[]
    odir=iodir+'/'+sat+"/"
    ext_folders = os.listdir(odir)
    for ext_folder in ext_folders:
        odir_filter=odir+ext_folder+"/filtered"
        if os.path.exists(odir_filter):
            csv_path=odir_filter+"/"+ext_folder+"_filtered_stats"
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
    breakpoint()
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
    breakpoint()
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


def comparator(NN,seeds,vourlidas,gcs):
    columns=["NN_DATE_TIME","SEEDS_DATE_TIME","VOURLIDAS_DATE_TIME","NN_CPA_ANG_MEDIAN","NN_CPA_ANG_STD","SEEDS_CPA_ANG","VOURLIDAS_CPA_ANG","NN_WIDE_ANG_MEDIAN","NN_WIDE_ANG_STD","SEEDS_WIDE_ANG","VOURLIDAS_WIDE_ANG","GCS_CPA_ANG","GCS_WIDE_ANG"]
    NN_ang_col=['CPA_ANG', 'WIDE_ANG']
    
    compare=[]
    #converts to datetime objs and sort the df
    seeds['DATE_TIME'] = pd.to_datetime(seeds['DATE_TIME'], format="YYYY/MM/DD HH:MM:SS")
    NN['DATE_TIME'] = pd.to_datetime(NN['DATE_TIME'])
    vourlidas['Date_Time'] =vourlidas['Date']+" "+vourlidas['Time']
    vourlidas['Date_Time'] = pd.to_datetime(vourlidas['Date_Time'], format="mixed")
    vourlidas = vourlidas.sort_values(by='Date_Time')
    NN.sort_values(by='DATE_TIME', inplace=True)
    seeds.sort_values(by='DATE_TIME', inplace=True)
    NN['DATE'] = NN['DATE_TIME'].dt.date
    NN['TIME'] = pd.to_datetime(NN['DATE_TIME']).dt.time
    gcs['DATE_TIME'] = pd.to_datetime(gcs['DATE_TIME'])
    gcs = gcs.sort_values(by='DATE_TIME')
    
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
    #calculate the unique dates in NN and conservs only those ones in seeds
    unique_dates_NN = NN['DATE_TIME'].dt.date.unique()
    seeds = seeds[seeds['DATE_TIME'].dt.date.isin(unique_dates_NN)].reset_index()
    vourlidas = vourlidas[vourlidas['Date_Time'].dt.date.isin(unique_dates_NN)].reset_index()
    
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
        if not time_diff_seeds.empty and not time_diff_vourlidas.empty:
            
            idx_seeds = time_diff_seeds['TIME_DIFF'].idxmin()
            seeds_data = seeds.loc[idx_seeds]
            cpa_ang_seeds=seeds_data["CPA_ANG"]
            wide_ang_seeds=seeds_data["WIDE_ANG"]
            idx_vourlidas = time_diff_vourlidas['Time_Diff'].idxmin()
            vourlidas_data = vourlidas.loc[idx_vourlidas]
            cpa_ang_vourlidas=vourlidas_data["CPA"]
            wide_ang_vourlidas=vourlidas_data["Width"]
            NN_prev=NN_min["DATE_TIME"][i]-pd.Timedelta(hours=4)
            NN_post=NN_max["DATE_TIME"][i]+pd.Timedelta(hours=4)
            gcs_data=gcs.loc[gcs["DATE_TIME"]==NN_date]
            if len(gcs_data)>0:
                cpa_ang_gcs = (gcs_data["CPA_ANG"].values)[0]
                wide_ang_gcs = (gcs_data["WIDE_ANG"].values)[0]
            else:
                cpa_ang_gcs = np.nan
                wide_ang_gcs = np.nan
            
            if (NN_prev<=vourlidas_data["Date_Time"]<=NN_post)and(NN_prev<=seeds_data["DATE_TIME"]<=NN_post):
                compare.append([NN_date,seeds_data["DATE_TIME"],vourlidas_data["Date_Time"],df["CPA_ANG"]["median"][i],df["CPA_ANG"]["std"][i],cpa_ang_seeds,cpa_ang_vourlidas,df["WIDE_ANG"]["min"][i],df["WIDE_ANG"]["std"][i],wide_ang_seeds,wide_ang_vourlidas,cpa_ang_gcs,wide_ang_gcs])
            elif (NN_prev<=vourlidas_data["Date_Time"]<=NN_post) and not(NN_prev<=seeds_data["DATE_TIME"]<=NN_post):
                compare.append([NN_date,np.nan,vourlidas_data["Date_Time"],df["CPA_ANG"]["median"][i],df["CPA_ANG"]["std"][i],np.nan,cpa_ang_vourlidas,df["WIDE_ANG"]["min"][i],df["WIDE_ANG"]["std"][i],np.nan,wide_ang_vourlidas,cpa_ang_gcs,wide_ang_gcs])
            elif not(NN_prev<=vourlidas_data["Date_Time"]<=NN_post)and(NN_prev<=seeds_data["DATE_TIME"]<=NN_post):
                
                compare.append([NN_date,seeds_data["DATE_TIME"],np.nan,df["CPA_ANG"]["median"][i],df["CPA_ANG"]["std"][i],cpa_ang_seeds,np.nan,df["WIDE_ANG"]["min"][i],df["WIDE_ANG"]["std"][i],wide_ang_seeds,np.nan,cpa_ang_gcs,wide_ang_gcs])
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
################################################################################### MAIN ######################################################################################
odir="/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat_L1"
folder="/gehme-gpu/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn/neural_cme_seg/applications"
sat="cor2_b"#cor2_a

#-----------------
plot_dir=odir+'/'+sat+'_comparison'
os.makedirs(plot_dir, exist_ok=True)

NN=get_NN(odir,sat)
seeds=get_seeds(folder,sat)
vourlidas= get_vourlidas(folder,sat)
gcs=get_GCS(odir,sat)
df=comparator(NN,seeds,vourlidas,gcs)

# date_to_tag_vourlidas = pd.to_datetime([datetime(2008, 5, 17)])
# date_to_tag_seeds = pd.to_datetime([datetime(2008, 5, 17)])
# date_to_tag_nn = pd.to_datetime([datetime(2008, 5, 17)])

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(df["GCS_CPA_ANG"],df["VOURLIDAS_CPA_ANG"], color='green')#, label=label)
ax.scatter(df["GCS_CPA_ANG"], df["NN_CPA_ANG_MEDIAN"], color='red')#, label=label)
ax.scatter(df["GCS_CPA_ANG"], df["SEEDS_CPA_ANG"], color='blue')#, label=label)
ax.plot([0, 450], [0, 450], color='black', linestyle='-',linewidth=0.5)
ax.set_xlim(0, ax.get_xlim()[1])
ax.set_xlabel('GCS')
ax.set_title('CPA [deg]')
ax.legend()
ax.grid(True)
fig.savefig(plot_dir + '/CPA_ANG_all2.png', dpi=300, bbox_inches='tight')
breakpoint()

# fig, ax = plt.subplots(figsize=(6, 6))
# label = 'VOURLIDAS ; '+get_r(df["GCS_CPA_ANG"], df["VOURLIDAS_CPA_ANG"])
# #ax.errorbar(df["NN_CPA_ANG_MEDIAN"], df["VOURLIDAS_CPA_ANG"], xerr=df["NN_CPA_ANG_STD"], fmt='o', color='green', ecolor='gray', capsize=5, label=label)
# # add a tag to the plot

# ax.scatter(df["GCS_CPA_ANG"],df["VOURLIDAS_CPA_ANG"], color='green', label=label)
# for date in date_to_tag_vourlidas:
#     x = df.loc[df["VOURLIDAS_DATE_TIME"].dt.date==date]["GCS_CPA_ANG"]
#     y = df.loc[df["VOURLIDAS_DATE_TIME"].dt.date==date]["VOURLIDAS_CPA_ANG"]
#     if len(x) > 0:
#         ax.text(x, y, date.strftime('%Y-%m-%d'), ha='left', va='top')
# label = 'SEEDS ; '+get_r(df["GCS_CPA_ANG"], df["SEEDS_CPA_ANG"])
# #ax.errorbar(df["GCS_CPA_ANG"], df["SEEDS_CPA_ANG"], xerr=df["GCS_CPA_ANG"], fmt='o', color='blue', ecolor='gray', capsize=5, label=label)
# ax.scatter(df["GCS_CPA_ANG"], df["SEEDS_CPA_ANG"], color='blue', label=label)
# # add a tag to the plot
# for date in date_to_tag_seeds:
#     x = df.loc[df["SEEDS_DATE_TIME"].dt.date==date]["GCS_CPA_ANG"]
#     y = df.loc[df["SEEDS_DATE_TIME"].dt.date==date]["SEEDS_CPA_ANG"]
#     if len(x) > 0:
#         ax.plot(x,y)
#         ax.text(x, y, date.strftime('%Y-%m-%d'), ha='left', va='top')
# label = 'NN ; '+get_r(df["GCS_CPA_ANG"],df["NN_CPA_ANG_MEDIAN"])
# ax.scatter(df["GCS_CPA_ANG"], df["NN_CPA_ANG_MEDIAN"], color='red', label=label)
# for date in date_to_tag_nn:
#     x = df.loc[df["NN_DATE_TIME"].dt.date==date]["GCS_CPA_ANG"]
#     y = df.loc[df["NN_DATE_TIME"].dt.date==date]["NN_CPA_ANG_MEDIAN"]
#     if len(x) > 0:
#         ax.text(x, y, date.strftime('%Y-%m-%d'), ha='left', va='top')


# ax.plot([0, 450], [0, 450], color='black', linestyle='-',linewidth=0.5)
# ax.set_xlim(0, ax.get_xlim()[1])
# ax.set_xlabel('GCS')
# ax.set_title('CPA [deg]')
# ax.legend()
# ax.grid(True)
# fig.savefig(plot_dir + '/CPA_ANG_all2.png', dpi=300, bbox_inches='tight')

# fig2, ax2 = plt.subplots(figsize=(6, 6))
# label = 'VOURLIDAS ; '+get_r(df["GCS_WIDE_ANG"], df["VOURLIDAS_WIDE_ANG"])
# ax2.scatter(df["GCS_WIDE_ANG"],df["VOURLIDAS_WIDE_ANG"], color='green', label=label)
# #ax2.errorbar(df["GCS_WIDE_ANG"], df["VOURLIDAS_WIDE_ANG"], xerr=df["NN_WIDE_ANG_STD"], fmt='o', color='green', ecolor='gray', capsize=5, label=label)
# # add a tag to the plot
# for date in date_to_tag_vourlidas:
#     x = df.loc[df["VOURLIDAS_DATE_TIME"].dt.date==date]["GCS_WIDE_ANG"]
#     y = df.loc[df["VOURLIDAS_DATE_TIME"].dt.date==date]["VOURLIDAS_WIDE_ANG"]
#     if len(x) > 0:
#         ax2.text(x, y, date.strftime('%Y-%m-%d'), ha='left', va='top')
# label = 'SEEDS ; '+get_r(df["GCS_WIDE_ANG"], df["SEEDS_WIDE_ANG"])
# ax2.scatter(df["GCS_WIDE_ANG"], df["SEEDS_WIDE_ANG"], color='blue', label=label)
# #ax2.errorbar(df["NN_WIDE_ANG_MEDIAN"], df["SEEDS_WIDE_ANG"], xerr=df["NN_WIDE_ANG_STD"], fmt='o', color='blue', ecolor='gray', capsize=5, label=label)
# # add a tag to the plot
# for date in date_to_tag_seeds:
#     x = df.loc[df["SEEDS_DATE_TIME"].dt.date==date]["GCS_WIDE_ANG"]
#     y = df.loc[df["SEEDS_DATE_TIME"].dt.date==date]["SEEDS_WIDE_ANG"]
#     if len(x) > 0:
#         ax2.text(x, y, date.strftime('%Y-%m-%d'), ha='left', va='top')

# label = 'NN ; '+get_r(df["GCS_WIDE_ANG"],df["NN_WIDE_ANG_MEDIAN"])
# ax2.scatter(df["GCS_WIDE_ANG"], df["NN_WIDE_ANG_MEDIAN"], color='red', label=label)
# for date in date_to_tag_nn:
#     x = df.loc[df["NN_DATE_TIME"].dt.date==date]["GCS_WIDE_ANG"]
#     y = df.loc[df["NN_DATE_TIME"].dt.date==date]["NN_WIDE_ANG_MEDIAN"]
#     if len(x) > 0:
#         ax.text(x, y, date.strftime('%Y-%m-%d'), ha='left', va='top')

# ax2.plot([0, 250], [0, 250], color='black', linestyle='-',linewidth=0.5)
# ax2.set_xlim(0, ax2.get_xlim()[1])
# ax2.set_xlabel('GCS')
# ax2.set_title('AW [deg]')
# ax2.legend()
# ax2.grid(True)
# fig2.savefig(plot_dir + '/WIDE_ANG_all2.png', dpi=300, bbox_inches='tight')


# fig3, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(18, 6))
# ax3.errorbar(df["GCS_WIDE_ANG"], df["VOURLIDAS_WIDE_ANG"], fmt='o', color='blue', ecolor='gray', capsize=5, label='GCS vs VOURLIDAS')
# plot_fit(ax3,df["GCS_WIDE_ANG"], df["VOURLIDAS_WIDE_ANG"])
# ax3.plot([0, 350], [0, 350], color='black', linestyle='-',linewidth=0.5)
# ax3.set_xlim(0, ax3.get_xlim()[1])
# ax3.set_xlabel('GCS')
# ax3.set_ylabel('Vourlidas')
# ax3.set_title('AW [deg]')
# ax3.legend()
# ax3.grid(True)
# ax4.errorbar( df["GCS_WIDE_ANG"], df["SEEDS_WIDE_ANG"], fmt='o', color='blue', ecolor='gray', capsize=5, label='GCS vs SEEDS')
# plot_fit(ax4,df["GCS_WIDE_ANG"], df["SEEDS_WIDE_ANG"])
# ax4.plot([0, 450], [0, 450], color='black', linestyle='-',linewidth=0.5)
# ax4.set_xlim(0, ax4.get_xlim()[1])
# ax4.set_ylabel('Seeds')
# ax4.set_xlabel('GCS')
# ax4.set_title('AW [deg]')
# ax4.legend()
# ax4.grid(True)
# ax5.errorbar(df["GCS_WIDE_ANG"],df["NN_WIDE_ANG_MEDIAN"], fmt='o', color='blue', ecolor='gray', capsize=5, label='GCS vs NN')
# plot_fit(ax5,df["GCS_WIDE_ANG"],df["NN_WIDE_ANG_MEDIAN"])
# ax5.plot([0, 550], [0, 550], color='black', linestyle='-',linewidth=0.5)
# ax5.set_xlim(0, ax5.get_xlim()[1])
# ax5.set_ylabel('NN')
# ax5.set_xlabel('GCS')
# ax5.set_title('AW [deg]')
# ax5.legend()
# ax5.grid(True)
# fig3.savefig(plot_dir + '/WIDE_ANG2.png', dpi=300, bbox_inches='tight')
             
# fig4, (ax6, ax7, ax8) = plt.subplots(1, 3, figsize=(18, 6))
# ax6.errorbar(df["GCS_CPA_ANG"], df["VOURLIDAS_CPA_ANG"], fmt='o', color='blue', ecolor='gray', capsize=5, label='GCS vs SEEDS')
# plot_fit(ax6,df["GCS_CPA_ANG"], df["VOURLIDAS_CPA_ANG"])
# ax6.plot([0, 650], [0, 650], color='black', linestyle='-',linewidth=0.5)
# ax6.set_xlim(0, ax6.get_xlim()[1])
# ax6.set_xlabel('GCS')
# ax6.set_ylabel('Vourlidas')
# ax6.set_title('CPA [deg]')
# ax6.legend()
# ax6.grid(True)
# ax7.errorbar( df["GCS_CPA_ANG"], df["SEEDS_CPA_ANG"], fmt='o', color='blue', ecolor='gray', capsize=5, label='GCS vs SEEDS')
# plot_fit(ax7,df["GCS_CPA_ANG"], df["SEEDS_CPA_ANG"])
# ax7.plot([0, 500], [0, 500], color='black', linestyle='-',linewidth=0.5)
# ax7.set_xlim(0, ax7.get_xlim()[1])
# ax7.set_ylabel('Seeds')
# ax7.set_xlabel('GCS')
# ax7.set_title('CPA [deg]')
# ax7.legend()
# ax7.grid(True)
# # print thge date of all events that have Seeds/NN CPA > 1.5
# print(f'Seeds/GCS CPA>1.5 \n',df.loc[df["SEEDS_CPA_ANG"]/df["GCS_CPA_ANG"]>1.5]["NN_DATE_TIME"])
# print(f'Seeds/GCS CPA<0.5 \n',df.loc[df["SEEDS_CPA_ANG"]/df["GCS_CPA_ANG"]<0.5]["NN_DATE_TIME"])
# ax8.errorbar(df["GCS_CPA_ANG"],df["NN_CPA_ANG_MEDIAN"], fmt='o', color='blue', ecolor='gray', capsize=8, label='GCS vs NN')
# plot_fit(ax8,df["GCS_CPA_ANG"],df["NN_CPA_ANG_MEDIAN"])
# ax8.plot([0, 500], [0, 500], color='black', linestyle='-',linewidth=0.5)
# #ax8.set_xlim(0, ax8.get_xlim()[1])
# ax8.set_ylabel('NN')
# ax8.set_xlabel('GCS')
# ax8.set_title('CPA [deg]')
# ax8.legend()
# ax8.grid(True)
# fig4.savefig(plot_dir + '/CPA_ANG2.png', dpi=300, bbox_inches='tight')
             
# plt.show()
# plt.close()



# #df.loc[df["SEEDS_WIDE_ANG"]>200]