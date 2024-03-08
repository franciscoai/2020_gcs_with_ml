import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
"""
To DO: 
-generate a function to read the pickle files and return the dictionary with the data
-generate a function to plot the data
"""


#Asuming a txt file with a list of pickle files + path (containing all the parameters calculated from the mask, product of compute_mask_prop)
# and the corresponding label (runXXX, GCS, Img)
#
#
#
plot_apex = True
plot_AW   = True
plot_cpa  = True

#path and name of file including all the pickle files + labels
dir_list = '/gehme/projects/2023_eeggl_validation/output/2011-02-15/pickle/'
list = "2011_02_15_pickle.txt"
#paht de salida
opath = dir_list
instrument = "Cor2A"

def read_pickle(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data

def read_txt(list_file,param):
    #param (APEX, CPA, MASK_ID, SCR, WA, CME_ID)
    x_global = []
    y_global = []
    labels = []

    with open(list_file, 'r') as file:
        data = file.readlines()
        for i in range(len(data)):
            parts = data[i].replace('\n','').split(',')
            labels.append(parts[1])
            all_data = read_pickle(parts[0])
            x_global.append(all_data['x_points'+param])
            y_global.append(all_data['y_points'+param])
            
    return x_global, y_global, labels


def plot_params(lista,param,opath,title):
    x_global, y_global, labels = read_txt(lista,param)
    list_legend=[]
    fig, ax = plt.subplots()
    ax.set_xlabel("Date and hour")
    if param == "WA":
        ax.set_ylabel(param+" [deg]")
    if param == "CPA":
        ax.set_ylabel(param+" [deg]")
    if param == "APEX":
        ax.set_ylabel(param+" [km]")   
    colors=['r','b','g','k','y','m','c','w']
    instrument=['Cor2A','Cor2B','C2']
    #use heach element of the x_global and y_global list to plot x vs y
    for contador in range(len(x_global)):
        #ax.plot(x_global[contador], y_global[contador], style,color=colors[contador])
        marker="*"
        linestyle=''
        if labels[contador].find('GCS') != -1:
            linestyle='--'
        if param == "WA":
            y_global_units = np.degrees(y_global[contador])
        if param == "CPA":
            y_global_units = np.degrees(y_global[contador])
        if param == "APEX":
            y_global_units = y_global[contador]
        line1, = ax.plot(x_global[contador], y_global_units, label=labels[contador], marker=marker,linestyle=linestyle,color=colors[contador])
        #create a list of legends
        position = (0.22,1.0-contador*0.07)
        list_legend.append(ax.legend(handles=[line1], bbox_to_anchor=position))

# Add legends to the plot
    for legend in list_legend:
        ax.add_artist(legend)  # Add legend back to the plot  
    #breakpoint()
    # Add title using param
    ax.set_title(title)
    plt.grid()
    #save plot on opath directory
    ending = '_test2_plot'
    fig.savefig(opath+'/'+str.lower(param)+ending+".png")
    
    return

if plot_apex == True:
    plot_params(dir_list+list,"APEX",opath,title=instrument)

#breakpoint()
if plot_AW == True:
    plot_params(dir_list+list,"WA",opath,title=instrument)

if plot_cpa == True:
    plot_params(dir_list+list,"CPA",opath,title=instrument)