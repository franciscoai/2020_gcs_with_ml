import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
"""
TODO: 
- En lugar de generar plots individuales, generar un panel con los tres plots para todos los instrumentos. por ahora seria 3x3.
"""


#Asuming a txt file with a list of pickle files + path (containing all the parameters calculated from the mask, product of compute_mask_prop)
# and the corresponding label (runXXX, GCS, Img)
#
#
#
full_plot = True
if full_plot == True:
    plot_apex       = True
    plot_AW         = True
    plot_wcpa       = True
    plot_true_cpa   = True
    plot_apex_angle = True
    plot_score_ares = True
    plot_area_score = True
    plot_AW_MIN     = True
    plot_AW_MAX     = True

if full_plot == False:
    plot_apex       = False
    plot_AW         = False
    plot_wcpa       = False
    plot_true_cpa   = False
    plot_apex_angle = False
    plot_score_ares = False
    plot_area_score = False
    plot_AW_MIN     = False
    plot_AW_MAX     = False
plot_true_cpa = True

#path and name of file including all the pickle files + labels
dir_lista = '/gehme/projects/2023_eeggl_validation/output/2011-02-15/pickle/'
lista = "2011_02_15_pickle.txt"
#paht de salida
opath = dir_lista
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
            #breakpoint()
            x_global.append(all_data['x_points'+param])
            y_global.append(all_data['y_points'+param])            
    return x_global, y_global, labels

#define function to convert counter-clockwise to clockwise
#input: list of angle in degrees
#output: list of angle in degrees
def convert_to_clockwise(list_angles):
    """Converts a list of angles in degrees measured clockwise from the positive x-axis to counter-clockwise from the positive y-axis.
    Args:
        list_angles: A list of angles in degrees.
    Returns:
        A new list of angles in degrees measured clockwise.
    """
    max_angle = 360
    clockwise_angles = []
    
    #Shift the angles 90 deg because compute_mask_prop calculates the angle from the positive x-axis clockwise
    shift = 90
    list_angles = [a + shift for a in list_angles]
    for angle in list_angles:
        clockwise_angle = (max_angle - angle) % max_angle
        clockwise_angles.append(clockwise_angle)
    return clockwise_angles


def plot_params(lista,param,opath,title,clockwise=False):
    #check if param variable is a list
    if type(param) != list:
        x_global, y_global, labels = read_txt(lista,param)

    if type(param) == list:#True CPA calculation
        y_global =[]
        #in case of params = [aw_min, aw_max], this part calculates the "true CPA" -> CPA=(AW_MIN+AW_MAX)/2 + AW_MIN
        x_global0, y_global0, labels0 = read_txt(lista,param[0])
        x_global1, y_global1, labels1 = read_txt(lista,param[1])
        #sum each element of y_global0 and y_global1 and divide each element by 2
        for contador in range(len(x_global0)):
            #(a+b)/2 = distance, independet of axis, then
            d = [(a - b)/2 + b for a, b in zip(y_global0[contador], y_global1[contador])]
            #a_ccw = convert_to_clockwise(y_global0[contador])
            #onvertir a en ccw
            #aux = [a + (a + b)/2 for a, b in zip(y_global0[contador], y_global1[contador])]
            #aux = [a + b for a, b in zip(y_global1[contador], d)]
            y_global.append(d)
        labels = labels0
        x_global = x_global0
        param = "True_CPA"
    list_legend=[]
    fig, ax = plt.subplots()
    ax.set_xlabel("Date and hour")
    if param == "AW" or param == "AW_MIN" or param == "AW_MAX" or param == "APEX_ANGL" or param == "True_CPA":
        ax.set_ylabel(param+" [deg]")
    #if param == "AW_MIN":
    #    ax.set_ylabel(param+" [deg]")
    #if param == "AW_MAX":
    #    ax.set_ylabel(param+" [deg]")
    if param == "CPA":
        ax.set_ylabel("W"+param+" [deg]")
    if param == "APEX":
        ax.set_ylabel(param+" [Rsun]")   
    #if param == "APEX_ANGL":
    #    ax.set_ylabel(param+" [deg]")
    if param == "AREA_SCORE":
        ax.set_ylabel(param+"%")
    #if param == "True_CPA":
    #    ax.set_ylabel(param+ "[deg]")
    
    colors=['r','b','g','k','y','m','c','w']
    instrument=['Cor2A','Cor2B','C2']
    #use heach element of the x_global and y_global list to plot x vs y
    for contador in range(len(x_global)):
        marker="*"
        linestyle=''
        if labels[contador].find('GCS') != -1:
            linestyle='--'
        if labels[contador].find('Img') != -1:
            linestyle='dotted'            
        if param == "AW" or param == "AW_MIN" or param == "AW_MAX" or param == "CPA" or param == "APEX_ANGL":
            y_global_units = np.degrees(y_global[contador])
            if clockwise == True:
                y_global_units = convert_to_clockwise(y_global_units)
        if param == "True_CPA":
            y_global_units = np.degrees(y_global[contador])
            y_global_units = convert_to_clockwise(y_global_units)
        #if param == "AW_MIN":
        #    y_global_units = np.degrees(y_global[contador])
        #if param == "AW_MAX":
        #    y_global_units = np.degrees(y_global[contador])
        #if param == "CPA":
        #    y_global_units = np.degrees(y_global[contador])
        #    if clockwise == True: 
                #breakpoint()
        #        y_global_units = convert_to_clockwise(y_global_units)
        if param == "APEX" or param == "AREA_SCORE":
            y_global_units = y_global[contador]
        #if param == "APEX_ANGL":
        #    y_global_units = np.degrees(y_global[contador])
        #if param == "AREA_SCORE":
        #    y_global_units = y_global[contador]
        #if param == "True_CPA":
        #    y_global_units = np.degrees(y_global[contador])

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
    ending = '_'+title+'_plot'
    fig.savefig(opath+'/'+str.lower(param)+ending+".png")
    
    return

if plot_apex == True:
    plot_params(dir_lista+lista,"APEX",opath,title=instrument)

if plot_area_score == True:
    plot_params(dir_lista+lista,"AREA_SCORE",opath,title=instrument)

if plot_AW == True:
    plot_params(dir_lista+lista,"AW",opath,title=instrument)

if plot_wcpa == True:
    plot_params(dir_lista+lista,"CPA",opath,title=instrument+"_ccw",clockwise=True)

if plot_apex_angle == True:
    plot_params(dir_lista+lista,"APEX_ANGL",opath,title=instrument+"_ccw",clockwise=True)

if plot_true_cpa == True:
    plot_params(dir_lista+lista,["AW_MAX","AW_MIN"],opath,title=instrument+"_ccw",clockwise=True)

if plot_AW_MIN == True:
    plot_params(dir_lista+lista,"AW_MIN",opath,title=instrument+"_ccw",clockwise=True)

if plot_AW_MAX == True:
    plot_params(dir_lista+lista,"AW_MAX",opath,title=instrument+"_ccw",clockwise=True)

