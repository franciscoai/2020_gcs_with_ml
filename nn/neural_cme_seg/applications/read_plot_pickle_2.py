import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
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
            if param != 'MASK':
                x_global.append(all_data['x_points'+param])
                if param == 'APEX_ANGL_PER' or param == 'APEX_DIST_PER':
                    y_sub_global=[]
                    for p in range(len(all_data['y_points'+param])):
                        y_sub_global.append( np.median(all_data['y_points'+param][p]) )
                    y_global.append(y_sub_global)
                else:
                    y_global.append(all_data['y_points'+param])           
            if param == 'MASK':
                #return list of dataframes containing the masks
                x_global.append(all_data['df'])
                y_global.append(all_data['df'])
        #breakpoint()
    return x_global, y_global, labels

#define function to convert counter-clockwise to clockwise
#input: list of angle in degrees
#output: list of angle in degrees
def convert_to_clockwise(list_angles,negative_angle=False):
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
        if negative_angle and clockwise_angle >180:
            clockwise_angle = 360 - clockwise_angle
        clockwise_angles.append(clockwise_angle)
    return clockwise_angles


def plot_params(lista,param,opath,title,clockwise=False):
    #TODO: check for repeated masks for same date_time. seleck the one with better SCR value
    #check if param variable is a list
    if type(param) != list:
        x_global, y_global, labels = read_txt(lista,param)

    if type(param) == list:#True CPA calculation
        y_global =[]
        #in case of params = [aw_max,aw_min], this part calculates the "true CPA" -> CPA=(AW_MAX-AW_MIN)/2 + AW_MIN
        x_global0, y_global0, labels0 = read_txt(lista,param[0])
        x_global1, y_global1, labels1 = read_txt(lista,param[1])
        #sum each element of y_global0 and y_global1 and divide each element by 2
        for contador in range(len(x_global0)):
            #(a-b)/2 = distance, independet of axis, then is not necesarry to convert to counter-clockwise
            d = [(a - b)/2 + b for a, b in zip(y_global0[contador], y_global1[contador])]
            y_global.append(d)
        labels = labels0
        x_global = x_global0
        param = "True_CPA"
    list_legend=[]
    fig, ax = plt.subplots()
    ax.set_xlabel("Date and hour")
    if param == "AW" or param == "APEX_ANGL" or param == "True_CPA" or param =='APEX_ANGL_PER':
        ax.set_ylabel(param+" [deg]")
    if param == "AW_MIN" or param == "AW_MAX":
        #check if title parameter ends with "ccw"
        #when using ccw, max and min are inverted.
        if title.find("ccw") != -1:
            if param == "AW_MIN":
                ax.set_ylabel("AW_MAX"+" [deg]")
            if param == "AW_MAX":
                ax.set_ylabel("AW_MIN"+" [deg]")
    if param == "CPA":
        ax.set_ylabel("W"+param+" [deg]")
    if param == "APEX" or param=="APEX_DIST_PER":
        ax.set_ylabel(param+" [Rsun]")   
    if param == "AREA_SCORE":
        ax.set_ylabel(param+"%")
    
    colors=['k','g','r','b','y','m','c','w']
    instrument=['Cor2A','Cor2B','C2']
    #use heach element of the x_global and y_global list to plot x vs y
    for contador in range(len(x_global)):
        marker="*"
        linestyle=''
        if labels[contador].find('GCS') != -1:
            linestyle='--'
        if labels[contador].find('Img') != -1:
            linestyle='dotted'            
        if param == "AW" or param == "AW_MIN" or param == "AW_MAX" or param == "CPA" or param == "APEX_ANGL" or param =='APEX_ANGL_PER':
            y_global_units = np.degrees(y_global[contador])
            if clockwise == True:
                if param == "AW_MAX":
                    y_global_units = convert_to_clockwise(y_global_units,negative_angle=True)
                else:
                    y_global_units = convert_to_clockwise(y_global_units)

        if param == "True_CPA":
            y_global_units = np.degrees(y_global[contador])
            if clockwise == True:
                y_global_units = convert_to_clockwise(y_global_units)
        if param == "APEX" or param == "AREA_SCORE" or param=="APEX_DIST_PER":
            y_global_units = y_global[contador]
        line1, = ax.plot(x_global[contador], y_global_units, label=labels[contador], marker=marker,linestyle=linestyle,color=colors[contador])
        #create a list of legends
        position = (0.22,1.0-contador*0.07)
        list_legend.append(ax.legend(handles=[line1], bbox_to_anchor=position))

# Add legends to the plot
    for legend in list_legend:
        ax.add_artist(legend)  # Add legend back to the plot  
    # Add title using param
    ax.set_title(title)
    plt.grid()
    #save plot on opath directory
    ending = '_'+title+'_plot'
    fig.savefig(opath+'/'+str.lower(param)+ending+".png")
    
    return


#Define a function that read 2 pickle files and use both mask to calculate Intersection over Union, and return the value
def calculate_IOU(lista,opath,title=''):
    #read pickle files
    x_df, y_df, labels = read_txt(lista,'MASK')

    #en el caso de Img real, necesito remover mascaras que se hayan filtrado en el infer2. EN este caso la que tiene score<0.65
    aux = y_df[0]
    new_data2_df = aux.drop(aux[aux['SCR'] <0.65].index).reset_index(drop=True)
    y_df[0] = new_data2_df
    #mask and dates for real images
    date_img    = y_df[0]['DATE_TIME'].tolist()
    masks_img   = y_df[0]['MASK'].tolist()

    list_legend=[]
    colors=['k','g','r','b','y','m','c','w']
    fig, ax = plt.subplots()
    ax.set_xlabel("Date and hour")
    ax.set_ylabel("IoU Score")
    title='iou_score'+title

    for contador in range(1,len(y_df)):
        matching_index_list = []
        #filter y_df[contador] base on date_img timestamp, and keeping the mask with higher SCR value.
        for time in date_img:
            #breakpoint()
            matching_indices = y_df[contador]['DATE_TIME'].isin([time])
            count_true = (matching_indices == True).sum()
            if count_true == 0:
                #breakpoint()
                #search in time +- delta_time
                delta_time = 2
                matching_indices = y_df[contador]['DATE_TIME'].between(time-timedelta(minutes=delta_time),time+timedelta(minutes=delta_time))
                count_true2 = (matching_indices == True).sum()
                if count_true2 >1:
                    #keep the first True element
                    matching_indices = matching_indices.idxmax()

            if count_true > 1:
                #keep the one whose mask score is grater
                selected_df_aux = y_df[contador][matching_indices]
                selected_df_aux = selected_df_aux[selected_df_aux['SCR'] == selected_df_aux['SCR'].max()]
                matching_indices = selected_df_aux.index[0]
            if count_true == 1:
                matching_indices = matching_indices.idxmax()
            matching_index_list.append(matching_indices)
        breakpoint()
        #integer_positions = [idx[0] for idx in matching_index_list]
        #filter de y_df[contador] dataframe with the integer_positions list
        matching_indices = y_df[contador].index.isin(matching_index_list)
        #breakpoint()
        selected_df = y_df[contador][matching_indices]
        #matching_indices = y_df[contador]['DATE_TIME'].isin(y_df[0]['DATE_TIME'])
        #selected_df = y_df[contador][matching_indices]

        date_2  = selected_df['DATE_TIME'].tolist()
        masks_2 = selected_df['MASK'].tolist()
        #faltaria filtrar masks_2 con score>0.65
        if len(date_2) < len(date_img):
            print("WARNING: The number of masks in the second pickle file is less than the number of masks in the first pickle file")
            masks_img_aux = masks_img[:len(date_2)]
            date_img_aux = date_img[:len(date_2)]
        else:
            masks_img_aux = masks_img
            date_img_aux = date_img
        #calculate intersection over union for each mask
        iou_score = []
        breakpoint()
        for i in range(len(masks_img_aux)):
            intersection = np.logical_and(masks_img_aux[i], masks_2[i])
            union = np.logical_or(masks_img_aux[i], masks_2[i])
            iou_score.append(np.sum(intersection) / np.sum(union))
        
        marker="*"
        linestyle='--'
        if labels[contador].find('GCS') != -1:
            linestyle='--'
        if labels[contador].find('Img') != -1:
            linestyle='dotted'  
        #breakpoint()

        line1, = ax.plot(date_img_aux, iou_score, label=labels[contador], marker=marker,linestyle=linestyle,color=colors[contador])
        position = (0.22,1.0-contador*0.07)
        list_legend.append(ax.legend(handles=[line1], bbox_to_anchor=position,loc='upper right'))

    for legend in list_legend:
        ax.add_artist(legend)  
    # Add title using param
    ax.set_title(title)
    plt.grid()
    #save plot on opath directory
    ending = title+'_plot'
    fig.savefig(opath+'/'+ending+".png") 
    return iou_score


if __name__ == "__main__":
    full_plot = False
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
        plot_apex_angl_per = True
        plot_apex_dist_per = True

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
        plot_apex_angl_per = False
        plot_apex_dist_per = False

    #path and name of file including all the pickle files + labels
    dir_lista = '/gehme/projects/2023_eeggl_validation/output/2011-02-15/pickle/'
    lista = "2011_02_15_pickle.txt"
    #paht de salida
    opath = dir_lista
    instrument = "Cor2A"

    file1='/gehme/projects/2023_eeggl_validation/output/2011-02-15/gcs/cor2a/mask_props/parametros_filtered.pkl'
    file2='/gehme/projects/2023_eeggl_validation/output/2011-02-15/Cor2A/occ_medida_RD_infer2/model_v5/mask_props/parametros_filtered.pkl'
    calculate_IOU(dir_lista+lista,opath,title=instrument)

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

    if plot_apex_angl_per == True:
        plot_params(dir_lista+lista,"APEX_ANGL_PER",opath,title=instrument+"_ccw",clockwise=True)

    if plot_apex_dist_per == True:
        plot_params(dir_lista+lista,"APEX_DIST_PER",opath,title=instrument)
