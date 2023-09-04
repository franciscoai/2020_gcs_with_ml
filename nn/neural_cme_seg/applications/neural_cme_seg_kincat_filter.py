import pandas as pd
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import least_squares

def plot_mask_prop(x,param,param_name, opath, ending='_filtered',x_title='Date and hour',y_title=None, style='or', save=True, overplot=None):
        if y_title is None:
            y_title = param_name
        if param_name.endswith("ANG"):        
            b = [np.degrees(i) for i in param]
        else:
            b = param
        if overplot is None:
            fig2, ax2 = plt.subplots()
        else:
            fig2, ax2 = overplot
        ax2.plot(x, b, style, label=y_title)
        ax2.set_xlabel(x_title)
        ax2.set_title('Filtered '+ y_title)
        ax2.legend()
        ax2.grid('both')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save:
            fig2.savefig(opath+'/'+str.lower(y_title)+ending+".png") 
            plt.close()
        else:
            return fig2, ax2

def graphics(df, parameters, opath, fit_err_func, fit_func, in_cond):
    #max,cpa and min angles graphic
   
    x = []  
    max_ang_y = []  
    min_ang_y = []
    cpa_ang_y = []  

    for idx, row in df.iterrows():
        date_time = row['DATE_TIME']
        x.append(date_time)

    # #mass center coordinates graphic
    # cm_y_list = []
    # cm_x_list = []  
    # for idx, row in df.iterrows():
    #     cm_x = row['MASS_CENTER_X']
    #     cm_y = row['MASS_CENTER_Y']
    #     cm_x_list.append(cm_x)
    #     cm_y_list.append(cm_y)
    # fig1, ax1 = plt.subplots()
    # ax1.scatter(cm_x_list, cm_y_list, color='red', label='max_ang')
    # ax1.set_xlabel('x coordinates')
    # ax1.set_ylabel('y coordinates')
    # ax1.set_title('Filtered dispersion graphic of mass center coordinates')
    # ax1.legend()
    # ax1.grid('both')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # fig1.savefig(data_dir+"/"+ext_folder+"/stats/"+"mass_center_coordinates_filtered.png") 
    # plt.close()

    #generates graphics for the parameters
    idx=0
    for i in parameters:     
        #fits func
        x_points=np.array([r.timestamp() for r in x]) #df['DATE_TIME'].apply(lambda x: x.timestamp())
        y_points=df[i]       
        weights = np.ones(len(x_points))
        fit=least_squares(fit_err_func[idx], in_cond[idx] , method='lm', kwargs={'x': x_points-x_points[0], 'y': y_points, 'w': weights}) # first fit to get a good initial condition
        fit=least_squares(fit_err_func[idx], fit.x, loss='soft_l1', kwargs={'x': x_points-x_points[0], 'y': y_points, 'w': weights}) # second fit to ingnore outliers    
        x_line = np.linspace(x_points.min(), x_points.max(), 100)
        y_line = fit_func[idx](x_line-x_points[0], *fit.x)
        x_line = pd.to_datetime(x_line, unit='s') 
        idx+=1
        # plots
        fig = plot_mask_prop(x,df[i],i, '', save=False)
        plot_mask_prop(x_line,y_line,i, opath, ending='_filtered', style='--', save=True, overplot=fig)        
   

def quadratic(t,a,b,c):
    return a*t**2. + b*t + c

def quadratic_error(p, x, y, w):
    return w*(quadratic(x, *p) - y)

def linear(t,a,b):
    return a*t + b

def linear_error(p, x, y, w):
    return w*(linear(x, *p) - y)

def filter_param(x,y, error_func, fit_func, in_cond, criterion, percentual=True):
    '''
    Deletes the points that are more than criterion from the fit function given by fit_func
    '''
    # fit the function using least_squares
    weights = np.ones(len(x))
    weights[-4:] = 10 # gives more weight to the last 3 points   
    fit=least_squares(error_func, in_cond , method='lm', kwargs={'x': x-x[0], 'y': y, 'w': weights}) # first fit to get a good initial condition
    fit=least_squares(error_func, fit.x, loss='soft_l1', kwargs={'x': x-x[0], 'y': y, 'w': weights}) # second fit to ingnore outliers    

    #calculate the % distance from the fit
    if percentual:
        dist = np.abs(fit_func(x-x[0], *fit.x)-y)/y
    else:
        dist = np.abs(fit_func(x-x[0], *fit.x)-y)
    #get the index of the wrong points
    ok_ind = np.where(dist<criterion)[0]
    return ok_ind

def filter_mask(df):
    '''Iterates over the df and filter the points using various kinetic and morphologic motivated criteria'''
    cpa_criterion = np.radians(15.) # criterion for the cpa angle [deg]
    apex_rad_crit = 0.2 # criterion for the apex radius [%]
    date_crit = 5. # criterion for the date, max [hours] for an acceptable gap. if larger it keeps later group only
    aw_crit = np.radians(8.) # criterion for the angular width [deg]

    # filters DATES too far from the median
    x_points=np.array([i.timestamp() for i in df['DATE_TIME']])
    date_diff = x_points[1:]-x_points[:-1]
    date_diff = np.abs(date_diff-np.median(date_diff))
    gap_idx = np.where(date_diff>date_crit*3600.)[0]
    # keeps only the later group
    if len(gap_idx)>0:
        gap_idx = gap_idx[0]
        df = df.iloc[gap_idx+1:]

    # filters on 'CPA_ANG'
    x_points=np.array([i.timestamp() for i in df['DATE_TIME']])
    y_points=np.array(df['CPA_ANG'])
    ok_idx = filter_param(x_points, y_points, linear_error, linear, [1.,1.], cpa_criterion, percentual=False)
    df = df.iloc[ok_idx]

    # filters on 'APEX_RADIUS'
    x_points=np.array([i.timestamp() for i in df['DATE_TIME']])
    y_points=np.array(df['APEX_RADIUS'])
    ok_idx = filter_param(x_points,y_points, quadratic_error, quadratic,[0., 1.,1.], apex_rad_crit, percentual=True)
    df = df.iloc[ok_idx]
    
    # filters on 'WIDTH_ANG'
    x_points=np.array([i.timestamp() for i in df['DATE_TIME']])
    y_points=np.array(df['WIDE_ANG'])
    ok_idx = filter_param(x_points,y_points, linear_error, linear,[1.,1.], aw_crit, percentual=False)
    df = df.iloc[ok_idx]

    return df

# def _filter_param( in_x, in_y, error_func, fit_func, in_cond, criterion, percentual=True, weights=[2]):
#         '''
#         Deletes the points that are more than criterion from the fit function given by fit_func
#         in_x: timestamps of the points to be filtered
#         in_y: y values of the points to be filtered
#         error_func: function to calculate the error between the fit and the data
#         fit_func: function to fit the data
#         in_cond: initial condition for the fit
#         criterion: maximum distance from the fit to consider a point as ok
#         percentual: if True, the criterion is a percentage of the y value
#         weights: data points weights for the fit. Must be a cevtor of len len(in_x) or int. 
#                  If It's a scalar int gives more weight to the last weights dates bcause CME is supoused to be larger and better defined
#         '''
#         #deletes points with y is nan
#         ok_ind = np.where(~np.isnan(in_y))[0]
#         x = in_x[ok_ind]
#         y = in_y[ok_ind]

#         # fit the function using least_squares
#         used_weights = np.ones(len(x))
#         #breakpoint()
#         if len(weights) > 1:
#             used_weights = weights
#         fit=least_squares(error_func, in_cond , method='lm', kwargs={'x': x-x[0], 'y': y, 'w': used_weights}) # first fit to get a good initial condition
#         fit=least_squares(error_func, fit.x, loss='soft_l1', kwargs={'x': x-x[0], 'y': y, 'w': used_weights}) # second fit to ingnore outliers    

#         #calculate the % distance from the fit
#         if percentual:
#             dist = np.abs(fit_func(x-x[0], *fit.x)-y)/y
#         else:
#             dist = np.abs(fit_func(x-x[0], *fit.x)-y)

#         #get the index of the wrong points
#         ok_ind2 = np.where(dist<criterion)[0]

#         #get the index of the ok points
#         ok_ind = ok_ind[ok_ind2]

#         return ok_ind


# def _filter_masks(dates, masks, scores, labels, boxes, mask_prop):
#         '''
#         Filters the masks based on the cpa, aw and apex radius evolution consistency
#         mask_prop: list of lists with the mask properties for each mask. Each maks's list has the following format: [id,float(scores[i]),cpa_ang, wide_ang, apex_dist]
#         '''
#         cpa_criterion = np.radians(20.) # criterion for the cpa angle [deg]
#         apex_rad_crit = 0.2 # criterion for the apex radius [%]
#         date_crit = 5. # criterion for the date, max [hours] for an acceptable gap. if larger it keeps later group only
#         aw_crit = np.radians(20.) # criterion for the angular width [deg]
        
#         # keeps only a max of two masks per image with the highest score
#         for i in range(len(dates)):
#             ok_idx = []
#             if len(masks[i]) > 2:
#                 score_idx = np.argsort(scores[i])
#                 ok_idx.append(score_idx[-1])
#                 ok_idx.append(score_idx[-2])
#                 ok_idx = np.array(ok_idx)
#                 masks[i] = [masks[i][ok_idx[j]] for j in range(len(ok_idx))]
#                 scores[i] = [scores[i][ok_idx[j]] for j in range(len(ok_idx))]
#                 labels[i] = [labels[i][ok_idx[j]] for j in range(len(ok_idx))]
#                 boxes[i] = [boxes[i][ok_idx[j]] for j in range(len(ok_idx))]
#                 mask_prop[i] = [mask_prop[i][ok_idx[j]] for j in range(len(ok_idx))]
                        
            
#         # filters DATES too far from the median
#         x_points = np.array([i.timestamp() for i in dates])
#         date_diff = x_points[1:]-x_points[:-1]
#         date_diff = np.abs(date_diff-np.median(date_diff))
#         gap_idx = np.where(date_diff>date_crit*3600.)[0]
        
#         # keeps only the later group
#         if len(gap_idx)>0:
#             gap_idx = gap_idx[0]
#             ok_dates = dates[gap_idx+1:]
#             ok_masks = masks[gap_idx+1:]
#             ok_scores = scores[gap_idx+1:]
#             ok_labels = labels[gap_idx+1:]
#             ok_boxes = boxes[gap_idx+1:]
#             ok_mask_prop = mask_prop[gap_idx+1:]
#         else:
#             ok_dates = dates
#             ok_masks = masks
#             ok_scores = scores
#             ok_labels = labels
#             ok_boxes = boxes
#             ok_mask_prop = mask_prop

#         # flatten all lists
#         all_masks = np.array([i for j in ok_masks for i in j])
#         all_scores = np.array([i for j in ok_scores for i in j])
#         all_lbl = np.array([i for j in ok_labels for i in j])
#         all_boxes = np.array([i for j in ok_boxes for i in j])
#         all_mask_prop = np.array([i for j in ok_mask_prop for i in j])
#         all_dates = []
#         for i in range(len(ok_masks)):
#             all_dates.append([ok_dates[i]]*len(ok_masks[i]))
#         all_dates = np.array([i for j in all_dates for i in j])

#         # filters on 'CPA_ANG'
#         x_points =np.array([i.timestamp() for i in all_dates])
#         y_points =np.array(all_mask_prop[:,2]).astype('float32')
#         ok_idx = _filter_param(x_points, y_points, linear_error, linear, [1.,1.], cpa_criterion, percentual=False)
#         all_masks = all_masks[ok_idx]
#         all_scores = all_scores[ok_idx]
#         all_lbl = all_lbl[ok_idx]
#         all_boxes = all_boxes[ok_idx]
#         all_mask_prop = all_mask_prop[ok_idx]
#         all_dates = all_dates[ok_idx]

#         # # filters on 'APEX_RADIUS'
#         # x_points =np.array([i.timestamp() for i in all_dates])
#         # y_points =np.array(all_mask_prop[:,4]).astype('float32')
#         # # use the mask score as fit weight
#         # weights = np.array(all_mask_prop[:,1]).astype('float32')
#         # ok_idx = self._filter_param(x_points,y_points, quadratic_error, quadratic,[0., 1.,1.], apex_rad_crit, weights=weights, percentual=True)
#         # all_masks = all_masks[ok_idx]
#         # all_scores = all_scores[ok_idx]
#         # all_lbl = all_lbl[ok_idx]
#         # all_boxes = all_boxes[ok_idx]
#         # all_mask_prop = all_mask_prop[ok_idx]
#         # all_dates = all_dates[ok_idx]
        
#         # # filters on 'WIDTH_ANG'
#         # x_points=np.array([i.timestamp() for i in all_dates])
#         # y_points =np.array(all_mask_prop[:,3]).astype('float32')
#         # ok_idx = self._filter_param(x_points,y_points, linear_error, linear,[1.,1.], aw_crit, percentual=False)
#         # all_masks = all_masks[ok_idx]
#         # all_scores = all_scores[ok_idx]
#         # all_lbl = all_lbl[ok_idx]
#         # all_boxes = all_boxes[ok_idx]
#         # all_mask_prop = all_mask_prop[ok_idx]
#         # all_dates = all_dates[ok_idx]

#         #if multiple maks remain for a single date, keeps the one with highest score
#         ok_idx = []
#         for i in np.unique(all_dates):
#             date_idx = np.where(all_dates == i)[0]
#             if len(date_idx) > 1:
#                 score_idx = np.argmax(all_scores[date_idx])
#                 ok_idx.append(date_idx[score_idx])
#             else:
#                 ok_idx.append(date_idx[0])                  
#         all_masks = all_masks[ok_idx]
#         all_scores = all_scores[ok_idx]
#         all_lbl = all_lbl[ok_idx]
#         all_boxes = all_boxes[ok_idx]
#         all_mask_prop = all_mask_prop[ok_idx]
#         all_dates = all_dates[ok_idx]
        
#         #converts the first dim of all ouput np arrays to list of arrays
#         all_masks = [all_masks[i,:,:] for i in range(len(all_dates))]
#         all_scores = [all_scores[i] for i in range(len(all_dates))]
#         all_lbl = [all_lbl[i] for i in range(len(all_dates))]
#         all_boxes = [all_boxes[i,:] for i in range(len(all_dates))]
#         all_mask_prop = [all_mask_prop[i,:] for i in range(len(all_dates))]
#         all_dates = [all_dates[i] for i in range(len(all_dates))]
                   
#         return all_dates,all_masks, all_scores, all_lbl, all_boxes,all_mask_prop



######################################################### MAIN ###################################################################
data_dir='/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat_L1/cor2_a'
parameters_to_plot =  ['WIDE_ANG','MASS_CENTER_RADIUS','MASS_CENTER_ANG','APEX_RADIUS','APEX_ANG','CPA_ANG','MIN_ANG','MAX_ANG']
fit_err_func = [linear_error, quadratic_error, linear_error, quadratic_error, linear_error, linear_error, linear_error, linear_error]
fit_func = [linear, quadratic, linear, quadratic, linear, linear, linear, linear]
in_cond= [[1.,1.],[1.,1.,1.],[1.,1.],[1.,1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.]]

ext_folders = os.listdir(data_dir)
for ext_folder in ext_folders:
    try:
        print('Processing '+ext_folder)
        opath = data_dir+"/"+ext_folder+"/filtered"
        csv_path=data_dir+"/"+ext_folder+"/stats/"+ext_folder+"_stats"
        df=pd.read_csv(csv_path)
        df["DATE_TIME"]= pd.to_datetime(df["DATE_TIME"])
        df = df.sort_values(by='DATE_TIME')
        df_filtered=filter_mask(df)
        #df_filtered=_filter_masks(df)
        

        # saves the filtered df
        os.makedirs(opath, exist_ok=True)
        df_filtered.to_csv(opath+'/'+str(ext_folder)+'.csv', index=False)
        graphics(df_filtered, parameters_to_plot, opath, fit_err_func, fit_func, in_cond)
    except:
        print('Error processing '+ext_folder)


    