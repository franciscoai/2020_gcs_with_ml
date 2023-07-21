import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from nn_training.get_cme_mask import get_mask_cloud
from pyGCS_raytrace import pyGCS


def maskFromCloud(params, sat, satpos, imsize, plotranges):
    clouds = pyGCS.getGCS(params[0], params[1], params[2], params[3], params[4], params[5], satpos)               
    x = clouds[sat, :, 1]
    y = clouds[0, :, 2]
    p_x,p_y=deg2px(x,y,plotranges,imsize,sat)
    mask=get_mask_cloud(p_x,p_y,imsize)
    return mask

def deg2px(x,y,plotranges,imsize,sat):
    '''
    Computes spatial plate scale in both dimensions
    '''    
    scale_x = (plotranges[sat][1]-plotranges[sat][0])/imsize[0]
    scale_y =(plotranges[sat][3]-plotranges[sat][2])/imsize[1]
    x_px=[]
    y_px=[]    
    for i in range(len(x)):
        v_x= (np.round((x[i]-plotranges[sat][0])/scale_x)).astype("int") 
        v_y= (np.round((y[i]-plotranges[sat][2])/scale_y)).astype("int")
        if v_x<512 and v_y<512:
            x_px.append(v_x)
            y_px.append(v_y)
    return(y_px,x_px)