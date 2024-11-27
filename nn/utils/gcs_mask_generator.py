import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from nn_training.get_cme_mask import get_mask_cloud
from nn.utils.coord_transformation import deg2px
from pyGCS_raytrace import pyGCS


def maskFromCloud(params, satpos, imsize, plotranges, sat=0):
    clouds = pyGCS.getGCS(params[0], params[1], params[2], params[3], params[4], params[5], satpos)               
    x = clouds[sat, :, 1]
    y = clouds[0, :, 2]
    p_x,p_y=deg2px(x,y,plotranges,imsize,sat)
    mask=get_mask_cloud(p_x,p_y,imsize)
    if len(p_x)==0 or len(p_y)==0:
        breakpoint()
    mask = np.flip(mask, axis=0)
    return mask

def maskFromCloud_3d(params, satpos, imsize, plotranges, occ_size=None):
    '''
    Same as maskFromCloud but returns as many mask as satpos provided
    :params: CMElon, CMElat, CMEtilt, height, k, ang, satpos
    '''
    clouds = pyGCS.getGCS(params[0], params[1], params[2], params[3], params[4], params[5], satpos)        
    all_mask = []
    for sat in range(len(satpos)):       
        x = clouds[sat, :, 1]
        y = clouds[0, :, 2]
        p_x,p_y=deg2px(x,y,plotranges, imsize, sat)
        #breakpoint()
        #if len(p_x)==0 or len(p_y)==0:
        #    breakpoint()
        #Si p_x o p_y son vacios, entonces devolver mascara vacia??
        #podria pasar que en la exploracion de valores de la minimizacion, la proyeccion caiga en el oculter??
        if occ_size is None:
            mask=get_mask_cloud(p_x,p_y,imsize)
        else:
            mask=get_mask_cloud(p_x,p_y,imsize, occ_size=occ_size[sat])
        all_mask.append(mask)
    return all_mask