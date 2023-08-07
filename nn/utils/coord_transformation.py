import numpy as np


def center_rSun_pixel(headers, plotranges, sat):
    '''
    Gets the location of Suncenter in pixels
    '''    
    x_cS = headers[sat]['CRPIX1'] #(headers[sat]['CRPIX1']*plotranges[sat][sat]*2) / headers[sat]['NAXIS1'] - plotranges[sat][sat]
    y_cS = headers[sat]['CRPIX2'] #(headers[sat]['CRPIX2']*plotranges[sat][sat]*2) / headers[sat]['NAXIS2'] - plotranges[sat][sat]
    return x_cS, y_cS

def deg2px(x,y,plotranges,imsize,sat):
    '''
    Computes spatial plate scale in both dimensions
    '''
    #dischard points outside the plotranges
    mask = (x > plotranges[sat][0]) & (x < plotranges[sat][1]) & (y > plotranges[sat][2]) & (y < plotranges[sat][3])
    x_ok = x[mask]
    y_ok = y[mask]
    scale_x = (plotranges[sat][1]-plotranges[sat][0])/imsize[0]
    scale_y =(plotranges[sat][3]-plotranges[sat][2])/imsize[1]
    x_px=[]
    y_px=[]    
    for i in range(len(x_ok)):
        v_x= (np.round((x_ok[i]-plotranges[sat][0])/scale_x)).astype("int") 
        v_y= (np.round((y_ok[i]-plotranges[sat][2])/scale_y)).astype("int")
        if np.abs(v_x)<imsize[0] and np.abs(v_y)<imsize[1]:
            x_px.append(v_x)
            y_px.append(v_y)
    return(y_px,x_px)

def pnt2arr(x,y,plotranges,imsize,sat):
    '''
    Returns an array
    points:list of (x,y) points
    imsize: size of the output array
    '''
    p_x,p_y=deg2px(x,y,plotranges,imsize, sat)
    points=[]
    for i in range(len(p_x)):
        points.append([p_x[i],p_y[i]])       
    arr=np.zeros(imsize)
    for i in range(len(points)):
        arr[points[i][0], points[i][1]] = 1
    return arr