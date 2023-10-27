import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy

def pnt2arr(points,imsize):
    '''
    Returns an array
    points:list of (x,y) points
    imsize: size of the output array
    '''
    arr=np.zeros(imsize)
    for i in range(len(points)):
        arr[points[i][0], points[i][1]] = 1
    return arr


def get_cme_mask(sample_image, inner_cme=True):
    '''
    Returns a binary mask for the CME in the input (CME-only brigthness image)
    sample_image: array from rtraytracewcs
    inner_cme: Set to True to make the cme mask excludes the inner void of the gcs (if visible)
    '''    
    img_sz=np.shape(sample_image)[0]*2
    norm_img =(sample_image-np.min(sample_image))/(np.max(sample_image)-np.min(sample_image))
    img = cv2.resize(norm_img,(img_sz,img_sz))
    _,thresh = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(cv2.Canny(thresh.astype("uint8"),0,255),None,cv2.BORDER_CONSTANT, borderValue=1)
    
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2], key=cv2.contourArea)
    
    cnt= [i for i in cnt if np.size(i)<(img_sz*8-50)]#deletes the image outer border

    if len(cnt) == 0:
        mask = np.zeros((np.shape(sample_image)[0],np.shape(sample_image)[0]))
    else:
        if len(cnt)>=3 and inner_cme:
            cme_outer = cnt[-2]
            cme_inner = cnt[-3]#[i for i in cnt[0:-2] if np.size(i)<(np.size(cme_outer)-100)] 
            mask_inner = np.zeros((img_sz,img_sz), np.uint8)
            mask_outer = np.zeros((img_sz,img_sz), np.uint8)
            mask_inner = cv2.drawContours(mask_inner, [cme_inner],-1, 1, -1)
            mask_outer = cv2.drawContours(mask_outer, [cme_outer],-1, 1, -1)
            mask = mask_outer - mask_inner
        else: # for when the inner border of the cme i not visible
            length =np.argsort([len(i) for i in cnt ])
            cme_outer = cnt[length[-1]]              
            mask_outer = np.zeros((img_sz,img_sz), np.uint8)
            mask_outer = cv2.drawContours(mask_outer, [cme_outer],-1, 1, -1)
            mask = mask_outer 
        mask = cv2.resize(mask,(np.shape(sample_image)[0],np.shape(sample_image)[0]))
        mask= np.array(mask)
    return(mask)


def get_mask_cloud(p_x,p_y,imsize, occ_size=None):
    '''
    Returns a mask from the cloud points
    p_x: x values for pixels
    p_y: y values for pixels
    imsize: size of the output array
    occ_size: radius of the occulter [px]. Pixels within the occuler are set to 0. Center is assumed at imsize/2
    OPATH: output path for the image
    '''
    if sum(p_x) == len(p_x) * p_x[0] or sum(p_y) == len(p_y) * p_y[0]:
        line = np.zeros(imsize)
        line[p_x, p_y] = 1
        return line

    points=[]
    for i in range(len(p_x)):
        points.append([p_x[i],p_y[i]])
    arr_cloud=pnt2arr(points,imsize)

    #creates a bounding box arround the cme cloud
    square = np.zeros_like(arr_cloud)
    square[np.min(p_x):np.max(p_x),np.min(p_y)] = 1 # Right edge
    square[np.min(p_x):np.max(p_x),np.max(p_y)] = 1 # Left edge
    square[np.min(p_x),np.min(p_y):np.max(p_y)] = 1 # Right edge
    square[np.max(p_x),np.min(p_y):np.max(p_y)] = 1 # Left edge
    result = arr_cloud+square

    #interpolation of the cme cloud points
    box=[[np.min(p_x),np.max(p_x)],[np.min(p_y),np.max(p_y)]]
    x_len=box[0][1]-box[0][0]
    y_len=box[1][1]-box[1][0]
    grid = np.indices((x_len,y_len))
    values = np.ones(len(p_x))
    xi = np.transpose(np.array([grid[0].flatten()+box[0][0], grid[1].flatten()+box[1][0]]))
    mask = scipy.interpolate.griddata(points, values, xi, method='linear',fill_value=0)
    arr_mask=np.zeros(imsize)

    if occ_size is None:
        for i in range(len(xi)):
            arr_mask[int(xi[i][0]), int(xi[i][1])] = mask[i]
    else:        
        for i in range(len(xi)):
            px_dist_to_center = np.sqrt((xi[i][0]-imsize[0]/2)**2 + (xi[i][1]-imsize[1]/2)**2)
            if px_dist_to_center >= occ_size:
                arr_mask[int(xi[i][0]), int(xi[i][1])] = mask[i]
    
    arr_mask[arr_mask>0]=1
    
    if np.sum(arr_mask) == 0:
        arr_mask = np.zeros(imsize)

    return arr_mask
  
