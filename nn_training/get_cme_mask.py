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

def get_cme_mask(sample_image, inner_cme=True,occ_size=None):
    '''
    Returns a binary mask for the CME in the input (CME-only brigthness image)
    sample_image: array from rtraytracewcs
    inner_cme: Set to True to make the cme mask excludes the inner void of the gcs (if visible)
    '''    
    img_sz=np.shape(sample_image)[0]*2
    blur = cv2.GaussianBlur(sample_image,(3,3),0)
    norm_img =(blur-np.min(blur[blur != 0]))/(np.max(blur)-np.min(blur[blur != 0]))
    norm_img[norm_img<0]=0
    img = cv2.resize(norm_img,(img_sz,img_sz))
    cota = np.percentile(img[img != 0],5)
    _,thresh = cv2.threshold(img, cota, 255, cv2.THRESH_BINARY_INV)
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

        mask_aux = np.zeros((img_sz,img_sz), dtype=np.uint8)
        occulter_size = occ_size
        set_w = int(img_sz/2)
        set_h = int(img_sz/2)
        cv2.circle(mask_aux, (set_w,set_h), occulter_size, 1, -1)
        mask[mask_aux==1] = 1

        #Fill holes using contours
        #inverted_mask = cv2.bitwise_not(mask)
        #contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #filled_mask = cv2.drawContours(mask.copy(), contours, -1, 0, -1)#thickness=cv2.FILLED)

        #Fill holes using morphological operations
        #kernel = np.ones((10, 10), np.uint8)
        #filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=6)
        #filled_mask[mask_aux==1] = 0

        #Fill holes using multiple kernels morphological operations
        #kernel_sizes = [(1,1),(5, 5), (15, 15), (30, 30)]  # Adjust based on hole sizes
        #closed_mask = mask.copy()
        #for size in kernel_sizes:
        #    kernel = np.ones(size, np.uint8)
        #    filled_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_CLOSE, kernel)

        #Fill holes using floodfill
        # Invert the mask
        inverted_mask = mask.copy()
        # Flood fill from the edges (assume mask is surrounded by 0s at the edges)
        h, w = inverted_mask.shape[:2]
        flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)  # Create a flood fill mask with padding
        # Perform flood fill operation
        cv2.floodFill(inverted_mask, flood_fill_mask, (0, 0), 255)
        # Invert the flood filled image back
        inverted_filled_mask = cv2.bitwise_not(inverted_mask)
        # Combine the inverted filled mask with the original mask to fill the holes
        filled_mask = np.bitwise_or(mask, inverted_filled_mask)

        filled_mask[mask_aux==1] = 0
        mask = cv2.resize(filled_mask,(np.shape(sample_image)[0],np.shape(sample_image)[0]))
        mask[mask>0.01]=1
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
  
