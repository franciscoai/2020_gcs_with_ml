import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_cme_mask(sample_image, inner_cme=True):
    '''
    Returns a binary mask for the CME in the input (CME-only brigthness image)
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