import matplotlib.pyplot as plt
import numpy as np
import cv2

def segmentation(sample_image):
    img_sz=np.shape(sample_image)[0]*2
    norm_img =(sample_image-np.min(sample_image))/(np.max(sample_image)-np.min(sample_image))
    img = cv2.resize(norm_img,(img_sz,img_sz))
    _,thresh = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(cv2.Canny(thresh.astype("uint8"),0,255),None,cv2.BORDER_CONSTANT, borderValue=1)
    
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2], key=cv2.contourArea)
    [print(np.size(i)) for i in cnt]
    cnt= [i for i in cnt if np.size(i)<(img_sz*8-50)]#deleates the image outer border
    cme_outer = cnt[-2]
    cme_inner = cnt[-3]#[i for i in cnt[0:-2] if np.size(i)<(np.size(cme_outer)-100)] 
    print(np.size(cme_outer))
    print(np.size(cme_inner))

    mask_inner = np.zeros((img_sz,img_sz), np.uint8)
    mask_outer = np.zeros((img_sz,img_sz), np.uint8)
    mask_inner = cv2.drawContours(mask_inner, [cme_inner],-1, 1, -1)
    mask_outer = cv2.drawContours(mask_outer, [cme_outer],-1, 1, -1)
    mask = mask_outer - mask_inner
    mask = cv2.resize(mask,(np.shape(sample_image)[0],np.shape(sample_image)[0]))
    mask= np.array(mask)
#     m = np.nanmean(mask)
#     sd = np.nanstd(mask)
#     fig2 = plt.figure(figsize=(4,4), facecolor='black')
#     ax2 = fig2.add_subplot()      
#     ax2.imshow(mask, origin='lower', cmap='gray', vmax=m+3*sd, vmin=m-3*sd)
#     fig2.savefig(opath+"/mask.jpg")

#     fig3 = plt.figure(figsize=(4,4), facecolor='black')
#     ax3 = fig3.add_subplot()      
#     ax3.imshow(edges, origin='lower', cmap='gray')#, vmax=m+3*sd, vmin=m-3*sd)
#     fig3.savefig(opath+"/edges.jpg")}

    return(mask)