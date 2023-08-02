def read_fits(file_path,smooth_kernel=[0,0]):
    imageSize=[512,512]
    
    try:       
        
        file=glob.glob((file_path)[0:-5]+"*")
        img = fits.open(file[0])
        img=(img[0].data).astype("float32")
        #img = gaussian_filter(img, sigma=smooth_kernel)

        # if smooth_kernel[0]!=0 and smooth_kernel[1]!=0: 
        img = rebin(img, imageSize,operation='mean')
        
        return img  
    except:
        print(f'WARNING. I could not find file {file_path}')
        return None
    
def rec2pol(mask,scores,labels,boxes,imsize):
    
    mask_threshold = 0.6 # value to consider a pixel belongs to the object
    scr_threshold = 0.6 # only detections with score larger than this value are considered
    nans = np.full(imsize, np.nan)
    try:
        if boxes is not None:
            nb = 0
            for b in boxes:
                if scores is not None:
                    scr = scores[nb]
                else:
                    scr = 0   
                if scr > scr_threshold:             
                    masked = nans.copy()
                    masked[:, :][mask[nb] > mask_threshold] = nb
                nb+=1

                height, width = masked.shape
                # Calcula el centro en cada dimensión
                center_x = width // 2
                center_y = height // 2
                pol_mask=[]
                center=[center_x,center_y]
                for x in range(width):
                    for y in range(height):
                        value=masked[x,y]
                        if not np.isnan(value):
                            x_dist = (center_x-x)
                            y_dist = (center_y-y)
                            distance= math.sqrt(x_dist**2 + y_dist**2)
                            angle=math.atan(x_dist / y_dist)
                            angle_deg = math.degrees(angle)
                            pol_mask.append([distance,angle_deg])
                
        return pol_mask,center
    except:
        print("A parameter its None")



def plot_to_png(ofile,orig_img, masks, pol_mask, center,imsize, title=None, labels=None, boxes=None, scores=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
    mask_threshold = 0.6 # value to consider a pixel belongs to the object
    scr_threshold = 0.6 # only detections with score larger than this value are considered
    color=['r','b','g','k','y']
    obj_labels = ['Occ', 'CME','N/A','N/A']
    #
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(orig_img[0]), np.nan)
    fig, axs = plt.subplots(1, len(orig_img)*2, figsize=(20, 10))
    axs = axs.ravel()
    angles = [s[1] for s in pol_mask]
    ang=[]
    points=[]
    min_ang = np.radians(min(angles))
    max_ang = np.radians(max(angles))
    cpa_ang=np.radians((max_ang-min_ang)/2)
    ang.append([min_ang,max_ang,cpa_ang])
    for i in ang[0]:
        vector=np.array([np.cos(i), np.sin(i)])
        start_point = center
        end_point = start_point + imsize[0] * vector
        x_points = np.array([start_point[0], end_point[0]])
        y_points = np.array([start_point[1], end_point[1]])
        points.append([x_points,y_points])



        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plotear la primera imagen en el primer subplot
        axs[0].imshow(orig_img, cmap='gray')
        axs[0].set_title('Imagen 1')
        axs[0].axis('off')

        # Plotear la segunda imagen en el segundo subplot
        axs[1].imshow(orig_img, cmap='gray')
        axs[1].set_title('Imagen 2')
        axs[1].axis('off')

        # Ajustar el espacio entre los subplots
        plt.tight_layout()

        # Guardar la figura como una imagen
        plt.savefig(opath)


    #for i in range(len(orig_img)):
        #axs[i].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')
        #axs[i].axis('off')
        #axs[i+1].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')        
        #axs[i+1].axis('off')        
        #if boxes is not None:
            #nb = 0
            #for b in boxes[i]:
                #if scores is not None:
                #    scr = scores[i][nb]
                #else:
                #    scr = 0   
                #if scr > scr_threshold:             
                    #masked = nans.copy()
                    #masked[:, :][masks[i][nb] > mask_threshold] = nb              
                    #axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1) # add mask
       
                    #axs[i+1].plot(points[0][0], points[0][1], color='blue', label='Recta')
                    #axs[i+1].plot(points[1][0],points[1][1] , color='blue', label='Recta')
                    #axs[i+1].plot(points[2][0], points[2][1], color='blue', label='Recta')
             
                    # Ajustar límites del eje y aspecto
                    #axs[i+1].set_xlim(0, imsize[0])
                    #axs[i+1].set_ylim(0, imsize[1])
                    #axs[i+1].set_aspect('equal', adjustable='box')


                    #if labels is not None:
                    #    axs[i+1].annotate(obj_labels[labels[i][nb]]+':'+'{:.2f}'.format(scr),xy=b[0:2], fontsize=15, color=color[nb])
                #nb+=1 
    #plt.tight_layout()
    #plt.show()
    #plt.savefig(ofile)
    #plt.close()



import pickle
import os
import sys
import math
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from ext_libs.rebin import rebin


data_dir="/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v3/infer_neural_cme_seg_kincat/cor2_a/"
image_dir="/gehme/data/stereo/secchi/L0/a/img/cor2/"

imsize=[512,512] #shape of the mask image



folders = os.listdir(data_dir)
for folder in folders:
    files=os.listdir(data_dir+folder)
    for file in files:
        if file.endswith(".pkl"):
            print("processing file "+str(file))
            img_path=image_dir+str(file[:-24])+"/"+file[:-11]+".fts"
            pkl_path=data_dir+folder+"/"+file
            with open(pkl_path, 'rb') as archivo_pickle:
                mask = pickle.load(archivo_pickle)
                scores = pickle.load(archivo_pickle)
                labels= pickle.load(archivo_pickle)
                boxes= pickle.load(archivo_pickle)

            pol_mask,center=rec2pol(mask,scores,labels,boxes,imsize=imsize)
            img=read_fits(img_path)
            f=file[:-11]
            opath="/gehme-gpu/projects/2020_gcs_with_ml/repo_flor/2020_gcs_with_ml/nn/neural_cme_seg/applications/img/"+folder+str(f)+"_stats.png"#data_dir+folder+"/"+str(f)+"_stats.png"

            plot_to_png(opath,[img],[mask], title=[f],labels=[labels], boxes=[boxes], scores=[scores],pol_mask=pol_mask, center=center,imsize=imsize)
            
