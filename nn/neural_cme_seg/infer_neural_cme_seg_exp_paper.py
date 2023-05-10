import os
import sys
from astropy.io import fits
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from ext_libs.rebin import rebin

def get_paths_cme_exp_sources():
    """
    Read all files for selected events of the CME exp sources project
    """
    data_path='/gehme/data' #Path to the dir containing /sdo ,/soho and /stereo data directories as well as the /Polar_Observations dir.
    gcs_path='/gehme/projects/2019_cme_expansion/Polar_Observations/Polar_Documents/francisco/GCSs'     #Path with our GCS data directories
    lasco_path=data_path+'/soho/lasco/level_1/c2'     #LASCO proc images Path
    secchipath=data_path+'/stereo/secchi/L0'
    #events to read
    dates =  ['20101212', '20101214', '20110317', '20110605', '20130123', '20130129',
             '20130209', '20130424', '20130502', '20130517', '20130527', '20130608']
    # pre event iamges per instrument
    pre_a = ['20130424_055400_d4c2A.fts']
    pre_b = ['20130424_055400_d4c2A.fts']
    pre_lasco = ['20130424_055400_d4c2A.fts']

    #get file event for each event
    temp = os.listdir(gcs_path)
    events_path = [os.path.join(gcs_path,d) for d in temp if str.split(d,'_')[-1] in dates]
    
    #gets .savs, andthe cor and lasco file event for each time instant in each event
    event = []
    for ev in events_path:
        cdict = {'date':[],'pro_files':[],'sav_files':[],'ima1':[], 'ima0':[],'imb1':[], 'imb0':[], 'lasco1':[],'lasco0':[],'pre_event':[]}
        tinst = os.listdir(ev)
        sav_files = sorted([os.path.join(ev,f) for f in tinst if f.endswith('.sav')])
        pro_files = sorted([os.path.join(ev,f)  for f in tinst if (f.endswith('.pro') and 'fit_' not in f and 'tevo_' not in f and 'm1.' not in f)])
        if len(sav_files) != len(pro_files):
            os.error('ERROR. Found different number of .sav and .pro files')
            sys.exit
        # reads the lasco and stereo files from within each pro
        ok_pro_files = []
        ok_sav_files = []
        for f in pro_files:
            with open(f) as of:
                for line in of:
                    if 'ima=sccreadfits(' in line:
                        cdict['ima1'].append(secchipath +line.split('\'')[1])
                        ok_pro_files.append(f)
                    if 'imaprev=sccreadfits(' in line:
                        cdict['ima0'].append(secchipath +line.split('\'')[1])
                    if 'imb=sccreadfits(' in line:
                        cdict['imb1'].append(secchipath +line.split('\'')[1])
                    if 'imbprev=sccreadfits(' in line:
                        cdict['imb0'].append(secchipath +line.split('\'')[1])          
                    if 'lasco1=readfits' in line:
                        cdict['lasco1'].append(lasco_path +line.split('\'')[1])
                    if 'lasco0=readfits' in line:
                        cdict['lasco0'].append(lasco_path +line.split('\'')[1]) 
        cdict['date']=ev
        cdict['pro_files']=ok_pro_files
        cdict['sav_files']=ok_sav_files    
        #cdict['pre_a']=                                                    
        event.append(cdict)
    # print('*****Eventos Cor A')        
    # [print(event[i]['ima0'][0]) for i in range(len(event))]
    # print('*****Eventos Cor B')
    # [print(event[i]['imb0'][0]) for i in range(len(event))]
    # print('*****Eventos Lasco')    
    # [print(event[i]['lasco0'][0]) for i in range(len(event))]
    # breakpoint()
    return event

#main

#------------------------------------------------------------------Testing the CNN-----------------------------------------------------------------
dataDir = '/gehme-gpu/projects/2020_gcs_with_ml/data/cme_seg_dataset_fran_test'
model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_fran/"
opath= "/gehme-gpu/projects/2020_gcs_with_ml/output/infer_neural_cme_seg_exp_paper"
file_ext=".png"
trained_model = '4999.torch'
testDir=  dataDir 
imageSize=[512,512]
do_run_diff = True # set to use running diff images

#main
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unles its not available
print(f'Using device:  {device}')

event = get_paths_cme_exp_sources() # get all files event
os.makedirs(opath, exist_ok=True)
for ev in event:
    for t in range(len(ev['pro_files'])):
        #reads files and computes a base diff
        if t == 0 or do_run_diff:
            i0 = fits.open(ev['ima0'][t])[0].data
            i0 = rebin(i0, imageSize)
        i1 = fits.open(ev['ima1'][t])[0].data
        i1 = rebin(i1, imageSize)
        img = np.abs(i1-i0)

        #loads model
        model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) 
        in_features = model.roi_heads.box_predictor.cls_score.in_features 
        model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=2)
        model.load_state_dict(torch.load(model_path + "/"+ trained_model)) #loads the last iteration of training 
        model.to(device)# move model to the right device
        model.eval()#set the model to evaluation state

        #inference
        print('Inference of imge:'+ev['ima1'][t])
        images = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)
        oimages = images.copy()
        images = cv2.normalize(images, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) # normalize to 0,1
        images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
        images=images.swapaxes(1, 3).swapaxes(2, 3)
        images = list(image.to(device) for image in images)
        with torch.no_grad(): #runs the image through the net and gets a prediction for the object in the image.
            pred = model(images)

        #The predicted object ‘masks’ are saved as a matrix in the same size as the image with each pixel 
        #having a value that corresponds to how likely it is part of the object. And only displays the ones with scores larger than 0.8
        #ssume that only pixels which values larger than 0.5 are likely to be part of the objects.
        #We display this by marking these pixels with a different random color for each object

        img =  np.array(oimages)[:,:,0]
        im2 = img.copy()
        nmasks = len(pred[0]['masks'])
        color=[0,255]
        if nmasks > 0:
            for i in range(nmasks):
                msk=pred[0]['masks'][i,0].detach().cpu().numpy()
                scr=pred[0]['scores'][i].detach().cpu().numpy()
                if scr>0.8 :
                    im2[:, :][msk > 0.5] = color[i]
                else:
                    scr="below_0.8" 
        else:
            scr="NO_mask" 
        ofile = os.path.join(opath,os.path.basename(ev['pro_files'][t])+'_scr_'+str(scr)+'.png')
        print(f'Saving file {ofile}')

        #saves in png
        im_range = 0.5 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[15, 5])
        m = np.mean(img)
        sd = np.std(img)
        ax1.imshow(img, vmin=m-im_range*sd, vmax=m+im_range*sd, cmap='gray')
        ax2.imshow(im2, vmin=m-im_range*sd, vmax=m+im_range*sd, cmap='gray')
        plt.tight_layout()
        plt.savefig(ofile)
        plt.close()

        # #write fits
        # occ = fits.PrimaryHDU(pic)
        # occ.writeto(ofile)