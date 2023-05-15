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

def convert_string(s):
    s = s.replace("(1)", "")
    s = s.replace("preped/", "")
    s = s.replace("L0", "L1")
    s = s.replace("_0B", "_1B")
    s = s.replace("_04", "_14")
    s = s.replace("level1/", "")    
    return s

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
    pre_event =["/soho/lasco/level_1/c2/20101212/25354377.fts",
                "/stereo/secchi/L1/a/seq/cor1/20101212/20101212_022500_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20101212/20101212_023500_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20101212/20101212_015400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20101212/20101212_015400_14c2B.fts",
                "/soho/lasco/level_1/c2/20101214/25354679.fts",
                "/stereo/secchi/L1/a/seq/cor1/20101214/20101214_150000_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20101214/20101214_150000_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20101214/20101214_152400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20101214/20101214_153900_14c2B.fts",
                "/soho/lasco/level_1/c2/20110317/25365446.fts",
                "/stereo/secchi/L1/a/seq/cor1/20110317/20110317_103500_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20110317/20110317_103500_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20110317/20110317_115400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20110317/20110317_123900_14c2B.fts",
                "/soho/lasco/level_1/c2/20110605/25374823.fts",
                "/stereo/secchi/L1/a/seq/cor1/20110605/20110605_021000_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20110605/20110605_021000_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20110605/20110605_043900_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20110605/20110605_043900_14c2B.fts",
                "/soho/lasco/level_1/c2/20130123/25445617.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130123/20130123_131500_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130123/20130123_125500_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130123/20130123_135400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130123/20130123_142400_14c2B.fts",
                "/soho/lasco/level_1/c2/20130129/25446296.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130129/20130129_012500_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130129/20130129_012500_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130129/20130129_015400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130129/20130129_015400_14c2B.fts",
                "/soho/lasco/level_1/c2/20130209/25447666.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130209/20130209_054000_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130209/20130209_054500_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130209/20130209_062400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130209/20130209_062400_14c2B.fts",
                "/soho/lasco/level_1/c2/20130424_1/25456651.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130424/20130424_051500_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130424/20130424_051500_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130424/20130424_055400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130424/20130424_065400_14c2B.fts",
                "/soho/lasco/level_1/c2/20130502/25457629.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130502/20130502_045000_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130502/20130502_050000_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130502/20130502_012400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130502/20130502_053900_14c2B.fts"
                "/soho/lasco/level_1/c2/20130517/25459559.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130517/20130517_194500_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130517/20130517_194500_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130517/20130517_203900_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130517/20130517_205400_14c2B.fts",
                "/soho/lasco/level_1/c2/20130527_2/25460786.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130527/20130527_183000_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130527/20130527_183000_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130527/20130527_192400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130527/20130527_195400_14c2B.fts",
                "/soho/lasco/level_1/c2/20130608/25462149.fts",
                "/stereo/secchi/L1/a/seq/cor1/20130607/20130607_221500_1B4c1A.fts",
                "/stereo/secchi/L1/b/seq/cor1/20130607/20130607_223000_1B4c1B.fts",
                "/stereo/secchi/L1/a/img/cor2/20130607/20130607_225400_14c2A.fts",
                "/stereo/secchi/L1/b/img/cor2/20130607/20130607_232400_14c2B.fts"]
    pre_event = [data_path + f for f in pre_event]

    #get file event for each event
    temp = os.listdir(gcs_path)
    events_path = [os.path.join(gcs_path,d) for d in temp if str.split(d,'_')[-1] in dates]
    
    #gets .savs, andthe cor and lasco file event for each time instant in each event
    event = []
    for ev in events_path:
        cdict = {'date':[],'pro_files':[],'sav_files':[],'ima1':[], 'ima0':[],'imb1':[], 'imb0':[], 'lasco1':[],'lasco0':[],'pre_ima':[],'pre_imb':[],'pre_lasco':[]}
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
                        cline = secchipath +line.split('\'')[1]
                        if 'cor1' in cline:
                            cor = 'cor1'
                        if 'cor2' in cline:
                            cor = 'cor2'
                        cdate = cline[cline.find('/preped/')+8:cline.find('/preped/')+16]
                        cline = convert_string(cline)
                        cdict['ima1'].append(cline)
                        ok_pro_files.append(f)                           
                        cpre = [s for s in pre_event if (cdate in s and cor in s and '/a/' in s )]
                        if len(cpre) == 0:
                            print(f'Cloud not find pre event image for {cdate}')
                            breakpoint()
                        cdict['pre_ima'].append(cpre[0])
                        cdict['pre_imb'].append([s for s in pre_event if (cdate in s and  cor in s and '/a/' in s )][0])
                    if 'imaprev=sccreadfits(' in line:
                        cline = convert_string(secchipath +line.split('\'')[1])
                        cdict['ima0'].append(cline)
                    if 'imb=sccreadfits(' in line:
                        cline = convert_string(secchipath +line.split('\'')[1])
                        cdict['imb1'].append(cline)
                    if 'imbprev=sccreadfits(' in line:
                        cline = convert_string(secchipath +line.split('\'')[1])
                        cdict['imb0'].append(cline)          
                    if 'lasco1=readfits' in line:
                        cline = lasco_path +line.split('\'')[1]
                        cdict['lasco1'].append(cline)  
                        cdate = cline[cline.find('/preped/')+8:cline.find('/preped/')+16]
                        cpre= [s for s in pre_event if (cdate in s and '/c2/' in s)]
                        if len(cpre) == 0:
                            print(f'Cloud not find pre event image for {cdate}')
                            breakpoint()                        
                        cdict['pre_lasco'].append(cpre[0])                                           
                    if 'lasco0=readfits' in line:
                        cdict['lasco0'].append(lasco_path +line.split('\'')[1]) 
        cdict['date']=ev
        cdict['pro_files']=ok_pro_files
        cdict['sav_files']=ok_sav_files                                                      
        event.append(cdict)
    return event

def normalize(image):
    '''
    Normalizes the values of the input image to have a given range (as fractions of the sd around the mean)
    maped to [0,1]. It clips output values outside [0,1]
    '''
    sd_range=1.
    m = np.mean(image)
    sd = np.std(image)
    image = (image - m + sd_range * sd) / (2 * sd_range * sd)
    image[image >1]=1
    image[image <0]=0
    return image

#main
#------------------------------------------------------------------Testing the CNN-----------------------------------------------------------------
model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_new/"
opath= model_path + "/infer_neural_cme_seg_exp_paper"
file_ext=".png"
trained_model = '8000.torch'
imageSize=[512,512]
do_run_diff = True # set to use running diff images

#main
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unles its not available
print(f'Using device:  {device}')

event = get_paths_cme_exp_sources() # get all files event
os.makedirs(opath, exist_ok=True)
for ev in event:
    for t in range(len(ev['pro_files'])):
        cimg= ev['ima1'][t]
        cpre = ev['pre_ima'][t]
        #reads files and computes a base diff
        if t == 0 or do_run_diff:
            i0 = fits.open(cpre)[0].data
            i0 = rebin(i0, imageSize)
        try:
            i1 = fits.open(cimg)[0].data
            i1 = rebin(i1, imageSize)
            img = i1-i0

            #loads model

            # # Initialize the Weight Transforms
            # weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            # preprocess = weights.transforms()
            # img_transformed = preprocess(img)

            model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) 
            in_features = model.roi_heads.box_predictor.cls_score.in_features 
            model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=2)
            model.load_state_dict(torch.load(model_path + "/"+ trained_model)) #loads the last iteration of training 
            model.to(device)# move model to the right device
            model.eval()#set the model to evaluation state

            #inference
            print('Inference of imge:'+cimg)
            images = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            images = cv2.resize(images, imageSize, cv2.INTER_LINEAR)        
            images = normalize(images) #cv2.normalize(images, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # normalize to 0,1
            oimages = images.copy()                
            print(np.min(images), np.max(images), np.mean(images), np.std(images))            
            images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
            images=images.swapaxes(1, 3).swapaxes(2, 3)
            images = list(image.to(device) for image in images)
            with torch.no_grad(): #runs the image through the net and gets a prediction for the object in the image.
                pred = model(images)

            # mask image and saves it along with the original
            img =  np.array(oimages)[:,:,0]
            im2 = img.copy()
            nmasks = len(pred[0]['masks'])
            color=[-255,-255]
            if nmasks > 0:
                for i in range(nmasks):
                    msk=pred[0]['masks'][i,0].detach().cpu().numpy()
                    scr=pred[0]['scores'][i].detach().cpu().numpy()
                    if scr>0.5 : # Score threshold to conside a valid mask
                        im2[:, :][msk > 0.5] = color[i] # assumes a likelyhood threshold of 0.5
                    else:
                        scr="below_0.8" 
            else:
                scr="NO_mask" 
            ofile = os.path.join(opath,os.path.basename(ev['pro_files'][t])+'_scr_'+str(scr)+'.png')
            #print(f'Saving file {ofile}')

            #saves in png
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[15, 5])
            ax1.imshow(img, vmin=0, vmax=1, cmap='gray')
            ax2.imshow(im2, vmin=0, vmax=1, cmap='gray')
            plt.tight_layout()
            plt.savefig(ofile)
            plt.close()

            # #write fits
            # occ = fits.PrimaryHDU(pic)
            # occ.writeto(ofile)            
        except:
            print(f'WARNING. I could not find file {cimg}, skipping...')