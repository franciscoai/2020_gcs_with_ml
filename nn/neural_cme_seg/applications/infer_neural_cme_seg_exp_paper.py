import os
import sys
from astropy.io import fits
import numpy as np
import torch.utils.data
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from ext_libs.rebin import rebin
import csv
from nn.neural_cme_seg.neural_cme_seg import neural_cme_segmentation

def correct_path(s):
    s = s.replace("(1)", "")
    s = s.replace("(2)", "")
    return s

def convert_string(s, level):   
    if level==1:
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
    level=0 # set the reduction level of the images
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
                        cline = convert_string(cline, level)
                        cline = correct_path(cline)
                        cdict['ima1'].append(cline)
                        ok_pro_files.append(f)                           
                        cpre = [s for s in pre_event if (cdate in s and cor in s and '/a/' in s )]
                        if len(cpre) == 0:
                            print(f'Cloud not find pre event image for {cdate}')
                            breakpoint()
                        cdict['pre_ima'].append(cpre[0])
                        cdict['pre_imb'].append([s for s in pre_event if (cdate in s and  cor in s and '/a/' in s )][0])
                    if 'imaprev=sccreadfits(' in line:
                        cline = convert_string(secchipath +line.split('\'')[1], level)
                        cline = correct_path(cline)
                        cdict['ima0'].append(cline)
                    if 'imb=sccreadfits(' in line:
                        cline = convert_string(secchipath +line.split('\'')[1], level)
                        cline = correct_path(cline)
                        cdict['imb1'].append(cline)
                    if 'imbprev=sccreadfits(' in line:
                        cline = convert_string(secchipath +line.split('\'')[1], level)
                        cline = correct_path(cline)
                        cdict['imb0'].append(cline)          
                    if 'lasco1=readfits' in line:
                        cline = lasco_path +line.split('\'')[1]
                        cline = correct_path(cline)
                        cdict['lasco1'].append(cline)  
                        cdate = cline[cline.find('/preped/')+8:cline.find('/preped/')+16]
                        cpre= [s for s in pre_event if (cdate in s and '/c2/' in s)]
                        if len(cpre) == 0:
                            print(f'Cloud not find pre event image for {cdate}')
                            breakpoint()                        
                        cdict['pre_lasco'].append(cpre[0])                                           
                    if 'lasco0=readfits' in line:
                        cline = lasco_path +line.split('\'')[1]
                        cline = correct_path(cline)
                        cdict['lasco0'].append(cline) 
        cdict['date']=ev
        cdict['pro_files']=ok_pro_files
        cdict['sav_files']=ok_sav_files                                                      
        event.append(cdict)
    return event

def read_fits(file_path):
    imageSize=[512,512]
    try:       
        img = fits.open(file_path)[0].data
        img = rebin(img, imageSize)   
        return img  
    except:
        print(f'WARNING. I could not find file {file_path}')
        return None

def plot_to_png(ofile,orig_img, masks, title=None, labels=None, boxes=None, scores=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
    mask_threshold = 0.5 # value to consider a pixel belongs to the object
    scr_threshold = 0.3 # only detections with score larger than this value are considered
    color=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w']
    obj_labels = ['Back', 'Occ','CME','N/A']
    #
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(orig_img[0]), np.nan)
    fig, axs = plt.subplots(2, 3, figsize=[20,10])
    axs = axs.ravel()
    for i in range(len(orig_img)):
        axs[i].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')
        axs[i].axis('off')
        axs[i+3].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')        
        axs[i+3].axis('off')        
        if boxes is not None:
            nb = 0
            for b in boxes[i]:
                if scores is not None:
                    scr = scores[i][nb]
                else:
                    scr = 0   
                if scr > scr_threshold:             
                    masked = nans.copy()
                    masked[:, :][masks[i][nb] > mask_threshold] = nb              
                    axs[i+3].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1) # add mask
                    box =  mpl.patches.Rectangle(b[0:2],b[2]-b[0],b[3]-b[1], linewidth=2, edgecolor=color[nb], facecolor='none') # add box
                    axs[i+3].add_patch(box)
                    if labels is not None:
                        axs[i+3].annotate(obj_labels[labels[i][nb]]+':'+'{:.2f}'.format(scr),xy=b[0:2], fontsize=15, color=color[nb])
                nb+=1
    axs[0].set_title(f'Cor A: {len(boxes[0])} objects detected') 
    axs[1].set_title(f'Cor B: {len(boxes[1])} objects detected')               
    axs[2].set_title(f'Lasco: {len(boxes[2])} objects detected')     
    if title is not None:
        fig.suptitle('\n'.join([title[i]+' ; '+title[i+1] for i in range(0,len(title),2)]) , fontsize=16)   

    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()
    # #write fits
    # occ = fits.PrimaryHDU(pic)
    # occ.writeto(ofile)   
     
      
#main
#------------------------------------------------------------------Testing the CNN-----------------------------------------------------------------
model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4"
model_version="v4"
opath= model_path + "/infer_neural_cme_seg_exp_paper"
file_ext=".png"
trained_model = '6000.torch'
do_run_diff = True # set to use running diff instead of base diff (False)
occ_size = [0,45,90] # Artifitial occulter radius in pixels. Use 0 to avoid. [Stereo a/b C1, Stereo a/b C2, Lasco C2]

#main
gpu=0 # GPU to use
device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu') #runing on gpu unless its not available
print(f'Using device:  {device}')

os.makedirs(opath, exist_ok=True)
event = get_paths_cme_exp_sources() # get all files event
file = open(opath + '/all_events.csv', 'w')
writer = csv.writer(file)
writer.writerow(event[0].keys())
for e in event:
    writer.writerow(e.values())
file.close()

#loads nn model
nn_seg = neural_cme_segmentation(device, pre_trained_model = model_path + "/"+ trained_model, version=model_version)

for ev in event:
    print(f'Processing event {ev["date"]}')

    #cora    
    all_occ_size = []
    all_diff_img = []    
    for t in range(len(ev['pro_files'])):
        cimga= ev['ima1'][t]
        # sets occ radius
        if cimga.find('cor1') > 0:
            cur_occ_size = occ_size[0]
        elif cimga.find('cor2') > 0:
            cur_occ_size = occ_size[1]     
        else:
            cur_occ_size = 0   
        all_occ_size.append(cur_occ_size)
        
        if do_run_diff:
            cprea = ev['ima0'][t]
        else:
            cprea = ev['pre_ima'][t]
            
        img = read_fits(cimga) -read_fits(cprea) 
        all_diff_img.append(img)
    # inference of all images in the event
    orig_img, mask, scr, labels, boxes  = nn_seg.infer_event(all_diff_img, occulter_size=all_occ_size)

    #corb
    all_occ_size = []
    all_diff_img = []
    for t in range(len(ev['pro_files'])):
        cimgb= ev['imb1'][t]
        # sets occ radius
        if cimgb.find('cor1') > 0:
            cur_occ_size = occ_size[0]
        elif cimgb.find('cor2') > 0:
            cur_occ_size = occ_size[1]     
        else:
            cur_occ_size = 0  

        if do_run_diff:
            cpreb = ev['imb0'][t]
        else:
            cpreb = ev['pre_imb'][t]
        print('Inference of imge:'+cimgb)
        try:
            img = read_fits(cimgb) - read_fits(cpreb)
            orig_imgb, maskb, scrb, labelsb, boxesb = nn_seg.infer(img, occulter_size=cur_occ_size)
        except:
            orig_imgb = np.zeros((512,512))
            print(f'Inference skipped')
    # inference of all images in the event
    orig_img, mask, scr, labels, boxes  = nn_seg.infer_event(all_diff_img, occulter_size=all_occ_size)
            

    #lasco
    all_occ_size = []
    all_diff_img = []
    for t in range(len(ev['pro_files'])):
        cimgl= ev['lasco1'][t]
        if do_run_diff:
            cprel = ev['lasco0'][t]
        else:
            cprel = ev['pre_lasco'][t]
        print('Inference of imge:'+cimgl)
        try:
            img = read_fits(cimgl) - read_fits(cprel)
            orig_imgl, maskl, scrl, labelsl, boxesl = nn_seg.infer(img, occulter_size=occ_size[2])
        except:
            orig_imgl = np.zeros((512,512))
            print(f'Inference skipped')   

    # inference of all images in the event
    orig_img, mask, scr, labels, boxes  = nn_seg.infer_event(all_diff_img, occulter_size=all_occ_size)
            


# for ev in event:
#     print(f'Processing event {ev["date"]}')
#     for t in range(len(ev['pro_files'])):
#         #cora
#         cimga= ev['ima1'][t]
#         # sets occ radius
#         if cimga.find('cor1') > 0:
#             cur_occ_size = occ_size[0]
#         elif cimga.find('cor2') > 0:
#             cur_occ_size = occ_size[1]     
#         else:
#             cur_occ_size = 0   

#         if do_run_diff:
#             cprea = ev['ima0'][t]
#         else:
#             cprea = ev['pre_ima'][t]
            
#         print('Inference of imge:'+cimga)     
#         try:
#             img = read_fits(cimga) -read_fits(cprea) 
#             orig_imga, maska, scra, labelsa, boxesa  = nn_seg.infer(img, occulter_size=cur_occ_size)
#         except:
#             orig_imga = np.zeros((512,512))
#             print(f'Inference skipped')

#         #corb
#         cimgb= ev['imb1'][t]
#         # sets occ radius
#         if cimgb.find('cor1') > 0:
#             cur_occ_size = occ_size[0]
#         elif cimgb.find('cor2') > 0:
#             cur_occ_size = occ_size[1]     
#         else:
#             cur_occ_size = 0  

#         if do_run_diff:
#             cpreb = ev['imb0'][t]
#         else:
#             cpreb = ev['pre_imb'][t]
#         print('Inference of imge:'+cimgb)
#         try:
#             img = read_fits(cimgb) - read_fits(cpreb)
#             orig_imgb, maskb, scrb, labelsb, boxesb = nn_seg.infer(img, occulter_size=cur_occ_size)
#         except:
#             orig_imgb = np.zeros((512,512))
#             print(f'Inference skipped')
                
#         #lasco
#         cimgl= ev['lasco1'][t]
#         if do_run_diff:
#             cprel = ev['lasco0'][t]
#         else:
#             cprel = ev['pre_lasco'][t]
#         print('Inference of imge:'+cimgl)
#         try:
#             img = read_fits(cimgl) - read_fits(cprel)
#             orig_imgl, maskl, scrl, labelsl, boxesl = nn_seg.infer(img, occulter_size=occ_size[2])
#         except:
#             orig_imgl = np.zeros((512,512))
#             print(f'Inference skipped')   

#         ofile = os.path.join(opath,os.path.basename(ev['pro_files'][t])+'.png')
#         plot_to_png(ofile, [orig_imga,orig_imgb, orig_imgl], [maska,maskb,maskl], \
#                     title=[cimga, cprea, cimgb, cpreb, cimgl, cprel], labels=[labelsa,labelsb, labelsl], \
#                     boxes=[boxesa, boxesb, boxesl], scores=[scra, scrb, scrl])
