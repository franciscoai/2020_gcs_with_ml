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
import datetime

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

def read_fits(file_path, header=False, imageSize=[512,512]):
    try:       
        img = fits.open(file_path)[0].data
        img = rebin(img, imageSize)   
        if header:
            return img, fits.open(file_path)[0].header
        else:
            return img 
    except:
        print(f'WARNING. I could not find file {file_path}')
        return None

def save_png(array, ofile=None, range=None):
    '''
    pltos array to an image in ofile without borders axes or anything else
    ofile: if not give only the image object is generated and returned
    range: defines color scale limits [vmin,vmax]
    '''    
    fig = plt.figure(figsize=(4,4), facecolor='white')
    if range is not None:
        vmin=range[0]
        vmax=range[1]
    else:
        vmin=None
        vmax=None
    plt.imshow(array, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)#, aspect='auto')#,extent=plotranges[sat])
    plt.axis('off')         
    if ofile is not None:
        fig.savefig(ofile, facecolor='white', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return 1
    else:
        return fig
    
def plot_to_png(ofile, orig_img, masks, title=None, labels=None, boxes=None, scores=None, save_masks=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    save_masks: set to a list of fits headers to save the masks as fits files
    """    
    mask_threshold = 0.5 # value to consider a pixel belongs to the object
    scr_threshold = 0.25 # only detections with score larger than this value are considered
    color=['b','r','g','k','y','m','c','w','b','r','g','k','y','m','c','w']
    obj_labels = ['Back', 'Occ','CME','N/A']
    #
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(orig_img[0]), np.nan)
    fig, axs = plt.subplots(2, 3, figsize=[20,10])
    axs = axs.ravel()
    for i in range(len(orig_img)):
        axs[i].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray', origin='lower')
        axs[i].axis('off')
        axs[i+3].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray', origin='lower')        
        axs[i+3].axis('off')        
        if boxes is not None:
            nb = 0
            for b in boxes[i]:
                scr = 0
                if scores is not None:
                    if scores[i][nb] is not None:
                        scr = scores[i][nb]
                if scr > scr_threshold:             
                    masked = nans.copy()
                    masked[:, :][masks[i][nb] > mask_threshold] = nb              
                    axs[i+3].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1, origin='lower')
                    box =  mpl.patches.Rectangle(b[0:2],b[2]-b[0],b[3]-b[1], linewidth=2, edgecolor=color[nb], facecolor='none') # add box
                    axs[i+3].add_patch(box)
                    if labels is not None:
                        axs[i+3].annotate(obj_labels[labels[i][nb]]+':'+'{:.2f}'.format(scr),xy=b[0:2], fontsize=15, color=color[nb])
                nb+=1
    axs[0].set_title(f'Cor A: {len(boxes[0])} objects detected') 
    axs[1].set_title(f'Cor B: {len(boxes[1])} objects detected')               
    axs[2].set_title(f'Lasco: {len(boxes[2])} objects detected')     
    if title is not None:
        fig.suptitle('\n'.join([title[i] for i in range(0,len(title))]) , fontsize=16)   
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

    if save_masks is not None:
        zeros = np.zeros(np.shape(orig_img[0]))
        for i in range(len(orig_img)):
            for nb in range(len(boxes[i])):
                scr = 0
                if scores is not None:
                    if scores[i][nb] is not None:
                        scr = scores[i][nb]
                if scr > scr_threshold:             
                    masked = zeros.copy()
                    masked[:, :][masks[i][nb] > mask_threshold] = 1
                    # safe fits
                    ofile_fits = os.path.join(os.path.dirname(ofile), os.path.basename(ofile).split('.')[0] + '_'+str(i)+'.fits')
                    h0 = save_masks[i]

                    # adapts hdr because we use smaller im size
                    sz_ratio = np.array(masked.shape)/np.array([h0['NAXIS1'], h0['NAXIS2']])
                    h0['NAXIS1'] = masked.shape[0]
                    h0['NAXIS2'] = masked.shape[1]
                    h0['CDELT1'] = h0['CDELT1']/sz_ratio[0]
                    h0['CDELT2'] = h0['CDELT2']/sz_ratio[1]
                    h0['CRPIX2'] = int(h0['CRPIX2']*sz_ratio[1])
                    h0['CRPIX1'] = int(h0['CRPIX1']*sz_ratio[1])                    

                    fits.writeto(ofile_fits, masked, h0, overwrite=True, output_verify='ignore')

     
def inference(nn_seg, ev, imgs_labels, occ_size, do_run_diff, ev_opath, filter=True):
    imageSize=[512,512]
    all_occ_size = []
    all_diff_img = []    
    all_dates = []
    all_plate_scl = []
    all_headers = []
    for t in range(len(ev['pro_files'])):
        cimga= ev[imgs_labels[0]][t]
        # sets occ radius
        if cimga.find('cor1') > 0:
            cur_occ_size = occ_size[0]
        elif cimga.find('cor2') > 0:
            cur_occ_size = occ_size[1]     
        elif cimga.find('c2') > 0:
            cur_occ_size = occ_size[2]  
        else:
            cur_occ_size = 0   
        all_occ_size.append(cur_occ_size)
        
        if do_run_diff:
            cprea = ev[imgs_labels[1]][t]
        else:
            cprea = ev[imgs_labels[2]][t]
        imga, ha = read_fits(cimga, header=True,imageSize=imageSize)
        img = imga-read_fits(cprea, imageSize=imageSize)
        all_diff_img.append(img)
        all_dates.append(datetime.datetime.strptime(ha['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f'))
        
        if ha['NAXIS1'] != imageSize[0]:
            plt_scl = ha['CDELT1'] * ha['NAXIS1']/imageSize[0] 
        else:
            plt_scl = ha['CDELT1']
        all_plate_scl.append(plt_scl)
        all_headers.append(ha)
    # inference of all images in the event
    #print(f'Running inference for {imgs_labels[0]} dates {all_dates}')
    orig_img, dates, mask, scr, labels, boxes, mask_porp =  nn_seg.infer_event(all_diff_img, all_dates, filter=filter, plate_scl=all_plate_scl, occulter_size=all_occ_size,  plot_params=ev_opath+'/'+imgs_labels[0])
    
    # adds mask_prop to headers as keywords, [mask_id,score,cpa_ang, wide_ang, apex_dist]
    for i in range(len(all_headers)):
        if mask_porp[i][1] is not None:
            all_headers[i]['NN_SCORE'] = mask_porp[i][1]
            all_headers[i]['NN_C_ANG'] = np.degrees(mask_porp[i][2])
            all_headers[i]['NN_W_ANG'] = np.degrees(mask_porp[i][3])
            # apex rom px to solar radii
            apex_sr = mask_porp[i][4] * all_plate_scl[i] / all_headers[i]['RSUN']
            all_headers[i]['NN_APEX'] = apex_sr
        else:
            all_headers[i]['NN_SCORE'] = 'None'
            all_headers[i]['NN_C_ANG'] = 'None'
            all_headers[i]['NN_W_ANG'] = 'None'
            all_headers[i]['NN_APEX'] = 'None'
            
    return  orig_img, dates, mask, scr, labels, boxes, mask_porp, all_headers

#main
#------------------------------------------------------------------Testing the CNN--------------------------------------------------------------------------
model_path= "/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4"
model_version="v4"
opath= model_path + "/infer_neural_cme_seg_exp_paper_filtered"
file_ext=".png"
trained_model = '6000.torch'
do_run_diff = True # set to use running diff instead of base diff (False)
occ_size = [90,35,75] # Artifitial occulter radius in pixels. Use 0 to avoid. [Stereo a/b C1, Stereo a/b C2, Lasco C2]
filter = True 

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
    ev_opath = os.path.join(opath, ev['date'].split('/')[-1]) + '_filter_'+str(filter)   
    #cora    
    imgs_labels = ['ima1', 'ima0','pre_ima']
    orig_imga, datesa, maska, scra, labelsa, boxesa, mask_porpa, ha = inference(nn_seg, ev, imgs_labels, occ_size, do_run_diff, ev_opath, filter=filter)
    datesa = [d.strftime('%Y-%m-%dT%H:%M:%S.%f') for d in datesa]      
    #corb
    imgs_labels = ['imb1', 'imb0','pre_imb']
    orig_imgb, datesb, maskb, scrb, labelsb, boxesb, mask_porpb, hb  = inference(nn_seg, ev, imgs_labels, occ_size, do_run_diff, ev_opath, filter=filter) 
    datesb = [d.strftime('%Y-%m-%dT%H:%M:%S.%f') for d in datesb]        
    #lasco
    imgs_labels = ['lasco1', 'lasco0','pre_lasco']
    orig_imgl, datesl, maskl, scrl, labelsl, boxesl, mask_porpl, hl  = inference(nn_seg, ev, imgs_labels, occ_size, do_run_diff, ev_opath, filter=filter)
    datesl = [d.strftime('%Y-%m-%dT%H:%M:%S.%f') for d in datesl]    
    
    for t in range(len(ev['pro_files'])):
        ofile = os.path.join(ev_opath,os.path.basename(ev['pro_files'][t])+'.png')
        if filter:
            plot_to_png(ofile, [orig_imga[t],orig_imgb[t], orig_imgl[t]], [[maska[t]],[maskb[t]],[maskl[t]]], \
                        title=[datesa[t], datesb[t], datesl[t]],labels=[[labelsa[t]],[labelsb[t]], [labelsl[t]]], \
                        boxes=[[boxesa[t]], [boxesb[t]], [boxesl[t]]], scores=[[scra[t]], [scrb[t]], [scrl[t]]], save_masks=[ha[t],hb[t],hl[t]])
        else:
            plot_to_png(ofile, [orig_imga[t],orig_imgb[t], orig_imgl[t]], [maska[t],maskb[t],maskl[t]], \
                    title=[datesa[t], datesb[t], datesl[t]],labels=[labelsa[t],labelsb[t], labelsl[t]], \
                    boxes=[boxesa[t], boxesb[t], boxesl[t]], scores=[scra[t], scrb[t], scrl[t]])         
