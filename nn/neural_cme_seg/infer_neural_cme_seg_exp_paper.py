import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


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
    
    #get file event for each event
    temp = os.listdir(gcs_path)
    events_path = [os.path.join(gcs_path,d) for d in temp if str.split(d,'_')[-1] in dates]
    
    #gets .savs, andthe cor and lasco file event for each time instant in each event
    event = []
    for ev in events_path:
        cdict = {'date':[],'pro_files':[],'sav_files':[],'ima1':[], 'ima0':[],'imb1':[], 'imb0':[], 'lasco1':[],'lasco0':[]}
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
        event.append(cdict)
    return event

#main
event = get_paths_cme_exp_sources() # get all files event

for ev in event:
    for t in range(len(ev['pro_files'])):
        i0 = fits.open(ev['ima0'][t])[0].data
        i1 = fits.open(ev['ima1'][t])[0].data
        oimg = i1-i0
        plt.imshow(oimg)
        plt.show()
