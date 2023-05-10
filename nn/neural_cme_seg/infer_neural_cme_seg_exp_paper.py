import os
import sys
import numpy as np


def get_paths_cme_exp_sources():
    """
    Read all files for selected events of the CME exp sources project
    """
    #Path to the dir containing /sdo ,/soho and /stereo data directories as well as the /Polar_Observations dir.
    data_path='/gehme/data'
    #Path with our GCS data directories
    gcs_path='/gehme/projects/2019_cme_expansion/Polar_Observations/Polar_Documents/francisco/GCSs'
    #LASCO proc images Path
    lasco_path=data_path+'/soho/lasco/level_1/c2'
    #events to read
    dates =  ['20101212', '20101214', '20110317', '20110605', '20130123', '20130129',
             '20130209', '20130424', '20130502', '20130517', '20130527', '20130608']
    #get file paths
    temp = os.listdir(gcs_path)
    ok_paths = [os.path.join(gcs_path,d) for d in temp if str.split(d,'_')[-1] in dates]

    breakpoint()

    return paths


#main
get_paths_cme_exp_sources()