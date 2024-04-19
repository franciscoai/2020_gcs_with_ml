import os
import sys

def correct_path(s):
    s = s.replace("(1)", "")
    s = s.replace("(2)", "")
    return s

def convert_string(s, level):
    if level == 1:
        s = s.replace("preped/", "")
        s = s.replace("L0", "L1")
        s = s.replace("_0B", "_1B")
        s = s.replace("_04", "_14")
        s = s.replace("level1/", "")
    return s

def get_paths_cme_exp_sources(dates=None):
    """
    Read all files for selected events of the CME exp sources project
    """
    data_path = '/gehme/data'  # Path to the dir containing /sdo ,/soho and /stereo data directories as well as the /Polar_Observations dir.
    # Path with our GCS data directories
    gcs_path = '/gehme-gpu/projects/2020_gcs_with_ml/repo_mariano/2020_cme_expansion/GCSs'
    lasco_path = data_path+'/soho/lasco/level_1/c2'  # LASCO proc images Path
    secchipath = data_path+'/stereo/secchi/L0'
    level = 0  # set the reduction level of the images
    # events to read
    if dates is None:
        dates = ['20101212', '20101214', '20110317', '20110605', '20130123', '20130129',
                '20130209', '20130424', '20130502', '20130517', '20130527', '20130608']
    # pre event iamges per instrument
    pre_event = ["/soho/lasco/level_1/c2/20101212/25354377.fts",
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

    # get file event for each event
    temp = os.listdir(gcs_path)
    events_path = [os.path.join(gcs_path, d)
                   for d in temp if str.split(d, '_')[-1] in dates]

    # gets .savs, andthe cor and lasco file event for each time instant in each event
    event = []
    for ev in events_path:
        cdict = {'date': [], 'pro_files': [], 'sav_files': [], 'ima1': [], 'ima0': [], 'imb1': [
        ], 'imb0': [], 'lasco1': [], 'lasco0': [], 'pre_ima': [], 'pre_imb': [], 'pre_lasco': []}
        tinst = os.listdir(ev)
        sav_files = sorted([os.path.join(ev, f)
                           for f in tinst if f.endswith('.sav')])
        pro_files = sorted([os.path.join(ev, f) for f in tinst if (f.endswith(
            '.pro') and 'fit_' not in f and 'tevo_' not in f and 'm1.' not in f)])
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
                        cline = secchipath + line.split('\'')[1]
                        if 'cor1' in cline:
                            cor = 'cor1'
                        if 'cor2' in cline:
                            cor = 'cor2'
                        cdate = cline[cline.find(
                            '/preped/')+8:cline.find('/preped/')+16]
                        cline = convert_string(cline, level)
                        cline = correct_path(cline)
                        cdict['ima1'].append(cline)
                        ok_pro_files.append(f)
                        cpre = [s for s in pre_event if (
                            cdate in s and cor in s and '/a/' in s)]
                        if len(cpre) == 0:
                            print(
                                f'Cloud not find pre event image for {cdate}')
                            breakpoint()
                        cdict['pre_ima'].append(cpre[0])
                        cdict['pre_imb'].append([s for s in pre_event if (
                            cdate in s and cor in s and '/a/' in s)][0])
                    if 'imaprev=sccreadfits(' in line:
                        cline = convert_string(
                            secchipath + line.split('\'')[1], level)
                        cline = correct_path(cline)
                        cdict['ima0'].append(cline)
                    if 'imb=sccreadfits(' in line:
                        cline = convert_string(
                            secchipath + line.split('\'')[1], level)
                        cline = correct_path(cline)
                        cdict['imb1'].append(cline)
                    if 'imbprev=sccreadfits(' in line:
                        cline = convert_string(
                            secchipath + line.split('\'')[1], level)
                        cline = correct_path(cline)
                        cdict['imb0'].append(cline)
                    if 'lasco1=readfits' in line:
                        cline = lasco_path + line.split('\'')[1]
                        cline = correct_path(cline)
                        cdict['lasco1'].append(cline)
                        cdate = cline[cline.find(
                            '/preped/')+8:cline.find('/preped/')+16]
                        cpre = [s for s in pre_event if (
                            cdate in s and '/c2/' in s)]
                        if len(cpre) == 0:
                            print(
                                f'Cloud not find pre event image for {cdate}')
                            breakpoint()
                        cdict['pre_lasco'].append(cpre[0])
                    if 'lasco0=readfits' in line:
                        cline = lasco_path + line.split('\'')[1]
                        cline = correct_path(cline)
                        cdict['lasco0'].append(cline)
        cdict['date'] = ev
        cdict['pro_files'] = ok_pro_files
        cdict['sav_files'] = ok_sav_files
        event.append(cdict)
    return event
