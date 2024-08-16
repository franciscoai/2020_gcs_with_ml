

def manage_variables_niemela(cme_date_event,btot=False,real_img=False,auxiliar='',instr='',infer_event2=False,modified_masks=None,list_name=None):
    """
    This code is for managing the variables related to input path and outputh path
    simulation_run = 'run005'
    cme_date_event = '2011-02-15'
    instr = 'cor2_a'
    """

    #if btot:
    #    if cme_date_event == '2011-02-15':
    #        ipath = '/gehme/projects/2023_eeggl_validation/btot_pyGCS/2011-02-15/'
    #        opath = '/gehme/projects/2023_eeggl_validation/output/2011-02-15/gcs/'+instr+'/'
    if real_img:
        if cme_date_event == '2010-04-03':
            if instr[0:4] == 'cor2':
                which_stereo = instr[-1]
                ipath = '/gehme/data/stereo/secchi/L1/'+which_stereo+'/img/cor2/20100403/'
                aux = 'running_diff/neural_cme_seg_v4/'+auxiliar
                if infer_event2:
                    aux = aux+'infer2/'
                opath = '/gehme/projects/2023_eeggl_validation/niemela_project/2010-04-03/output/'+instr+'/'+aux
            if instr[0:5] == 'lasco':
                ipath = '/gehme/data/soho/lasco/level_1/c2/20100403/'
                aux = 'running_diff/neural_cme_seg_v4/'
                if infer_event2:
                    aux = aux+'infer2/'
                opath = '/gehme/projects/2023_eeggl_validation/niemela_project/2010-04-03/output/'+instr+'/'+aux
        

        if modified_masks:
            if instr == 'cor2_a':
                modified_masks = '/gehme/projects/2023_eeggl_validation/repo_diego/2020_gcs_with_ml/nn/neural_cme_seg/applications/niemela_proyect/new_masks20100403_cor2a_v4.pkl'
            if instr == 'cor2_b':
                modified_masks = '/gehme/projects/2023_eeggl_validation/repo_diego/2020_gcs_with_ml/nn/neural_cme_seg/applications/niemela_proyect/new_masks20100403_cor2b_v6.pkl'
            if instr == 'lascoC2':
                modified_masks = '/gehme/projects/2023_eeggl_validation/repo_diego/2020_gcs_with_ml/nn/neural_cme_seg/applications/niemela_proyect/new_masks20100403_lascoc2_v1.pkl'
            opath = '/gehme/projects/2023_eeggl_validation/niemela_project/2010-04-03/output/'+instr+'/'+aux
            opath = opath + 'modified_mask_v2/'
        if not list_name: 
            list_name = 'list.txt'
    #breakpoint()
    return ipath, opath,modified_masks,list_name
