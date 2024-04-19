


def manage_variables(cme_date_event,eeggl=False,btot=False,real_img=False,input_path='',output_path='',instr='',simulation_run=None,infer_event2=False,modified_masks=None,list_name=None):
    """
    This code is for managing the variables related to input path and outputh path
    simulation_run = 'run005'
    cme_date_event = '2011-02-15'
    instr = 'cor2_a'
    """
    if eeggl:
        if cme_date_event == '2011-02-15':
            ipath = '/gehme/projects/2023_eeggl_validation/eeggl_simulations/2011-02-15/'+simulation_run+'/'
            #aux = simulation_run+'/'+instr+'/runing_diff/testeo_histo/'
            aux = simulation_run+'/'+instr+'/runing_diff/neural_cme_seg_v5/'
            if infer_event2:
                aux = simulation_run+'/'+instr+'/runing_diff/neural_cme_seg_v5/infer2/test/'
            opath = '/gehme/projects/2023_eeggl_validation/output/2011-02-15/eeggl/'+aux
        
        if instr == 'cor2_a':
            aux_list = "sta_cor2"
        elif instr == 'cor2_b':
            aux_list = "stb_cor2"
        else:
            breakpoint()
        list_name = 'lista_'+aux_list+'_ordenada.txt'
    
    if btot:
        if cme_date_event == '2011-02-15':
            ipath = '/gehme/projects/2023_eeggl_validation/btot_pyGCS/2011-02-15/'
            opath = '/gehme/projects/2023_eeggl_validation/output/2011-02-15/gcs/'+instr+'/'

    if real_img:
        if cme_date_event == '2011-02-15':
            ipath = '/gehme-gpu/projects/2023_eeggl_validation/data/2011-02-14/'+instr+'/lvl1/'
            aux="occ_medida_RD_infer2/model_v5/"
            if infer_event2:
                aux = aux+'infer2/'
            opath = '/gehme/projects/2023_eeggl_validation/output/2011-02-15/'+instr+'/'+aux
        if modified_masks:
            modified_masks = '/gehme/projects/2023_eeggl_validation/repo_diego/2020_gcs_with_ml/nn/neural_cme_seg/applications/EEGGL_project/new_masks20110215_cor2a_v10.pkl'
            opath = opath + 'modified_mask_v2/'
        list_name = 'list.txt'
    breakpoint()
    return ipath, opath,modified_masks,list_name