from descargar_imagenes_clases import lascoc2_downloader
import pandas as pd
from datetime import datetime, timedelta

descarga_nrl = True
tabla = pd.read_csv("/gehme/projects/2020_gcs_with_ml/repo_diego/2020_gcs_with_ml/nn_training/corona_background/catalogues/Lista_Final_CMEs.csv", sep='\t', engine='python',encoding="utf-8", header=0)
tabla['pre_a_1h_download_c2'] = ''
tabla['pre_b_1h_download_c2'] = ''
tabla['pre_a_2h_download_c2'] = ''
tabla['pre_b_2h_download_c2'] = ''

for i in range(len(tabla)):
    print("chequeando elemento Numero {}".format(i))
    pre_even_a_1h = tabla['preevento_a_1h'][i]
    if (pre_even_a_1h != '*' and pre_even_a_1h != 'NaT'): 
        dt = datetime.strptime(pre_even_a_1h, '%Y-%m-%d %H:%M:%S')
        dt_ini = dt - timedelta(minutes=40)
        ini = dt_ini.strftime('%Y/%m/%d %H:%M:%S')
        dt_fin = dt + timedelta(minutes=40)
        fin = dt_fin.strftime('%Y/%m/%d %H:%M:%S')
        
        asd = lascoc2_downloader(start_time=ini,end_time=fin,nivel='level_05',size=2)
        #breakpoint()
        asd.search()
        if len(asd.search_lascoc2) >= 1:
            start_times = asd.search_lascoc2['Start Time']
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)
            asd.indices_descarga = [indice_fecha_cercana] 

            
            folder_year_month_day = asd.search_lascoc2[asd.indices_descarga[0]]['fileid'].split('/')[8]
            folder = asd.search_lascoc2['fileid'][asd.indices_descarga[0]].split('/')[-3]
            if int(folder[0:2]) < 50:
                suffix = str(2000+int(folder[0:2]))
            else:
                suffix = str(1900+int(folder[0:2])) 
            folder_full = suffix+folder[2:]+"/"
            
            fileid = str(asd.search_lascoc2[asd.indices_descarga[0]]['fileid']).split('/')[-1]

            asd.nrl_download = True
            asd.download()
            breakpoint()
            if not asd.nrl_download:
                asd.download()
                download_path = asd.dir_descarga+"level_05/c2/"+folder_full
            if asd.nrl_download:
                file_id = asd.search_lascoc2[asd.indices_descarga[0]]['fileid']
                download_path = asd.dir_descarga+"level_1/c2/"+folder_full
                asd.nrl_navy_download(file_id, download_path)

            breakpoint()

            tabla['pre_a_1h_download_c2'][i] = download_path+fileid

        else:
            tabla['pre_a_1h_download_c2'][i] = 'No data'
    else:
        tabla['pre_a_1h_download_c2'][i] = '*'

    pre_even_a_2h = tabla['preevento_a_2h'][i]
    if (pre_even_a_2h != '*' and pre_even_a_2h != 'NaT'):
        dt = datetime.strptime(pre_even_a_2h, '%Y-%m-%d %H:%M:%S')
        dt_ini = dt - timedelta(minutes=40)
        ini = dt_ini.strftime('%Y/%m/%d %H:%M:%S')
        dt_fin = dt + timedelta(minutes=40)
        fin = dt_fin.strftime('%Y/%m/%d %H:%M:%S')

        asd = lascoc2_downloader(start_time=ini,end_time=fin,nivel='level_05',size=2)
        asd.search()
        
        if len(asd.search_lascoc2) >= 1:
            start_times = asd.search_lascoc2['Start Time']#[0:-1]
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)
            asd.indices_descarga = [indice_fecha_cercana]
        
            #asd.download()            
            folder_year_month_day = asd.search_lascoc2[asd.indices_descarga[0]]['fileid'].split('/')[8]
            folder = asd.search_lascoc2['fileid'][asd.indices_descarga[0]].split('/')[-3]
            if int(folder[0:2]) < 50:
                suffix = str(2000+int(folder[0:2]))
            else:
                suffix = str(1900+int(folder[0:2])) 
            folder_full = suffix+folder[2:]+"/"
            
            fileid = str(asd.search_lascoc2[asd.indices_descarga[0]]['fileid']).split('/')[-1]
            
            asd.nrl_download = True
            if not asd.nrl_download:
                asd.download()
                download_path = asd.dir_descarga+"level_05/c2/"+folder_full
            if asd.nrl_download:
                file_id = asd.search_lascoc2[asd.indices_descarga[0]]['fileid']
                download_path = asd.dir_descarga+"level_1/c2/"+folder_full
                asd.nrl_navy_download(file_id, download_path)

            breakpoint()
            tabla['pre_a_2h_download_c2'][i] = download_path+fileid
        else:
            tabla['pre_a_2h_download_c2'][i] = 'No data'
    else:
        tabla['pre_a_2h_download_c2'][i] = '*'

    pre_even_b_1h = tabla['preevento_b_1h'][i]
    if (pre_even_b_1h != '*' and pre_even_b_1h != 'NaT'):
        dt = datetime.strptime(pre_even_b_1h, '%Y-%m-%d %H:%M:%S')
        dt_ini = dt - timedelta(minutes=40)
        ini = dt_ini.strftime('%Y/%m/%d %H:%M:%S')
        dt_fin = dt + timedelta(minutes=40)
        fin = dt_fin.strftime('%Y/%m/%d %H:%M:%S')

        asd = lascoc2_downloader(start_time=ini,end_time=fin,nivel='level_05',size=2)
        asd.search()
        #breakpoint()
        if len(asd.search_lascoc2) >= 1:
            start_times = asd.search_lascoc2['Start Time']#[0:-1]
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)
            asd.indices_descarga = [indice_fecha_cercana]
        
            asd.download()
            folder_year_month_day = asd.search_lascoc2[asd.indices_descarga[0]]['fileid'].split('/')[8]
            folder = asd.search_lascoc2['fileid'][asd.indices_descarga[0]].split('/')[-3]
            if int(folder[0:2]) < 50:
                suffix = str(2000+int(folder[0:2]))
            else:
                suffix = str(1900+int(folder[0:2])) 
            folder_full = suffix+folder[2:]+"/"
            download_path = asd.dir_descarga+"level_05/c2/"+folder_full
            fileid = str(asd.search_lascoc2[asd.indices_descarga[0]]['fileid']).split('/')[-1]
            tabla['pre_b_1h_download_c2'][i] = download_path+fileid
            #"/".join(str(asd.search_lascoc2[asd.indices_descarga]['fileid']).split('/')[1:])
  #          breakpoint()
        else:
            tabla['pre_b_1h_download_c2'][i] = 'No data'
    else:
        tabla['pre_b_1h_download_c2'][i] = '*'

    pre_even_b_2h = tabla['preevento_b_2h'][i]
    if (pre_even_b_2h != '*' and pre_even_b_2h != 'NaT'):
        dt = datetime.strptime(pre_even_b_2h, '%Y-%m-%d %H:%M:%S')
        dt_ini = dt - timedelta(minutes=40)
        ini = dt_ini.strftime('%Y/%m/%d %H:%M:%S')
        dt_fin = dt + timedelta(minutes=40)
        fin = dt_fin.strftime('%Y/%m/%d %H:%M:%S')

        asd = lascoc2_downloader(start_time=ini,end_time=fin,nivel='level_05',size=2)
        asd.search()
        if len(asd.search_lascoc2) >= 1:
            start_times = asd.search_lascoc2['Start Time']#[0:-1]
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)
            asd.indices_descarga = [indice_fecha_cercana]
        
            asd.download()
            folder_year_month_day = asd.search_lascoc2[asd.indices_descarga[0]]['fileid'].split('/')[8]
            folder = asd.search_lascoc2['fileid'][asd.indices_descarga[0]].split('/')[-3]
            if int(folder[0:2]) < 50:
                suffix = str(2000+int(folder[0:2]))
            else:
                suffix = str(1900+int(folder[0:2])) 
            folder_full = suffix+folder[2:]+"/"
            download_path = asd.dir_descarga+"level_05/c2/"+folder_full
            fileid = str(asd.search_lascoc2[asd.indices_descarga[0]]['fileid']).split('/')[-1]
            tabla['pre_b_2h_download_c2'][i] = download_path+fileid
            #"/".join(str(asd.search_lascoc2[asd.indices_descarga]['fileid']).split('/')[1:])
   #         breakpoint()
        else:
            tabla['pre_b_2h_download_c2'][i] = 'No data'
    else:
        tabla['pre_b_2h_download_c2'][i] = '*'

#    breakpoint()
breakpoint()
tabla.to_csv('/gehme/projects/2020_gcs_with_ml/repo_diego/2020_gcs_with_ml/nn_training/corona_background/Lista_Final_CMEs_downloads_lascoc2.csv', sep='\t', index=False)
breakpoint()
