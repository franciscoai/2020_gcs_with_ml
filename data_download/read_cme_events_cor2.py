from descargar_imagenes_clases import cor2_downloader
import pandas as pd
from datetime import datetime, timedelta


tabla = pd.read_csv("/gehme/projects/2020_gcs_with_ml/repo_diego/2020_gcs_with_ml/nn_training/corona_background/catalogues/Lista_Final_CMEs.csv", sep='\t', engine='python',encoding="utf-8", header=0)
tabla['pre_a_1h_download'] = ''
tabla['pre_b_1h_download'] = ''
tabla['pre_a_2h_download'] = ''
tabla['pre_b_2h_download'] = ''
suffix = "/gehme/data/stereo/secchi/"

for i in range(len(tabla)):
    print("chequeando elemento Numero {}".format(i))
    pre_even_a_1h = tabla['preevento_a_1h'][i]
#    if i <= 15:
#        continue
    if (pre_even_a_1h != '*' and pre_even_a_1h != 'NaT'): 
        dt = datetime.strptime(pre_even_a_1h, '%Y-%m-%d %H:%M:%S')
        dt_ini = dt - timedelta(minutes=40)
        ini = dt_ini.strftime('%Y/%m/%d %H:%M:%S')
        dt_fin = dt + timedelta(minutes=40)
        fin = dt_fin.strftime('%Y/%m/%d %H:%M:%S')
        
        asd = cor2_downloader(start_time=ini,end_time=fin,nave='STEREO_A',nivel='double',image_type='img',size=2)
        asd.search()
        if len(asd.search_cor2) >= 1:
            start_times = asd.search_cor2['Start Time']#[0:-1]
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)
            asd.indices_descarga = [indice_fecha_cercana] #necesito que sea una lista para que el download itere sobre ella
#            breakpoint()
            asd.download()
            #tabla['pre_a_1h_download'][i] = asd.search_cor2[asd.indices_descarga]['fileid']
            tabla['pre_a_1h_download'][i] = suffix+"/".join(str(asd.search_cor2[asd.indices_descarga]['fileid']).split('/')[1:])
        else:
            tabla['pre_a_1h_download'][i] = 'No img/double data'
    else:
        tabla['pre_a_1h_download'][i] = '*'

    pre_even_a_2h = tabla['preevento_a_2h'][i]
    if (pre_even_a_2h != '*' and pre_even_a_2h != 'NaT'):
        dt = datetime.strptime(pre_even_a_2h, '%Y-%m-%d %H:%M:%S')
        dt_ini = dt - timedelta(minutes=40)
        ini = dt_ini.strftime('%Y/%m/%d %H:%M:%S')
        dt_fin = dt + timedelta(minutes=40)
        fin = dt_fin.strftime('%Y/%m/%d %H:%M:%S')

        asd = cor2_downloader(start_time=ini,end_time=fin,nave='STEREO_A',nivel='double',image_type='img',size=2)
        asd.search()
        
        if len(asd.search_cor2) >= 1:
            start_times = asd.search_cor2['Start Time']#[0:-1]
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)
            asd.indices_descarga = [indice_fecha_cercana]
        
            asd.download()
            #tabla['pre_a_2h_download'][i] = asd.search_cor2[asd.indices_descarga]['fileid']
            tabla['pre_a_2h_download'][i] = suffix+"/".join(str(asd.search_cor2[asd.indices_descarga]['fileid']).split('/')[1:])
        else:
            tabla['pre_a_2h_download'][i] = 'No img/double data'
    else:
        tabla['pre_a_2h_download'][i] = '*'

    pre_even_b_1h = tabla['preevento_b_1h'][i]
    if (pre_even_b_1h != '*' and pre_even_b_1h != 'NaT'):
        dt = datetime.strptime(pre_even_b_1h, '%Y-%m-%d %H:%M:%S')
        dt_ini = dt - timedelta(minutes=40)
        ini = dt_ini.strftime('%Y/%m/%d %H:%M:%S')
        dt_fin = dt + timedelta(minutes=40)
        fin = dt_fin.strftime('%Y/%m/%d %H:%M:%S')

        asd = cor2_downloader(start_time=ini,end_time=fin,nave='STEREO_B',nivel='double',image_type='img',size=2)
        asd.search()
        #breakpoint()
        if len(asd.search_cor2) >= 1:
            start_times = asd.search_cor2['Start Time']#[0:-1]
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)
            asd.indices_descarga = [indice_fecha_cercana]
        
            asd.download()
            #tabla['pre_b_1h_download'][i] = asd.search_cor2[asd.indices_descarga]['fileid']
            tabla['pre_b_1h_download'][i] = suffix+"/".join(str(asd.search_cor2[asd.indices_descarga]['fileid']).split('/')[1:])
#            breakpoint()
        else:
            tabla['pre_b_1h_download'][i] = 'No img/double data'
    else:
        tabla['pre_b_1h_download'][i] = '*'

    pre_even_b_2h = tabla['preevento_b_2h'][i]
    if (pre_even_b_2h != '*' and pre_even_b_2h != 'NaT'):
        dt = datetime.strptime(pre_even_b_2h, '%Y-%m-%d %H:%M:%S')
        dt_ini = dt - timedelta(minutes=40)
        ini = dt_ini.strftime('%Y/%m/%d %H:%M:%S')
        dt_fin = dt + timedelta(minutes=40)
        fin = dt_fin.strftime('%Y/%m/%d %H:%M:%S')

        asd = cor2_downloader(start_time=ini,end_time=fin,nave='STEREO_B',nivel='double',image_type='img',size=2)
        asd.search()
        if len(asd.search_cor2) >= 1:
            start_times = asd.search_cor2['Start Time']#[0:-1]
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)
            asd.indices_descarga = [indice_fecha_cercana]
        
            asd.download()
            #tabla['pre_b_2h_download'][i] = asd.search_cor2[asd.indices_descarga]['fileid']
            tabla['pre_b_2h_download'][i] = suffix+"/".join(str(asd.search_cor2[asd.indices_descarga]['fileid']).split('/')[1:])
        else:
            tabla['pre_b_2h_download'][i] = 'No img/double data'
    else:
        tabla['pre_b_2h_download'][i] = '*'
        
breakpoint()
tabla.to_csv('/gehme/projects/2020_gcs_with_ml/repo_diego/2020_gcs_with_ml/nn_training/corona_background/Lista_Final_CMEs_downloads_cor2_v2.csv', sep='\t', index=False)
breakpoint()
