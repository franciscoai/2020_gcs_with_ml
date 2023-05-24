from descargar_imagenes_clases import cor1_downloader
import pandas as pd
from datetime import datetime, timedelta


tabla = pd.read_csv("/gehme/projects/2020_gcs_with_ml/repo_diego/2020_gcs_with_ml/nn_training/corona_background/Lista_Final_CMEs.txt", sep='\t', engine='python',encoding="utf-8", header=0)
tabla['pre_a_3h_download_cor1'] = ''
tabla['pre_b_3h_download_cor1'] = ''
tabla['preevento_a_3h'] = '' #debo crearlas
tabla['preevento_b_3h'] = ''
tabla['pre_a_2h_download_cor1'] = ''
tabla['pre_b_2h_download_cor1'] = ''

for i in range(len(tabla)):
    print("chequeando elemento Numero {}".format(i))
    pre_even_a_1h = tabla['preevento_a_1h'][i]
    #if i <= 4:
    #    continue
    if (pre_even_a_1h != '*' and pre_even_a_1h != 'NaT'): 
        dt = datetime.strptime(pre_even_a_1h, '%Y-%m-%d %H:%M:%S') - timedelta(hours=2) #=> pre-evento -3hs
        tabla['preevento_a_3h'][i] = dt.strftime('%Y/%m/%d %H:%M:%S')
        dt_ini = dt - timedelta(minutes=55)
        ini = dt_ini.strftime('%Y/%m/%d %H:%M:%S')
        dt_fin = dt + timedelta(minutes=55)
        fin = dt_fin.strftime('%Y/%m/%d %H:%M:%S')
        breakpoint()
        asd = cor1_downloader(start_time=ini,end_time=fin,size=2,image_type='seq',nave='STEREO_A')
        #breakpoint()
        asd.search()
        if len(asd.search_cor1) >= 1:
            start_times = asd.search_cor1['Start Time']
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)
            asd.indices_descarga = [indice_fecha_cercana] #necesito que sea una lista para que el download itere sobre ella

            resultado  = [lista_strings_times2[i] - lista_strings_times2[i+1] for i in range(len(lista_strings_times2) - 1)]
            condicion = lambda a,b: a == b
            seleccionados = []
            for i in range(len(resultado) - 1):
                if condicion(resultado[i], resultado[i+1]):
                    seleccionados.append(resultado[i])
                    seleccionados.append(resultado[i+1])
            
            breakpoint()
            asd.download()
            tabla['pre_a_3h_download_cor1'][i] = asd.search_cor1[asd.indices_descarga]['fileid']
#            breakpoint()
        else:
            tabla['pre_a_3h_download_cor1'][i] = 'No data'
    else:
        tabla['pre_a_3h_download_cor1'][i] = '*'

    pre_even_a_2h = tabla['preevento_a_2h'][i]
    if (pre_even_a_2h != '*' and pre_even_a_2h != 'NaT'):
        dt = datetime.strptime(pre_even_a_2h, '%Y-%m-%d %H:%M:%S')
        dt_ini = dt - timedelta(minutes=55)
        ini = dt_ini.strftime('%Y/%m/%d %H:%M:%S')
        dt_fin = dt + timedelta(minutes=55)
        fin = dt_fin.strftime('%Y/%m/%d %H:%M:%S')

        asd = cor1_downloader(start_time=ini,end_time=fin,size=2,image_type='seq',nave='STEREO_A')
        asd.search()
        
        if len(asd.search_cor1) >= 1:
            start_times = asd.search_cor1['Start Time']
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)
            asd.indices_descarga = [indice_fecha_cercana]
        
            asd.download()
            tabla['pre_a_2h_download_cor1'][i] = asd.search_cor1[asd.indices_descarga]['fileid']
 #           breakpoint()
        else:
            tabla['pre_a_2h_download_cor1'][i] = 'No data'
    else:
        tabla['pre_a_2h_download_cor1'][i] = '*'

    pre_even_b_1h = tabla['preevento_b_1h'][i]
    if (pre_even_b_1h != '*' and pre_even_b_1h != 'NaT'):
        dt = datetime.strptime(pre_even_b_1h, '%Y-%m-%d %H:%M:%S') - timedelta(hours=2) #=> pre-evento -3hs
        tabla['preevento_b_3h'][i] = dt.strftime('%Y/%m/%d %H:%M:%S')
        dt_ini = dt - timedelta(minutes=55)
        ini = dt_ini.strftime('%Y/%m/%d %H:%M:%S')
        dt_fin = dt + timedelta(minutes=55)
        fin = dt_fin.strftime('%Y/%m/%d %H:%M:%S')

        asd = cor1_downloader(start_time=ini,end_time=fin,size=2,image_type='seq',nave='STEREO_B')
        asd.search()
        #breakpoint()
        if len(asd.search_cor1) >= 1:
            start_times = asd.search_cor1['Start Time']
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)
            asd.indices_descarga = [indice_fecha_cercana]
        
            asd.download()
            tabla['pre_b_3h_download_cor1'][i] = asd.search_cor1[asd.indices_descarga]['fileid']
  #          breakpoint()
        else:
            tabla['pre_b_3h_download_cor1'][i] = 'No data'
    else:
        tabla['pre_b_3h_download_cor1'][i] = '*'

    pre_even_b_2h = tabla['preevento_b_2h'][i]
    if (pre_even_b_2h != '*' and pre_even_b_2h != 'NaT'):
        dt = datetime.strptime(pre_even_b_2h, '%Y-%m-%d %H:%M:%S')
        dt_ini = dt - timedelta(minutes=55)
        ini = dt_ini.strftime('%Y/%m/%d %H:%M:%S')
        dt_fin = dt + timedelta(minutes=55)
        fin = dt_fin.strftime('%Y/%m/%d %H:%M:%S')

        asd = cor1_downloader(start_time=ini,end_time=fin,size=2,image_type='seq',nave='STEREO_B')
        asd.search()
        if len(asd.search_cor1) >= 1:
            start_times = asd.search_cor1['Start Time']
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)
            asd.indices_descarga = [indice_fecha_cercana]
        
            asd.download()
            tabla['pre_b_2h_download_cor1'][i] = asd.search_cor1[asd.indices_descarga]['fileid']
   #         breakpoint()
        else:
            tabla['pre_b_2h_download_cor1'][i] = 'No data'
    else:
        tabla['pre_b_2h_download_cor1'][i] = '*'
        
breakpoint()
tabla.to_csv('/gehme/projects/2020_gcs_with_ml/repo_diego/2020_gcs_with_ml/nn_training/corona_background/Lista_Final_CMEs_downloads_lascoc2.txt', sep='\t', index=False)
breakpoint()
