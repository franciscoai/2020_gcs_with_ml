from descargar_imagenes_clases import cor1_downloader
import pandas as pd
from datetime import datetime, timedelta


tabla = pd.read_csv("/gehme/projects/2020_gcs_with_ml/repo_diego/2020_gcs_with_ml/nn_training/corona_background/catalogues/Lista_Final_CMEs.csv", sep='\t', engine='python',encoding="utf-8", header=0)
tabla['pre_a_3h_download_cor1'] = ''
tabla['pre_b_3h_download_cor1'] = ''
tabla['preevento_a_3h'] = '' #debo crearlas
tabla['preevento_b_3h'] = ''
tabla['pre_a_2h_download_cor1'] = ''
tabla['pre_b_2h_download_cor1'] = ''
pre_suffix = '/gehme/data/stereo/'

for i in range(len(tabla)):
    if i == 15: breakpoint()
    print("chequeando elemento Numero {}".format(i))
    pre_even_a_1h = tabla['preevento_a_1h'][i]
    if (pre_even_a_1h != '*' and pre_even_a_1h != 'NaT'): 
        dt = datetime.strptime(pre_even_a_1h, '%Y-%m-%d %H:%M:%S') - timedelta(hours=2) #=> pre-evento -3hs
        tabla['preevento_a_3h'][i] = dt.strftime('%Y/%m/%d %H:%M:%S')
        #breakpoint()
        dt_ini = dt - timedelta(minutes=55)
        ini = dt_ini.strftime('%Y/%m/%d %H:%M:%S')
        dt_fin = dt + timedelta(minutes=55)
        fin = dt_fin.strftime('%Y/%m/%d %H:%M:%S')
        
        asd = cor1_downloader(start_time=ini,end_time=fin,size=2,image_type='seq',nave='STEREO_A',nivel='s4c')
        asd.search()
        if len(asd.search_cor1) ==0:#a partir del 2009 aprox, practicamente todas las imagenes son size=0.5Mb 
            asd.size=0.5
            asd.search()
        if len(asd.search_cor1) >= 1:
            start_times = asd.search_cor1['Start Time']
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)

            
            if len(asd.search_cor1) ==3:#asumimos trios muy cercanos entonces esto es True.
                asd.indices_descarga=[0,1,2]
                asd.download()
                suffix = "/".join(str(asd.search_cor1[asd.indices_descarga[0]]['fileid']).split('/')[:-1])+"/"
                list_downloaded_fileid=[pre_suffix+suffix]
                for j in range(3):
                    list_downloaded_fileid.append(str(asd.search_cor1[asd.indices_descarga[j]]['fileid']).split('/')[-1])
                tabla['pre_a_3h_download_cor1'][i] = list_downloaded_fileid
            if len(asd.search_cor1) < 3:
                print("len of asd.search_cor1 <3 shoud not occur. WTF.")
                tabla['pre_a_3h_download_cor1'][i] = "REVISAR"
            if len(asd.search_cor1) > 3:
                
                if indice_fecha_cercana == 0:
                    trio = lista_strings_times2[0:3]
                if indice_fecha_cercana == len(asd.search_cor1)-1:
                    trio = lista_strings_times2[-3:]
                if indice_fecha_cercana != 0 and indice_fecha_cercana != len(asd.search_cor1)-1:
                    for ii in range(3):
                        trio = lista_strings_times2[indice_fecha_cercana-ii:indice_fecha_cercana+3-ii]
                        resultado = [trio[w] - trio[w+1] for w in range(len(trio)-1)]
                        if abs(resultado[0]-resultado[1])<= timedelta(seconds=30):
                            print("vamos a descargar")
                            asd.indices_descarga = [lista_strings_times2.index(elemento) for elemento in trio]

                    suffix = "/".join(str(asd.search_cor1[asd.indices_descarga[0]]['fileid']).split('/')[:-1])+"/"
                    list_downloaded_fileid=[pre_suffix+suffix]
                    for j in range(3):
                        list_downloaded_fileid.append(str(asd.search_cor1[asd.indices_descarga[j]]['fileid']).split('/')[-1])

                    tabla['pre_a_3h_download_cor1'][i] = list_downloaded_fileid
                    #"/".join(str(asd.search_cor1[asd.indices_descarga]['fileid']).split('/')[1:])
                
                asd.download()
            #breakpoint()
            
            
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

        asd = cor1_downloader(start_time=ini,end_time=fin,size=2,image_type='seq',nave='STEREO_A',nivel='s4c')
        asd.search()
        if len(asd.search_cor1) ==0:#a partir del 2009 aprox, practicamente todas las imagenes son size=0.5Mb 
            asd.size=0.5
            asd.search()
        if len(asd.search_cor1) >= 1:
            start_times = asd.search_cor1['Start Time']
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)

            if len(asd.search_cor1) ==3:
                asd.indices_descarga=[0,1,2]
                asd.download()
                suffix = "/".join(str(asd.search_cor1[asd.indices_descarga[0]]['fileid']).split('/')[:-1])+"/"
                list_downloaded_fileid=[pre_suffix+suffix]
                for j in range(3):
                    list_downloaded_fileid.append(str(asd.search_cor1[asd.indices_descarga[j]]['fileid']).split('/')[-1])
                tabla['pre_a_2h_download_cor1'][i] = list_downloaded_fileid
            if len(asd.search_cor1) < 3:
                print("len of asd.search_cor1 <3 shoud not occur. WTF.")
                tabla['pre_a_2h_download_cor1'][i] = "REVISAR"
            if len(asd.search_cor1) > 3:
                
                if indice_fecha_cercana == 0:
                    trio = lista_strings_times2[0:3]
                if indice_fecha_cercana == len(asd.search_cor1)-1:
                    trio = lista_strings_times2[-3:]
                if indice_fecha_cercana != 0 and indice_fecha_cercana != len(asd.search_cor1)-1:
                    for ii in range(3):
                        trio = lista_strings_times2[indice_fecha_cercana-ii:indice_fecha_cercana+3-ii]
                        resultado = [trio[w] - trio[w+1] for w in range(len(trio)-1)]
                        if abs(resultado[0]-resultado[1])<= timedelta(seconds=30):
                            print("vamos a descargar")
                            asd.indices_descarga = [lista_strings_times2.index(elemento) for elemento in trio]
                    suffix = "/".join(str(asd.search_cor1[asd.indices_descarga[0]]['fileid']).split('/')[:-1])+"/"
                    list_downloaded_fileid=[pre_suffix+suffix]
                    for j in range(3):
                        list_downloaded_fileid.append(str(asd.search_cor1[asd.indices_descarga[j]]['fileid']).split('/')[-1])
                    tabla['pre_a_2h_download_cor1'][i] = list_downloaded_fileid
                asd.download()
            #breakpoint()
            
        else:
            tabla['pre_a_2h_download_cor1'][i] = 'No data'
    else:
        tabla['pre_a_2h_download_cor1'][i] = '*'
    


    pre_even_b_1h = tabla['preevento_b_1h'][i]
    if (pre_even_b_1h != '*' and pre_even_b_1h != 'NaT'):
        dt = datetime.strptime(pre_even_b_1h, '%Y-%m-%d %H:%M:%S') - timedelta(hours=2) #=> pre-evento -3hs
        tabla['preevento_b_3h'][i] = dt.strftime('%Y/%m/%d %H:%M:%S')
        #breakpoint()
        dt_ini = dt - timedelta(minutes=55)
        ini = dt_ini.strftime('%Y/%m/%d %H:%M:%S')
        dt_fin = dt + timedelta(minutes=55)
        fin = dt_fin.strftime('%Y/%m/%d %H:%M:%S')

        asd = cor1_downloader(start_time=ini,end_time=fin,size=2,image_type='seq',nave='STEREO_B',nivel='s4c')
        asd.search()
        if len(asd.search_cor1) ==0:#a partir del 2009 aprox, practicamente todas las imagenes son size=0.5Mb 
            asd.size=0.5
            asd.search()        
        if len(asd.search_cor1) >= 1:
            start_times = asd.search_cor1['Start Time']
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)
            if len(asd.search_cor1) ==3:
                asd.indices_descarga=[0,1,2]
                asd.download()
                suffix = "/".join(str(asd.search_cor1[asd.indices_descarga[0]]['fileid']).split('/')[:-1])+"/"
                list_downloaded_fileid=[pre_suffix+suffix]
                for j in range(3):
                    list_downloaded_fileid.append(str(asd.search_cor1[asd.indices_descarga[j]]['fileid']).split('/')[-1])
                tabla['pre_b_3h_download_cor1'][i] = list_downloaded_fileid
                
            if len(asd.search_cor1) < 3:
                print("len of asd.search_cor1 <3 shoud not occur. WTF.")
                tabla['pre_b_3h_download_cor1'][i] = "REVISAR"
            if len(asd.search_cor1) > 3:
                
                if indice_fecha_cercana == 0:
                    trio = lista_strings_times2[0:3]
                if indice_fecha_cercana == len(asd.search_cor1)-1:
                    trio = lista_strings_times2[-3:]
                if indice_fecha_cercana != 0 and indice_fecha_cercana != len(asd.search_cor1)-1:
                    for ii in range(3):
                        trio = lista_strings_times2[indice_fecha_cercana-ii:indice_fecha_cercana+3-ii]
                        resultado = [trio[w] - trio[w+1] for w in range(len(trio)-1)]
                        if abs(resultado[0]-resultado[1])<= timedelta(seconds=30):
                            print("vamos a descargar")
                            asd.indices_descarga = [lista_strings_times2.index(elemento) for elemento in trio]
                    suffix = "/".join(str(asd.search_cor1[asd.indices_descarga[0]]['fileid']).split('/')[:-1])+"/"
                    list_downloaded_fileid=[pre_suffix+suffix]
                    for j in range(3):
                        list_downloaded_fileid.append(str(asd.search_cor1[asd.indices_descarga[j]]['fileid']).split('/')[-1])
                    tabla['pre_b_3h_download_cor1'][i] = list_downloaded_fileid
                asd.download()
            #breakpoint()
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

        asd = cor1_downloader(start_time=ini,end_time=fin,size=2,image_type='seq',nave='STEREO_B',nivel='s4c')
        asd.search()
        if len(asd.search_cor1) ==0:#a partir del 2009 aprox, practicamente todas las imagenes son size=0.5Mb 
            asd.size=0.5
            asd.search()        
        if len(asd.search_cor1) >= 1:
            start_times = asd.search_cor1['Start Time']
            lista_strings_times = [t.iso for t in start_times]
            lista_strings_times2 = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in lista_strings_times]
            fecha_cercana = min(lista_strings_times2, key=lambda x: abs(x - dt))
            indice_fecha_cercana = lista_strings_times2.index(fecha_cercana)
            if len(asd.search_cor1) ==3:
                asd.indices_descarga=[0,1,2]
                asd.download()
                suffix = "/".join(str(asd.search_cor1[asd.indices_descarga[0]]['fileid']).split('/')[:-1])+"/"
                list_downloaded_fileid=[pre_suffix+suffix]
                for j in range(3):
                    list_downloaded_fileid.append(str(asd.search_cor1[asd.indices_descarga[j]]['fileid']).split('/')[-1])
                tabla['pre_b_2h_download_cor1'][i] = list_downloaded_fileid
                
            if len(asd.search_cor1) < 3:
                print("len of asd.search_cor1 <3 shoud not occur. WTF.")
                tabla['pre_b_2h_download_cor1'][i] = "REVISAR"
            if len(asd.search_cor1) > 3:
                
                if indice_fecha_cercana == 0:
                    trio = lista_strings_times2[0:3]
                if indice_fecha_cercana == len(asd.search_cor1)-1:
                    trio = lista_strings_times2[-3:]
                if indice_fecha_cercana != 0 and indice_fecha_cercana != len(asd.search_cor1)-1:
                    for ii in range(3):
                        trio = lista_strings_times2[indice_fecha_cercana-ii:indice_fecha_cercana+3-ii]
                        resultado = [trio[w] - trio[w+1] for w in range(len(trio)-1)]
                        if abs(resultado[0]-resultado[1])<= timedelta(seconds=30):
                            print("vamos a descargar")
                            asd.indices_descarga = [lista_strings_times2.index(elemento) for elemento in trio]
                    suffix = "/".join(str(asd.search_cor1[asd.indices_descarga[0]]['fileid']).split('/')[:-1])+"/"
                    list_downloaded_fileid=[pre_suffix+suffix]
                    for j in range(3):
                        list_downloaded_fileid.append(str(asd.search_cor1[asd.indices_descarga[j]]['fileid']).split('/')[-1])
                    tabla['pre_b_2h_download_cor1'][i] = list_downloaded_fileid
                    #tabla['pre_b_2h_download_cor1'][i] = "/".join(str(asd.search_cor1[asd.indices_descarga]['fileid']).split('/')[1:])
                asd.download()
#            breakpoint()
            
        else:
            tabla['pre_b_2h_download_cor1'][i] = 'No data'
    else:
        tabla['pre_b_2h_download_cor1'][i] = '*'
    #breakpoint()

breakpoint()
tabla.to_csv('/gehme/projects/2020_gcs_with_ml/repo_diego/2020_gcs_with_ml/nn_training/corona_background/Lista_Final_CMEs_downloads_cor1.csv', sep='\t', index=False)
breakpoint()
