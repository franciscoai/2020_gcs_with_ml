
#import sunpy.map
#import datetime as dt
import astropy.units as u
from sunpy.net import Fido, attrs as a
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import pdb
import numpy as np
import requests

class lascoc2_downloader:
    """
    Definicion de la clase lasco C2 Downloader
    Mencionar ejemplo.
    """

    def __init__(self, start_time, end_time,nivel='',size='',nrl_download=''):
        try:
            self.start_time = start_time
            self.end_time = end_time
            self.instrumento = a.Instrument.lasco
            self.detector = a.Detector.c2
            self.nave  = 'SoHO'
            self.dir_descarga = '/gehme/data/soho/lasco/'
            #self.dir_descarga = '/data_local/GCS/gcs/Imagenes/'
            self.nivel = nivel #puede ser level_05
            self.indices_descarga = '' #debe ser una lista
            self.size = size #puede ser 1 o 2
            self.nrl_download = nrl_download

        except TypeError:
            print("Be sure to add start_time, end_time, ship name, level/type of image when creating of object of this class.")
            raise
        except:
            print("WTF")

    def search(self):
        """
        Definir metodo search
        """
        search_lascoc2 = Fido.search(a.Time(self.start_time,self.end_time),self.instrumento,self.detector)
        if not search_lascoc2:
            print("No provider! No images available.")
            #global provider
            #provider = 0
            self.search_lascoc2 = search_lascoc2 #lista vacía
        else:
            self.search_lascoc2 = search_lascoc2['vso']
        if self.nivel != '':#asumo nivel definido correctamente
            self.filtro()

    def filtro(self):
        """
        Definicion del metodo filtro
        """
        match self.nivel:
            case 'que_son':
                string_search = "level_05"#"22"
                #size_True_False = (self.search_lascoc2['Size'] < (1.4*u.Mibyte))

            case 'level_05':
                string_search = "level_05"#"22"
                #size_True_False = (self.search_lascoc2['Size'] < (2.5*u.Mibyte)) & (self.search_lascoc2['Size'] > (1.5*u.Mibyte))
                
            case 'level_1':
                string_search = "25"
                #size_True_False = np.ones(search_size, dtype=bool).tolist()

            case '':
                string_search = ""

        lista_filtro = [] 
        for j in range(len(self.search_lascoc2)):
            if (string_search in self.search_lascoc2[j]['fileid']):
                lista_filtro.append(j)
        if not lista_filtro: 
            print('Filtered images')
        else:
            self.search_lascoc2 = self.search_lascoc2[lista_filtro]
        #print(lista_filtro)
        if self.size and lista_filtro: 
            lista_filtro2 = []
            for j in range(len(self.search_lascoc2)):
                if int(round(self.search_lascoc2[j]['Size'].value)) == self.size:
                    lista_filtro2.append(j)
            
            #resultado = [True if int(round(self.search_lascoc2['Size'][i].value)) == self.size else False for i in range(len(self.search_lascoc2))]
            #self.search_lascoc2[resultado]
            if not lista_filtro2:
                print('Filtered images')
            else:
                self.search_lascoc2 = self.search_lascoc2[lista_filtro2]
            #print(lista_filtro2)
        
    def display(self):
        """
        Definicion del metodo display
        """
        lista_dates=[]#revisar scope, esto modificaria una lista fuera de esta definicion?
        #lista_cant_images = []
        #lista_colores = []
        lista_color_azul = []
        lista_color_rojo = []
        lista_color_verde = []
        lista_cant_images_azul = []
        lista_cant_images_rojo = []
        lista_cant_images_verde = []
        search_size = len(self.search_lascoc2)
        for i in range(search_size):
            lista_dates.append(datetime.fromisoformat(str(self.search_lascoc2[i][0])))
            string_search_lvl05 = "level_05"
            string_search_lvl10 = "25"
            string_search_queesesto = "level_05"
            if string_search_lvl05    in self.search_lascoc2[i]['fileid'] and int(round(self.search_lascoc2[i]['Size'].value)) == 2: 
                lista_color_azul.append(datetime.fromisoformat(str(self.search_lascoc2[i][0])))
                lista_cant_images_azul.append(i)
            if string_search_lvl10 in self.search_lascoc2[i]['fileid']: 
                lista_color_rojo.append(datetime.fromisoformat(str(self.search_lascoc2[i][0])))
                lista_cant_images_rojo.append(i)
            if string_search_queesesto in self.search_lascoc2[i]['fileid'] and int(round(self.search_lascoc2[i]['Size'].value)) == 1: 
                lista_color_verde.append(datetime.fromisoformat(str(self.search_lascoc2[i][0])))
                lista_cant_images_verde.append(i)

        #labels = ['lvl_0.5','lvl_1.0','que son estas?']
        #fig, ax = plt.subplots()
        fig = plt.figure()
        ax  = fig.subplots()
        #ax.plot(lista_dates, lista_cant_images,c=lista_colores)
        ax.plot(lista_color_rojo,lista_cant_images_rojo,'ro',label='level 1.0')
        ax.plot(lista_color_azul,lista_cant_images_azul,'bo',label='level 0.5 2M')
        ax.plot(lista_color_verde,lista_cant_images_verde,'go',label='level 0.5 1M')
        ax.set_xlabel('Dates')
        ax.set_ylabel('Images')
        ax.set_title('Available images per time')
        fig.autofmt_xdate()
        ax.legend()
        ax.grid(True)
        plt.show()


    def nrl_navy_download(self, file_id, download_path_nrl):
        url = "https://lasco-www.nrl.navy.mil/lz/level_1/"#101124/c2/25352163.fts.gz"  
        
        old_folder = file_id.split('/')[8]
        file_name_old = file_id.split('/')[-1]
        list_file = list(file_name_old)
        if list_file[1]=='2':
            list_file[1]='5'
        file_name_download = "".join(list_file)+".gz" # Nombre con el que se buscará y guardará el archivo 

        # Realizar la solicitud GET para obtener el contenido del archivo
        respuesta = requests.get(url+'/'+old_folder+'/c2/'+file_name_download)
        path_file = os.path.join(download_path_nrl,file_name_download)

        if os.path.exists(path_file):
            print(f"El archivo comprimido {path_file} ya existe, NO se procede a descargar.")
            return
        
        # Verificar si la solicitud fue exitosa (código de estado 200)
        
        if respuesta.status_code == 200:
            with open(path_file, "wb") as archivo:
                archivo.write(respuesta.content)
            print("El archivo se ha descargado exitosamente.")
            if os.path.exists(path_file[:-3]):
                print(f"El archivo a descomprimir {path_file[:-3]} ya existe, NO se procede a la descompresión del mismo.")
                return
            comando = f"gzip -d {path_file}"
            os.system(comando)
            print(f"unziping {path_file}")
            breakpoint()
            os.system(f"rm {path_file}")
            print(f"removing {path_file}")
        else:
            print("No se pudo descargar el archivo. Código de estado:", respuesta.status_code)
            print(f"Es posible que la imagen no se encuentre en {url+'/'+old_folder+'/c2/'+file_name_download}, o bien esté experimentando problemas de conectividad.")
        #return "".join(list_file)



    def download(self, download_path=None):
        """
        Definir metodo download
        """
        carpetas_creadas = []
        if getattr(self,'indices_descarga') == '': 
            cantidad = len(self.search_lascoc2)
            rango_descargas = range(cantidad)
        if getattr(self,'indices_descarga') != '':  #'indice_descarga' debe ser una lista de enteros contenidos en [0,len(self.search_cor2)]
            rango_descargas = self.indices_descarga

        for w in rango_descargas:
            folder_year_month_day = self.search_lascoc2[w]['fileid'].split('/')[8]
            #full_download_path = download_path+'/'+folder_year_month_day+'/'
            if not download_path:
                #download_path = self.dir_descarga+"/".join(self.search_lascoc2['fileid'][w].split('/')[0:-1])+'/'
                folder = self.search_lascoc2['fileid'][w].split('/')[-3]
                if int(folder[0:2]) < 50:
                    suffix = str(2000+int(folder[0:2]))
                else:
                    suffix = str(1900+int(folder[0:2]))                
                folder_full = suffix+folder[2:]+"/"
                if self.nrl_download:
                    download_path = self.dir_descarga+"level_1/c2/"+folder_full
                else:    
                    download_path = self.dir_descarga+"level_05/c2/"+folder_full

            if not os.path.exists(download_path):
                os.makedirs(download_path)
                print(f"Se ha creado el directorio {download_path}")
                carpetas_creadas.append(download_path)
            else:   
                print(f"El directorio {download_path} ya existe")

            if self.nrl_download:
                breakpoint()
                downloaded_files = self.nrl_navy_download(self.search_lascoc2[w]['fileid'], download_path)
            else:   
                ofile = download_path+self.search_lascoc2[w]['fileid'].split('/')[-1]
                if not os.path.isfile(ofile):#Si archivo no descargado entonces que descargue.
                    downloaded_files = Fido.fetch(self.search_lascoc2[w],path=download_path, max_conn=5, progress=True) 
                    os.system('chgrp gehme {}'.format(ofile[0:-9]+'*'))
                    os.system('chmod 775 {}'.format(ofile[0:-9]+'*'))

        print(f'Archivos descargados en: {download_path}')






class lascoc3_downloader:
    def __init__(self, start_time, end_time, wavelength):
        self.start_time = start_time
        self.end_time = end_time
        self.wavelength = wavelength





class cor1_downloader:
    """
    Descripcion de la clase cor1 downloader
    Mencionar ejemplo.
    self.nivel = double, normal, sequential
    self.image_type = admite "img" o "seq". img puede ser self.nivel norm, seq; seq puede ser double o normal. 
    """
    def __init__(self, start_time, end_time,nave,nivel='',image_type='',size=''):
        try:
            self.start_time = start_time
            self.end_time = end_time
            self.instrumento = a.Instrument.secchi
            self.detector = a.Detector.cor1
            self.nave  = nave
            self.dir_descarga = '/gehme/data/stereo/'
            #self.dir_descarga = '/data_local/GCS/gcs/Imagenes/'
            self.nivel = nivel #'s4c' es la calibracion sugerida que viene de a tríos
            self.indices_descarga = ''
            self.image_type = image_type #seq usualmente
            self.size = size #2M usualmente
        except TypeError:
            print("Be sure to add start_time, end_time, ship name, level/type of image when creating of object of this class.")
            raise
        except:
            print("WTF")

    def search(self):
        """
        Definicion del metodo search
        """
        search_cor1 = Fido.search(a.Time(self.start_time,self.end_time),self.instrumento,self.detector)
        if not search_cor1:
            print("No provider! No images available.")
            self.search_cor1 = search_cor1 #lista vacía
        else:
            match self.nave:
                case 'STEREO_A':
                    search_cor1_ = search_cor1['vso'][search_cor1['vso']['Source'] == 'STEREO_A'].copy()
                case 'STEREO_B':
                    search_cor1_ = search_cor1['vso'][search_cor1['vso']['Source'] == 'STEREO_B'].copy()        
            self.search_cor1 = search_cor1_
            if self.nivel != '' or self.size!='':#asumo nivel definido correctamente
                self.filtro()

    def display(self):
        """
        Definicion del metodo display
        """
        lista_dates=[]
        search_size = len(self.search_cor1)
        lista_color_verde = []
        lista_cant_images_verde = []
        for i in range(search_size):
            lista_dates.append(datetime.fromisoformat(str(self.search_cor1[i][0])))
            string_search_sequence = "s4c"
            if string_search_sequence in self.search_cor1[i]['fileid']:
                lista_color_verde.append(datetime.fromisoformat(str(self.search_cor1[i][0])))
                lista_cant_images_verde.append(i)

        fig = plt.figure()
        ax  = fig.subplots()
        ax.plot(lista_color_verde,lista_cant_images_verde,'go',label='sequence')
        ax.set_xlabel('Dates')
        ax.set_ylabel('Images')
        ax.set_title('Available images per time, {}'.format(self.image_type))
        fig.autofmt_xdate()
        ax.legend()
        ax.grid(True)
        plt.show()

    def filtro(self):
        """
        Definicion del metodo filtro
        """
        match self.nivel:
            case 's4c':
                string_search_nivel = "s4c"
            case 's5c':
                string_search_nivel = "s5c"
            case '':
                string_search_nivel = ""
        match self.image_type:
            case 'seq':
                string_search_type = "seq"
            case '':
                string_search_type = ""

        lista_filtro = [] 
        for j in range(len(self.search_cor1)):
            if (string_search_nivel in self.search_cor1[j]['fileid'] and string_search_type in self.search_cor1[j]['fileid']): lista_filtro.append(j)
        self.search_cor1 = self.search_cor1[lista_filtro]
        if not lista_filtro: print('Filtered images') 

        if self.size and lista_filtro: 
            lista_filtro2 = []
            for j in range(len(self.search_cor1)):
                #if self.size == int(round(self.search_cor1[j]['Size'].value)):
                if self.size == round((self.search_cor1[j]['Size'].value) * 2) / 2:
                    lista_filtro2.append(j)
            self.search_cor1 = self.search_cor1[lista_filtro2]
            if not lista_filtro2: print('Filtered images')


    def download(self, download_path=None):
        """
        Definicion del metodo download.
        """
        carpetas_creadas = []
        if getattr(self,'indices_descarga') == '': 
            cantidad = len(self.search_cor1)
            rango_descargas = range(cantidad)
        if getattr(self,'indices_descarga') != '':  #'indice_descarga' debe ser una lista de enteros contenidos en [0,len(self.search_cor2)]
            rango_descargas = self.indices_descarga

        for w in rango_descargas:
            folder_year_month_day = self.search_cor1[w]['fileid'].split('/')[5]
            #full_download_path = download_path+'/'+folder_year_month_day+'/'
            if not download_path:
                download_path = self.dir_descarga+"/".join(self.search_cor1['fileid'][w].split('/')[0:-1])+'/'

            if not os.path.exists(download_path):
                os.makedirs(download_path)
                print(f"Se ha creado el directorio {download_path}")
                carpetas_creadas.append(download_path)
            else:   
                print(f"El directorio {download_path} ya existe")

            ofile = download_path+self.search_cor1[w]['fileid'].split('/')[-1]          
            if not os.path.isfile(ofile):#Si archivo no descargado entonces que descargue.
                downloaded_files = Fido.fetch(self.search_cor1[w],path=download_path, max_conn=5, progress=True)       
                os.system('chgrp gehme {}'.format(ofile[0:-9]+'*'))
                os.system('chmod 775 {}'.format(ofile[0:-9]+'*'))
    

        print(f'Archivos descargados en: {download_path}')



class cor2_downloader:
    """
    Descripcion de la clase cor2 downloader
    Mencionar ejemplo.
    self.nivel = double, normal, sequential
    self.image_type = admite "img" o "seq". img puede ser self.nivel norm, seq; seq puede ser double o normal. 
    """
    def __init__(self, start_time, end_time,nave,nivel='',image_type='',size=''):
        try:
            self.start_time = start_time
            self.end_time = end_time
            self.instrumento = a.Instrument.secchi
            self.detector = a.Detector.cor2
            self.nave  = nave
            self.dir_descarga = '/gehme/data/stereo/'
            #self.dir_descarga = '/data_local/GCS/gcs/Imagenes/'
            self.nivel = nivel
            self.indices_descarga = '' # store the elements that will be effectivelly downloaded
            self.image_type = image_type #puede ser img o seq
            self.size = size
            self.ofiles = [] # files downloaded (full path)
        except TypeError:
            print("Be sure to add start_time, end_time, ship name, level/type of image when creating of object of this class.")
            raise
        except:
            print("WTF")

    def search(self):
        """
        Definicion del metodo search
        """
        search_cor2 = Fido.search(a.Time(self.start_time,self.end_time),self.instrumento,self.detector)
        if not search_cor2:
            print("No provider! No images available.")
            #global provider
            #provider = 0
            self.search_cor2 = search_cor2 #lista vacía
        else:
            match self.nave:
                case 'STEREO_A':
                    search_cor2_ = search_cor2['vso'][search_cor2['vso']['Source'] == 'STEREO_A'].copy()
                case 'STEREO_B':
                    search_cor2_ = search_cor2['vso'][search_cor2['vso']['Source'] == 'STEREO_B'].copy()        
            self.search_cor2 = search_cor2_
            if self.nivel != '':#asumo nivel definido correctamente
                self.filtro()

        #self.search_cor2 = search_cor2_

    def display(self):
        """
        Definicion del metodo display
        """
        lista_dates=[]#revisar scope, esto modificaria una lista fuera de esta definicion?
        search_size = len(self.search_cor2)
        #lista_colores = []
        lista_color_azul  = []
        lista_color_rojo  = []
        lista_color_verde = []
        lista_cant_images_azul  = []
        lista_cant_images_rojo  = []
        lista_cant_images_verde = []
        for i in range(search_size):
            lista_dates.append(datetime.fromisoformat(str(self.search_cor2[i][0])))
            #lista_cant_images.append(i)
            string_search_normal   = "n4c"
            string_search_double   = "d4c"
            string_search_sequence = "s4c"
            #if string_search_pb     in self.search_cor2[i]['fileid']: lista_colores.append('b')
            #if string_search_double in self.search_cor2[i]['fileid']: lista_colores.append('r')
            if string_search_normal in self.search_cor2[i]['fileid']: 
                lista_color_azul.append(datetime.fromisoformat(str(self.search_cor2[i][0])))
                lista_cant_images_azul.append(i)
            if string_search_double in self.search_cor2[i]['fileid']: 
                lista_color_rojo.append(datetime.fromisoformat(str(self.search_cor2[i][0])))
                lista_cant_images_rojo.append(i)
            if string_search_sequence in self.search_cor2[i]['fileid']:
                lista_color_verde.append(datetime.fromisoformat(str(self.search_cor2[i][0])))
                lista_cant_images_verde.append(i)

        #labels = ['double','pb']
        #fig, ax = plt.subplots()
        fig = plt.figure()
        ax  = fig.subplots()
        #ax.plot(lista_dates, lista_cant_images,c=lista_colores)
        ax.plot(lista_color_rojo,lista_cant_images_rojo  ,'ro',label='double')
        ax.plot(lista_color_azul,lista_cant_images_azul  ,'bo',label='normal')
        ax.plot(lista_color_verde,lista_cant_images_verde,'go',label='sequence')
        ax.set_xlabel('Dates')
        ax.set_ylabel('Images')
        ax.set_title('Available images per time, {}'.format(self.image_type))
        fig.autofmt_xdate()
        ax.legend()
        ax.grid(True)
        plt.show()



    def filtro(self):
        """
        Definicion del metodo filtro
        """
        #search_size = len(self.search_cor2)
        match self.nivel:
            case 'normal':
                string_search_nivel = "n4c"
                #search_cor2_ = self.search_cor2[self.search_cor2['Size'] == (8.03375*u.Mibyte)].copy()
            case 'double':
                string_search_nivel = "d4c"
                #search_cor2_ = self.search_cor2[self.search_cor2['Size'] == (8.0365*u.Mibyte)].copy()
            case 'sequence':
                string_search_nivel = "s4c"
            case '':
                string_search_nivel = ""
        match self.image_type:
            case 'img':
                string_search_type = "img"
            case 'seq':
                string_search_type = "seq"
            case '':
                string_search_type = ""

        lista_filtro = [] 
        for j in range(len(self.search_cor2)):
            if (string_search_nivel in self.search_cor2[j]['fileid'] and string_search_type in self.search_cor2[j]['fileid']): lista_filtro.append(j)
        self.search_cor2 = self.search_cor2[lista_filtro]  
        if not lista_filtro: print('Filtered images')

        if self.size and lista_filtro:
            lista_filtro2 = [] 
            for j in range(len(self.search_cor2)):
                if self.size <= int(round(self.search_cor2[j]['Size'].value)):
                    lista_filtro2.append(j)
            self.search_cor2 = self.search_cor2[lista_filtro2]
            
            if not lista_filtro2: print('Filtered images')

    def download(self, download_path_default=None):
        """
        Definicion del metodo download.
        """
        #match self.nivel:
        #    case 'polarizadas': suffix_nivel = ''
        #    case 'level_05':    suffix_nivel = 'L0'

        #match self.nave:
        #    case 'STEREO_A': suffix_nave = 'a'
        #    case 'STEREO_B': suffix_nave = 'b'

        #if not download_path:
            #download_path = self.dir_descarga+self.start_time.replace(" ","_").replace("/","").replace(":","_")+'/'+self.nave+'/Cor2/'+self.nivel
            #download_path = self.dir_descarga+suffix_nivel+'/'+suffix_nave+'/img/cor2/'  
            #download_path = self.dir_descarga+"/".join(self.search_cor2['fileid'][0].split('/')[0:-1])+'/'
        #if not os.path.exists(download_path):
        #    os.makedirs(download_path)
        #    print(f"Se ha creado el directorio {download_path}")
        #else:   
        #    print(f"El directorio {download_path} ya existe")
        carpetas_creadas = []
        if getattr(self,'indices_descarga') == '': 
            cantidad = len(self.search_cor2)
            rango_descargas = range(cantidad)
        if getattr(self,'indices_descarga') != '':  #'indice_descarga' debe ser una lista de enteros contenidos en [0,len(self.search_cor2)]
            rango_descargas = self.indices_descarga

        download_path=''
        for w in rango_descargas:
            if download_path:
                print(download_path)
            #folder_year_month_day = self.search_cor2[w]['fileid'].split('/')[5]
            #full_download_path = download_path+'/'+folder_year_month_day+'/'
            if download_path_default:
                download_path = download_path_default
            if not download_path_default:
                download_path = self.dir_descarga+"/".join(self.search_cor2['fileid'][w].split('/')[0:-1])+'/'
            if not os.path.exists(download_path):
                os.makedirs(download_path)
                print(f"Se ha creado el directorio {download_path}")
                carpetas_creadas.append(download_path)
            else:   
                print(f"El directorio {download_path} ya existe")

            ofile = download_path+self.search_cor2[w]['fileid'].split('/')[-1]
            self.ofiles.append(ofile)
            if not os.path.isfile(ofile):#Si archivo no descargado entonces que descargue.
                downloaded_files = Fido.fetch(self.search_cor2[w],path=download_path, max_conn=5, progress=True) 
                os.system('chgrp gehme {}'.format(ofile[0:-9]+'*'))
                os.system('chmod 775 {}'.format(ofile[0:-9]+'*'))
    

        print(f'Archivos descargados en: {download_path}')

class aia_downloader:
    def __init__(self, start_time, end_time, wavelength):
        self.start_time = start_time
        self.end_time = end_time
        self.wavelength = wavelength


class euvi_downloader:
    def __init__(self, start_time, end_time, wavelength):
        self.start_time = start_time
        self.end_time = end_time
        self.wavelength = wavelength





