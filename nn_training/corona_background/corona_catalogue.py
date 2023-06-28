# Crea un catalogo para descargar imagenes preevento, para buscar coronas de fondo, a partir del catálogo de CMEs se encuentra en http://www.affects-fp7.eu/cme-database/database.php
import pandas as pd
import os

exec_path = os.getcwd()
path=exec_path+'/catalogues/CMEs_sin_n.csv'
df= pd.read_csv(path , sep="\t")

# Convertir las columnas de fecha y hora en objetos de fecha y hora de Pandas
df['evento_a'] = pd.to_datetime(df['date_a'] + ' ' + df['time_a'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df['evento_b'] = pd.to_datetime(df['date_b'] + ' ' + df['time_b'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Crear una nueva columna con la fecha y hora concatenadas
#df['fecha_hora_a'] = df['fecha_hora_a'].dt.strftime('%Y-%m-%d %H:%M:%S')
#df['fecha_hora_b'] = df['fecha_hora_b'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Crear columnas nuevas con las horas restadas
df['preevento_a_1h'] = (df['evento_a'] - pd.to_timedelta('1 hour')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
df['preevento_b_1h'] = (df['evento_b'] - pd.to_timedelta('1 hour')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
df['preevento_a_2h'] = (df['evento_a'] - pd.to_timedelta('2 hours')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
df['preevento_b_2h'] = (df['evento_b'] - pd.to_timedelta('2 hours')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')

# Copiar los valores de la columna del preevento que si existe, en caso que un evento no esté. Esto es porque me interesan los preeventos , entonces si la CME esta o no me da igual
df.loc[df['preevento_b_1h'] == 'NaT', 'preevento_b_1h'] = df['preevento_a_1h']
df.loc[df['preevento_b_2h'] == 'NaT', 'preevento_b_2h'] = df['preevento_a_2h']
df.loc[df['preevento_a_1h'] == 'NaT', 'preevento_a_1h'] = df['preevento_b_1h']
df.loc[df['preevento_a_2h'] == 'NaT', 'preevento_a_2h'] = df['preevento_b_2h']

df.to_csv('Lista_Final_CMEs.txt',sep='\t', header=True,index=False)

print(df)

