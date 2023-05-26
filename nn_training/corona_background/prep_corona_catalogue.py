import pandas as pd
import os

"""
Reads pairs of LVL1 coronograph images from various instruments and saves a differential corona for each pair.
Images are resized
"""

exec_path = os.getcwd()
path=exec_path+'/Lista_Final_CMEs.csv' # file with the list of cor files
df= pd.read_csv(path , sep="\t")

# convertir las columnas de fecha y hora en objetos de fecha y hora de Pandas
df['evento_a'] = pd.to_datetime(df['date_a'] + ' ' + df['time_a'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df['evento_b'] = pd.to_datetime(df['date_b'] + ' ' + df['time_b'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# crear una nueva columna con la fecha y hora concatenadas
#df['fecha_hora_a'] = df['fecha_hora_a'].dt.strftime('%Y-%m-%d %H:%M:%S')
#df['fecha_hora_b'] = df['fecha_hora_b'].dt.strftime('%Y-%m-%d %H:%M:%S')

# crear columnas nuevas con las horas restadas
df['preevento_a_1h'] = (df['evento_a'] - pd.to_timedelta('1 hour')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
df['preevento_b_1h'] = (df['evento_b'] - pd.to_timedelta('1 hour')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')

df['preevento_a_3h'] = (df['evento_a'] - pd.to_timedelta('2 hours')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
df['preevento_b_3h'] = (df['evento_b'] - pd.to_timedelta('2 hours')).dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')

df.to_csv('Lista_Final_CMEs.txt',sep='\t', header=True,index=False)

print(df)

