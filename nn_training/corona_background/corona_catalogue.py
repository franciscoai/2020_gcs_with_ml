import pandas as pd
import os

exec_path = os.getcwd()
path=exec_path+'/catalogues/CMEs_sin_n.csv'
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

df.to_csv(exec_path+'/catalogues/Lista_Final_CMEs.csv',sep='\t', header=True,index=False)

print(df)

