import os
import pandas as pd
import matplotlib.pyplot as plt

data_dir='/gehme-gpu/projects/2020_gcs_with_ml/output/neural_cme_seg_v4/infer_neural_cme_seg_kincat/cor2_a'

fa=0
ext_folders = os.listdir(data_dir)
for ext_folder in ext_folders:
    if ext_folder=="35":
        print("Working on folder "+ext_folder+", folder "+str(fa)+" of "+str(len(ext_folder)-1))
        csv_path=data_dir+"/"+ext_folder+"/stats/"+ext_folder+"_stats"
        df=pd.read_csv(csv_path)
        df["DATE_TIME"]= pd.to_datetime(df["DATE_TIME"])
        df['HORA'] = df['DATE_TIME'].dt.hour
        grouped_data = df.groupby(['HORA']).agg({'MAX_ANG': 'max', 'MIN_ANG': 'min', 'CPA_ANG': 'mean'})
        #breakpoint()
        plt.figure(figsize=(10, 6))  # Tamaño del gráfico
        plt.scatter(grouped_data.index, grouped_data['MAX_ANG'], label='Ángulo Máximo', color='red')

        # Ángulo mínimo
        plt.scatter(grouped_data.index, grouped_data['MIN_ANG'], label='Ángulo Mínimo', color='blue')

        # Ángulo CPA promedio
        plt.scatter(grouped_data.index, grouped_data['CPA_ANG'], label='Ángulo CPA Promedio', color='green')

        # Agregar etiquetas y título
        plt.xlabel('Hora')
        plt.ylabel('Ángulo')
        plt.title('Ángulos por Hora')

        # Mostrar las etiquetas de las horas en el eje x
        plt.xticks(grouped_data.index)

        # Mostrar la leyenda
        plt.legend()

        # Mostrar el gráfico
        plt.tight_layout()
        plt.show()

         