import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
"""
To DO: generate a function to read the pickle files and return the dictionary with the data
generate a function to plot the data
"""


save = True
# Define an empty list to store data points
data_points = []

# Specify the directory containing your pickle files (replace with your actual path)
pickle_dir = "/gehme/projects/2023_eeggl_validation/output/2011-02-15/eeggl/run005/runing_diff/neural_cme_seg_v5/infer2/mask_props"
pickle_dir = "/gehme/projects/2023_eeggl_validation/output/2011-02-15/eeggl/run016/runing_diff/neural_cme_seg_v5/infer2/mask_props"
pickle_dir = "/gehme/projects/2023_eeggl_validation/output/2011-02-15/eeggl/run016/runing_diff"

# Create the plot
fig_cpa, ax_cpa   = plt.subplots()
fig_apex, ax_apex = plt.subplots()
# Agregar etiquetas a los ejes
ax_cpa.set_xlabel("x")
ax_cpa.set_ylabel("y")
ax_apex.set_xlabel("x")
ax_apex.set_ylabel("y")
style = '*'
parameters= ['CPA', 'MASK_ID', 'SCR', 'WA', 'APEX', 'CME_ID']
colors=['r','b','g','k','y','m','c','w']
instrument=['Cor2A','Cor2B','C2']
run=[5,16]
contador=0
# Loop through all files in the directory
for filename in os.listdir(pickle_dir):
    # Check if the file is a pickle file
    if filename.endswith("parametros_filtered.pkl"):
        # Open the file and load the dictionary
        with open(os.path.join(pickle_dir, filename), "rb") as f:
            data = pickle.load(f)
        
        #data_points.append((data["x_points"], data["y_points"]))

        x_list_cpa = data["x_pointsCPA"]
        y_list_cpa = data["y_pointsCPA"]
        ax_cpa.plot(x_list_cpa, y_list_cpa, style,color=colors[contador])
        x_list_apex = data["x_pointsAPEX"]
        y_list_apex = data["y_pointsAPEX"]
        ax_apex.plot(x_list_apex, y_list_apex, style,color=colors[contador])
        #set legend acording color
        ax_cpa.legend([contador])
        ax_apex.legend(parameters[contador])
    contador+=1
# Add labels and title
#y_title=data['parameter']
x_title='Date and hour'
ax_cpa.set_title(instrument[0])    
ax_cpa.set_xlabel(x_title)
ax_cpa.set_ylabel("CPA")
ax_apex.set_title(instrument[0])    
ax_apex.set_xlabel(x_title)
ax_apex.set_ylabel("Apex")
#plt.xticks(dt_list,rotation=90)

#hours = [str(timestamp.time()) for timestamp in x_list]
#hours1 = [datetime.strptime(timestamp, "%H:%M:%S") for timestamp in hours]
#hours2 = [timestamp.strftime("%H:%M") for timestamp in hours1]
#plt.xticks(x_list,hours2,rotation=90)
plt.grid()
ending = '_test2_plot'
if save:
    os.makedirs(pickle_dir, exist_ok=True)
    fig_cpa.savefig(pickle_dir+'/'+str.lower("cpa")+ending+".png")
    
    
    
    fig_apex.savefig(pickle_dir+'/'+str.lower("apex")+ending+".png")
    plt.close()

# Show the plot
#plt.show()