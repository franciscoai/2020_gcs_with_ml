
import scipy.io as sio
from scipy.io import readsav
sav_data = readsav('/gehme/projects/2020_gcs_with_ml/repo/gcs_idl/arguments/input_ok.sav')
print(sav_data.items())
