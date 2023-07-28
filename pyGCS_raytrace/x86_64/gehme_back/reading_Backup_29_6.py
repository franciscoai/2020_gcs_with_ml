#Reading and printing input and output variables from IDL
import os
import ctypes
import pathlib
import numpy as np
import scipy.io as sio
from scipy.io import readsav
from numpy.ctypeslib import ndpointer
from cslug import CSlug, ptr

print("PRINTING INPUTS")
sav_data_input = readsav('/gehme/projects/2020_gcs_with_ml/repo/gcs_idl/arguments/input_ok.sav')
order = sorted(sav_data_input) #, key=lambda x : x.keys)
for i in order:
    print('********',i,'********')
    print(type(sav_data_input[i]))   
    print(sav_data_input[i].dtype)  
    print('shape = ', np.shape(sav_data_input[i]))
    print('size = ', np.size(sav_data_input[i]))
    print("min: ",np.min(sav_data_input[i]))
    print("mean: ",np.mean(sav_data_input[i]))
    print("max: ",np.max(sav_data_input[i]))
    print("dimension",(sav_data_input[i]).ndim) 

print("PRINTING OUTPUTS")
sav_data_output = readsav('/gehme/projects/2020_gcs_with_ml/repo/gcs_idl/arguments/output_ok.sav')

 
#Set enviroment inputs
os.environ['RT_PATH'] = '/usr/local/ssw/stereo/secchi'
os.environ['RT_SOFILENAME'] =  'libraytrace.so'
os.environ['RT_SOTHREADFILENAME'] = 'libraytracethread.so'
os.environ['RT_SOEXTENSION'] =  'so'
os.environ['RT_RUNFROM'] =  'local'
os.environ['RT_DATAPATH'] =  '/usr/local/ssw/stereo/secchi/data/scraytrace'
os.environ['SSW'] =  '/usr/local/ssw'
os.environ['RT_SOSUBPATH'] =  'lib/linux/x86_64'
os.environ['RT_FORCELIBFILE'] =  ' '
os.environ['RT_LIBFILE'] =  '/usr/local/ssw/stereo/secchi/lib/linux/x86_64/libraytrace.so'
os.environ['RT_FORCELIBTHREAD'] =  ''
os.environ['RT_LIBFILETHREAD'] =  ''


#importing libraytrace.so from C++
if __name__ == "__main__":
    # Load the shared library into ctypes
     libraytrace = pathlib.Path('/usr/local/ssw/stereo/secchi/lib/linux/x86_64/libraytrace.so').absolute()
     c_lib = ctypes.CDLL(libraytrace)

float_pointer1 = np.ctypeslib.ndpointer(dtype=np.dtype('>f4'),ndim=1,flags="CONTIGUOUS")
float_pointer2 = np.ctypeslib.ndpointer(dtype=np.dtype('>f4'),ndim=2,flags="CONTIGUOUS")

long_pointer1 = np.ctypeslib.ndpointer(dtype=np.dtype('>i4'),ndim=1,flags="CONTIGUOUS")
long_pointer2 = np.ctypeslib.ndpointer(dtype=np.dtype('>i4'),ndim=2,flags="CONTIGUOUS")

#Working with Csulg



imsize_0 = ctypes.c_long(sav_data_input['imsize'][0])
imsize_1 = ctypes.c_long(sav_data_input['imsize'][1])                               
fovpix = ctypes.c_float(sav_data_input['fovpix'])                               
obspos = float_pointer1(sav_data_input['obspos'])
obsang = float_pointer1(sav_data_input['obsang'])                    
nepos = float_pointer1(sav_data_input['nepos'])      
neang = float_pointer1(sav_data_input['neang'])  
losnbp = ctypes.c_long(sav_data_input['losnbp'])                              
losrange = float_pointer1(sav_data_input['losrange'])
modelid = ctypes.c_long(sav_data_input['modelid'])                              
btot = float_pointer2(sav_data_input['btot'])
bpol = float_pointer2(sav_data_input['bpol'])
netot = float_pointer2(sav_data_input['netot'])     
modparam = float_pointer1(sav_data_input['modparam'])
crpix = float_pointer1(sav_data_input['crpix'])
rho = float_pointer2(sav_data_input['rho'])
mmlon = float_pointer1(sav_data_input['mmlon'])
mmlat = float_pointer1(sav_data_input['mmlat'])
rrr = float_pointer2(sav_data_input['rrr'])        
pofinteg = ctypes.c_long(sav_data_input['pofinteg'])                                
quiet = ctypes.c_long(sav_data_input['quiet'])                                  
neonly = ctypes.c_long(sav_data_input['neonly'])  
roi = long_pointer2(sav_data_input['roi'])
poiang = float_pointer1(sav_data_input['poiang'])
hlonlat = float_pointer1(sav_data_input['hlonlat'])
occrad = ctypes.c_float(sav_data_input['occrad'])                                        
adapthres = ctypes.c_float(sav_data_input['adapthres'])                                      
maxsubdiv = ctypes.c_long(sav_data_input['maxsubdiv'])
limbdark = ctypes.c_float(sav_data_input['limbdark'])                         
rotmat = float_pointer2(sav_data_input['rotmat'])
obslonlat = float_pointer1(sav_data_input['obslonlat'])
obslonlatflag = ctypes.c_long(sav_data_input['obslonlatflag'])
projtypecode = ctypes.c_long(sav_data_input['projtypecode'])                       
pv2_1 = ctypes.c_double(sav_data_input['pv2_1'])                               
pc = float_pointer2(sav_data_input['pc'])
frontinteg = ctypes.c_long(sav_data_input['frontinteg'])                          
uvinteg = ctypes.c_long(sav_data_input['uvinteg'])                              
nerotcntr = float_pointer1(sav_data_input['nerotcntr'])
nerotang = float_pointer1(sav_data_input['nerotang'])
netranslation = float_pointer1(sav_data_input['netranslation'])
nerotaxis = long_pointer1(sav_data_input['nerotaxis'])                                

slug = CSlug('/usr/local/ssw/stereo/secchi/lib/linux/x86_64/libraytrace.so')

slug.dll.rtraytracewcs(imsize_0,
imsize_1,
fovpix,
obspos,
obsang,
nepos,
neang,
losnbp,
losrange,
modelid,
ctypes.byref(btot),
ctypes.byref(bpol),
ctypes.byref(netot),
modparam,
crpix,
ctypes.byref(rho),
mmlon,
mmlat,
ctypes.byref(rrr),
pofinteg,
quiet,
neonly,
roi,
poiang,
hlonlat,
occrad,
adapthres,
maxsubdiv,
limbdark,
ctypes.byref(rotmat),
obslonlat,
obslonlatflag,
projtypecode,
pv2_1,
pc,
frontinteg,
uvinteg,
nerotcntr,
nerotang,
netranslation,
nerotaxis)



               


