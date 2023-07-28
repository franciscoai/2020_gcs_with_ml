
#Reading and printing input and output variables from IDL
import os
import ctypes
import pathlib
import numpy as np
import scipy.io as sio
from typing import cast
from scipy.io import readsav
from numpy.ctypeslib import ndpointer



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

#Inicialice quiet 
sav_data_input['quiet'] = np.int32(0)


#importing libraytrace.so from C++
if __name__ == "__main__":
    # Load the shared library into ctypes
     libraytrace = pathlib.Path('/usr/local/ssw/stereo/secchi/lib/linux/x86_64/libraytrace.so').absolute()
     c_lib = ctypes.CDLL(libraytrace)




float_pointer1 = np.ctypeslib.ndpointer(dtype=np.dtype('>f4'),ndim=1,flags="CONTIGUOUS")
float_pointer2 = np.ctypeslib.ndpointer(dtype=np.dtype('>f4'),ndim=2,flags="CONTIGUOUS")

long_pointer1 = np.ctypeslib.ndpointer(dtype=np.dtype('>i4'),ndim=1,flags="CONTIGUOUS")
long_pointer2 = np.ctypeslib.ndpointer(dtype=np.dtype('>i4'),ndim=2,flags="CONTIGUOUS")


c_lib.rtraytracewcs.restype = ctypes.c_int
c_lib.rtraytracewcs.argtypes = ctypes.c_int,ctypes.POINTER(ctypes.c_void_p) #elimine lo que decia fran pointer(pointer(v_void_p))
args = [ctypes.c_long,                          
ctypes.c_long,                                #imsize[1]
ctypes.c_float,                                  #fovpix
float_pointer1,                                  #obspos
float_pointer1,                                  #obsang
float_pointer1,                                   #nepos
float_pointer1,                                   #neang
ctypes.c_long,                                   #losnbp
float_pointer1,                                #losrange
ctypes.c_long,                                  #modelid
float_pointer2,                                    #btot
float_pointer2,                                    #bpol
float_pointer2,                                   #netot
float_pointer1,                                #modparam
float_pointer1,                                   #crpix
float_pointer2,                                  #rho
float_pointer1,                                  #mmlon
float_pointer1,                                  #mmlat
float_pointer2,                                  #rrr
ctypes.c_long,                                   #pofinteg
ctypes.c_long,                                   #quiet
ctypes.c_long,                                   #neonly
long_pointer2,                                   #roi
float_pointer1,                                  #poiang
float_pointer1,                                  #hlonlat
ctypes.c_float,                                  #occrad
ctypes.c_float,                                  #adapthres
ctypes.c_long,                                   #maxsubdiv
ctypes.c_float,                                  #limbdark
float_pointer2,                                  #rotmat
float_pointer1,                                  #obslonlat
ctypes.c_long,                                   #obslonlatflag
ctypes.c_long,                                   #projtypecode
ctypes.c_double,                                 #pv2_1
float_pointer2,                                  #pc
ctypes.c_long,                                   #frontinteg
ctypes.c_long,                                   #uvinteg
float_pointer1,                                  #nerotcntr
float_pointer1,                                  #nerotang
float_pointer1,                                  #netranslation
long_pointer1]

variables= [(sav_data_input['imsize'][0]),
(sav_data_input['imsize'][1]),
(sav_data_input['fovpix']),
(sav_data_input['obspos']),
(sav_data_input['obsang']),
(sav_data_input['nepos']),
(sav_data_input['neang']),
(sav_data_input['losnbp']),
(sav_data_input['losrange']),
(sav_data_input['modelid']),
(sav_data_input['btot']),
(sav_data_input['bpol']),
(sav_data_input['netot']),
(sav_data_input['modparam']),
(sav_data_input['crpix']),
(sav_data_input['rho']),
(sav_data_input['mmlon']),
(sav_data_input['mmlat']),
(sav_data_input['rrr']),
(sav_data_input['pofinteg']),
(sav_data_input['quiet']),
(sav_data_input['neonly']),
(sav_data_input['roi']),
(sav_data_input['poiang']),
(sav_data_input['hlonlat']),
(sav_data_input['occrad']),
(sav_data_input['adapthres']),
(sav_data_input['maxsubdiv']),
(sav_data_input['limbdark']),
(sav_data_input['rotmat']),
(sav_data_input['obslonlat']),
(sav_data_input['obslonlatflag']),
(sav_data_input['projtypecode']),
(sav_data_input['pv2_1']),
(sav_data_input['pc']),
(sav_data_input['frontinteg']),
(sav_data_input['uvinteg']),
(sav_data_input['nerotcntr']),
(sav_data_input['nerotang']),
(sav_data_input['netranslation']),
(sav_data_input['nerotaxis'])]

n_vars=len(args)
for i in range(len(args)):
    args[i]=variables[i]


# args = cast(args,ctypes.POINTER(ctypes.c_void_p))
# print(args)
c_lib.rtraytracewcs(n_vars,args)







# c_lib.rtraytracewcs.restype = ctypes.c_int

# c_lib.rtraytracewcs(sav_data_input['imsize'][0],
# sav_data_input['imsize'][1],                           
# sav_data_input['fovpix'],                                
# sav_data_input['obspos'],
# sav_data_input['obsang'],                    
# sav_data_input['nepos'],      
# sav_data_input['neang'],  
# sav_data_input['losnbp'],                                
# sav_data_input['losrange'],
# sav_data_input['modelid'],                               
# sav_data_input['btot'],
# sav_data_input['bpol'],
# sav_data_input['netot'],     
# sav_data_input['modparam'],  
# sav_data_input['crpix'],
# sav_data_input['rho'],
# sav_data_input['mmlon'],
# sav_data_input['mmlat'],
# sav_data_input['rrr'],           
# sav_data_input['pofinteg'],                                
# sav_data_input['quiet'],                                  
# sav_data_input['neonly'],  
# sav_data_input['roi'],
# sav_data_input['poiang'],
# sav_data_input['hlonlat'],
# sav_data_input['occrad'],                                         
# sav_data_input['adapthres'],                                      
# sav_data_input['maxsubdiv'],
# sav_data_input['limbdark'],                         
# sav_data_input['rotmat'],
# sav_data_input['obslonlat'],
# sav_data_input['obslonlatflag'], 
# sav_data_input['projtypecode'],                        
# sav_data_input['pv2_1'],                               
# sav_data_input['pc'],
# sav_data_input['frontinteg'],                          
# sav_data_input['uvinteg'],                              
# sav_data_input['nerotcntr'],
# sav_data_input['nerotang'],
# sav_data_input['netranslation'],
# sav_data_input['nerotaxis'])


print("corrio completo")

