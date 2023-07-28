
#Reading and printing input and output variables from IDL
import os
from ctypes import *
import pathlib
import numpy as np
import scipy.io as sio
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
     c_lib = CDLL(libraytrace)




float_pointer1 = np.ctypeslib.ndpointer(dtype=np.dtype('>f4'),ndim=1,flags="CONTIGUOUS")
float_pointer2 = np.ctypeslib.ndpointer(dtype=np.dtype('>f4'),ndim=2,flags="CONTIGUOUS")
long_pointer1 = np.ctypeslib.ndpointer(dtype=np.dtype('>i4'),ndim=1,flags="CONTIGUOUS")
long_pointer2 = np.ctypeslib.ndpointer(dtype=np.dtype('>i4'),ndim=2,flags="CONTIGUOUS")


c_lib.rtraytracewcs.restype = c_int
c_lib.rtraytracewcs.argtypes = c_int,POINTER(c_void_p)

# tuple of pointers with the appropiate type to point to all the input variables
args_types = (
pointer(c_long),                          
pointer(c_long),                                   #imsize[1]
pointer(c_float),                                  #fovpix
float_pointer1,                                  #obspos
float_pointer1,                                  #obsang
float_pointer1,                                  #nepos
float_pointer1,                                  #neang
pointer(c_long),                                   #losnbp
float_pointer1,                                  #losrange
pointer(c_long),                                   #modelid
float_pointer2,                                  #btot
float_pointer2,                                  #bpol
float_pointer2,                                  #netot
float_pointer1,                                  #modparam
float_pointer1,                                  #crpix
float_pointer2,                                  #rho
float_pointer1,                                  #mmlon
float_pointer1,                                  #mmlat
float_pointer2,                                  #rrr
pointer(c_long),                                   #pofinteg
pointer(c_long),                                   #quiet
pointer(c_long),                                   #neonly
long_pointer2,                                   #roi
float_pointer1,                                  #poiang
float_pointer1,                                  #hlonlat
pointer(c_float),                                  #occrad
pointer(c_float),                                  #adapthres
pointer(c_long),                                   #maxsubdiv
pointer(c_float),                                  #limbdark
float_pointer2,                                  #rotmat
float_pointer1,                                  #obslonlat
pointer(c_long),                                   #obslonlatflag
pointer(c_long),                                   #projtypecode
c_double,                                 #pv2_1
float_pointer2,                                  #pc
pointer(c_long),                                   #frontinteg
pointer(c_long),                                   #uvinteg
float_pointer1,                                  #nerotcntr
float_pointer1,                                  #nerotang
float_pointer1,                                  #netranslation
long_pointer1)

# tuple with all the input variables
args=[
sav_data_input['imsize'][0],
sav_data_input['imsize'][1],                           
sav_data_input['fovpix'],                                
sav_data_input['obspos'],
sav_data_input['obsang'],                    
sav_data_input['nepos'],      
sav_data_input['neang'],  
sav_data_input['losnbp'],                                
sav_data_input['losrange'],
sav_data_input['modelid'],                               
sav_data_input['btot'],
sav_data_input['bpol'],
sav_data_input['netot'],     
sav_data_input['modparam'],  
sav_data_input['crpix'],
sav_data_input['rho'],
sav_data_input['mmlon'],
sav_data_input['mmlat'],
sav_data_input['rrr'],           
sav_data_input['pofinteg'],                                
sav_data_input['quiet'],                                  
sav_data_input['neonly'],  
sav_data_input['roi'],
sav_data_input['poiang'],
sav_data_input['hlonlat'],
sav_data_input['occrad'],                                         
sav_data_input['adapthres'],                                      
sav_data_input['maxsubdiv'],
sav_data_input['limbdark'],                         
sav_data_input['rotmat'],
sav_data_input['obslonlat'],
sav_data_input['obslonlatflag'], 
sav_data_input['projtypecode'],                        
sav_data_input['pv2_1'],                               
sav_data_input['pc'],
sav_data_input['frontinteg'],                          
sav_data_input['uvinteg'],                              
sav_data_input['nerotcntr'],
sav_data_input['nerotang'],
sav_data_input['netranslation'],
sav_data_input['nerotaxis']]

all_args = (c_void_p * len(args))() # array of void pointers

1- Entender la notacion puntero[i] en c++ : Remember, the expression p[i] is functionally identical to *(p+i).
2- Seguramente hay que usar cast para pasar de punteros especificos a puntero np.void

c_lib.rtraytracewcs(len(args), all_args)



# c_lib.rtraytracewcs.restype = c_int

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

