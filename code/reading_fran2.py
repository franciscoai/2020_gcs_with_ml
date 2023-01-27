
#Reading and printing input and output variables from IDL
import os
from ctypes import *
import pathlib
import numpy as np
import scipy.io as sio
from scipy.io import readsav
from numpy.ctypeslib import load_library,ndpointer
from multiprocessing import sharedctypes
import matplotlib.pyplot as plt

#python_dict=True porque por defecto no es tipo python lo que entrega
sav_data_input = readsav('/gehme/projects/2020_gcs_with_ml/repo/gcs_idl/arguments/input_ok.sav',python_dict=True)
sav_data_output = readsav('/gehme/projects/2020_gcs_with_ml/repo/gcs_idl/arguments/output_ok.sav',python_dict=True)
# print(sav_data_output['fovpix'])
# print(sav_data_output['obspos'])
# print(sav_data_output['obsang'])
# print(sav_data_output['nepos'])
# print(sav_data_output['neang'])
# print(sav_data_output['nerotcntr'])
# print(sav_data_output['nerotang'])
# print(sav_data_output['nerotaxis'])
# print(sav_data_output['netranslation'])
# print(sav_data_output['losnbp'])
# print(sav_data_output['losrange'])
# print(sav_data_output['occrad'])
# print(sav_data_output['adapthres'])
# print(sav_data_output['maxsubdiv'])
# print(sav_data_output['limbdark'])
# print(sav_data_output['obslonlatflag'])
# print(sav_data_output['uvinteg'])
# print(sav_data_output['crpix'])
# print(sav_data_output['pc'])

order = sorted(sav_data_input) #, key=lambda x : x.keys)
""" for i in order:
    print('********',i,'********')
    print(type(sav_data_input[i]))   
    print(sav_data_input[i].dtype)  
    print('shape = ', np.shape(sav_data_input[i]))
    print('size = ', np.size(sav_data_input[i]))
    print("min: ",np.min(sav_data_input[i]))
    print("mean: ",np.mean(sav_data_input[i]))
    print("max: ",np.max(sav_data_input[i]))
    print("dimension",(sav_data_input[i]).ndim) 
""" 
 
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
# os.environ['RT_UVEMISSIONPATH'] =  '/usr/local/ssw/stereo/secchi/cpp/scraytrace/data' #added on 222.12.26

#Inicialice quiet 
sav_data_input['quiet'] = np.int32(0)


#importing libraytrace.so from C++
if __name__ == "__main__":
    # Load the shared library into ctypes
    c_lib = load_library('libraytrace.so','/usr/local/ssw/stereo/secchi/lib/linux/x86_64/')
    
#clase tipo estructura con los tipo de punteros
#queda más bonito y es más rápido cuando se llama a la función
class InputStructure(Structure):
    _fields_ = [
            ('imsize_0', c_void_p),
            ('imsize_1', c_void_p),
            ('fovpix', c_void_p),
            ('obspos', c_void_p),
            ('obsang', c_void_p),
            ('nepos', c_void_p),
            ('neang', c_void_p),
            ('losnbp', c_void_p),
            ('losrange', c_void_p),
            ('modelid', c_void_p),
            ('btot', c_void_p),
            ('bpol', c_void_p),
            ('netot', c_void_p),
            ('modparam', c_void_p),
            ('crpix', c_void_p),
            ('rho', c_void_p),
            ('mmlon', c_void_p),
            ('mmlat', c_void_p),
            ('rrr', c_void_p),
            ('pofinteg', c_void_p),
            ('quiet', c_void_p),
            ('neonly', c_void_p),
            ('roi', c_void_p),
            ('poiang', c_void_p),
            ('hlonlat', c_void_p),
            ('occrad', c_void_p),
            ('adapthres', c_void_p),
            ('maxsubdiv', c_void_p),
            ('limbdark', c_void_p),
            ('rotmat', c_void_p),
            ('obslonlat', c_void_p),
            ('obslonlatflag', c_void_p),
            ('projtypecode', c_void_p),
            ('pv2_1', c_void_p),
            ('pc', c_void_p),
            ('frontinteg', c_void_p),
            ('uvinteg', c_void_p),
            ('nerotcntr', c_void_p),
            ('nerotang', c_void_p),
            ('netranslation', c_void_p),
            ('nerotaxis', c_void_p)
   
    ]

# inputs de la estructura casteadas para que coincidan con el tipo void
# (las arrays con .copy() porque sino sale un error de array readonly)

imsize_0 = cast(pointer(c_int32(sav_data_input['imsize'][0])),c_void_p)
imsize_1 = cast(pointer(c_int32(sav_data_input['imsize'][1])),c_void_p)
fovpix = cast(pointer(c_float(sav_data_input['fovpix'])),c_void_p)
obspos = cast(np.ctypeslib.as_ctypes(sav_data_input['obspos'].astype('float32').copy()),c_void_p)
obsang = cast(np.ctypeslib.as_ctypes(sav_data_input['obsang'].astype('float32').copy()),c_void_p)
nepos = cast(np.ctypeslib.as_ctypes(sav_data_input['nepos'].astype('float32').copy()),c_void_p)
neang = cast(np.ctypeslib.as_ctypes(sav_data_input['neang'].astype('float32').copy()),c_void_p)
losnbp = cast(pointer(c_int32(sav_data_input['losnbp'])),c_void_p)
losrange = cast(np.ctypeslib.as_ctypes(sav_data_input['losrange'].astype('float32').copy()),c_void_p)
modelid = cast(pointer(c_int32(sav_data_input['modelid'])),c_void_p)
btot = cast(np.ctypeslib.as_ctypes(sav_data_input['btot'].astype('float32').copy()),c_void_p)
bpol = cast(np.ctypeslib.as_ctypes(sav_data_input['bpol'].astype('float32').copy()),c_void_p)
netot = cast(np.ctypeslib.as_ctypes(sav_data_input['netot'].astype('float32').copy()),c_void_p)
modparam = cast(np.ctypeslib.as_ctypes(sav_data_input['modparam'].astype('float32').copy()),c_void_p)
crpix = cast(np.ctypeslib.as_ctypes(sav_data_input['crpix'].astype('float32').copy()),c_void_p)
rho = cast(np.ctypeslib.as_ctypes(sav_data_input['rho'].astype('float32').copy()),c_void_p)
mmlon = cast(np.ctypeslib.as_ctypes(sav_data_input['mmlon'].astype('float32').copy()),c_void_p)
mmlat = cast(np.ctypeslib.as_ctypes(sav_data_input['mmlat'].astype('float32').copy()),c_void_p)
rrr = cast(np.ctypeslib.as_ctypes(sav_data_input['rrr'].astype('float32').copy()),c_void_p)       
pofinteg =  cast(pointer(c_int32(sav_data_input['pofinteg'])),c_void_p)   
quiet =  cast(pointer(c_int32(sav_data_input['quiet'])),c_void_p)                            
neonly =  cast(pointer(c_int32(sav_data_input['neonly'])),c_void_p)
roi =  cast(np.ctypeslib.as_ctypes(sav_data_input['roi'].astype('int32').copy()),c_void_p)
poiang = cast(np.ctypeslib.as_ctypes(sav_data_input['poiang'].astype('float32').copy()),c_void_p)
hlonlat = cast(np.ctypeslib.as_ctypes(sav_data_input['hlonlat'].astype('float32').copy()),c_void_p)
occrad = cast(pointer(c_float(sav_data_input['occrad'])),c_void_p)                                         
adapthres = cast(pointer(c_float(sav_data_input['adapthres'])),c_void_p)                                   
maxsubdiv =  cast(pointer(c_int32(sav_data_input['maxsubdiv'])),c_void_p)
limbdark = cast(pointer(c_float(sav_data_input['limbdark'])),c_void_p)                     
rotmat =  cast(np.ctypeslib.as_ctypes(sav_data_input['rotmat'].astype('float32').copy()),c_void_p)
obslonlat =  cast(np.ctypeslib.as_ctypes(sav_data_input['obslonlat'].astype('float32').copy()),c_void_p)
obslonlatflag = cast(pointer(c_int32(sav_data_input['obslonlatflag'])),c_void_p)#ver
projtypecode = cast(pointer(c_int32(sav_data_input['projtypecode'])),c_void_p)                       
pv2_1 = cast(pointer(c_double(sav_data_input['pv2_1'])),c_void_p)  #ver                          
pc = cast(np.ctypeslib.as_ctypes(sav_data_input['pc'].astype('float32').copy()),c_void_p)
frontinteg = cast(pointer(c_int32(sav_data_input['frontinteg'])),c_void_p)                        
uvinteg = cast(pointer(c_int32(sav_data_input['uvinteg'])),c_void_p)                             
nerotcntr = cast(np.ctypeslib.as_ctypes(sav_data_input['nerotcntr'].astype('float32').copy()),c_void_p)
nerotang = cast(np.ctypeslib.as_ctypes(sav_data_input['nerotang'].astype('float32').copy()),c_void_p)
netranslation = cast(np.ctypeslib.as_ctypes(sav_data_input['netranslation'].astype('float32').copy()),c_void_p)
nerotaxis= cast(np.ctypeslib.as_ctypes(sav_data_input['nerotaxis'].astype('int32').copy()),c_void_p)


input_obj = InputStructure(
                    imsize_0,
                    imsize_1,
                    fovpix,
                    obspos,
                    obsang,
                    nepos,
                    neang,
                    losnbp,
                    losrange,
                    modelid,
                    btot,
                    bpol,
                    netot,
                    modparam,
                    crpix,
                    rho,
                    mmlon,
                    mmlat,
                    rrr,
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
                    rotmat,
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
                    nerotaxis
)


print("********************Corriendo...")
c_lib.rtraytracewcs.restype = c_bool
c_lib.rtraytracewcs.argtypes = [c_int, POINTER(InputStructure)]
c_lib.rtraytracewcs(41,input_obj)
print("*******************corrio completo")

# obspos = np.ctypeslib.as_array((c_float * 3).from_address(input_obj.obspos))
# obspos = obspos.newbyteorder('<')
# print('obspos', obspos)

# bpol = np.ctypeslib.as_array((c_float * 262144).from_address(input_obj.bpol))#addressof(y.contents)))
# bpol = bpol.newbyteorder('<')
# print('bpol', bpol)
# bpol = np.reshape(bpol,(512,512))
# m=np.nanmean(bpol)
# sd=np.nanstd(bpol)
# print(m, sd)
# plt.imshow(bpol,vmax=m+3*sd,vmin=m-3*sd)
# plt.show()

btot = np.ctypeslib.as_array((c_float * 262144).from_address(input_obj.btot))#addressof(y.contents)))
btot = btot.newbyteorder('<')
print('btot', btot)
btot = np.reshape(btot,(512,512))
m=np.nanmean(btot)
sd=np.nanstd(btot)
print(m, sd)
plt.imshow(btot,vmax=m+3*sd,vmin=m-3*sd, origin='lower')
plt.show()

# #agradecimiento especial a:
# #https://torroja.dmt.upm.es/media/files/ctypes.pdf
# #https://www.youtube.com/watch?v=p_LUzwylf-Y&ab_channel=C%2FC%2B%2BDublinUserGroup
# #https://stackoverflow.com/questions/38839525/ctypes-to-int-conversion-with-variable-assigned-by-reference
# #https://blog.opencore.ch/posts/python-cpp-numpy-zero-copy/