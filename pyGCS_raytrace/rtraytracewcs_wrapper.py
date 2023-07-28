
# Reading and printing input and output variables from IDL
import os
from ctypes import *
import pathlib
import numpy as np
import scipy.io as sio
from scipy.io import readsav
from numpy.ctypeslib import load_library, ndpointer
from multiprocessing import sharedctypes
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def rtraytracewcs_wrapper(input_data, test=False,compare=False):
    """
    Wrapper of the C++ library  "ssw/stereo/secchi/lib/linux/x86_64/libraytrace.so" to compute ray traceing 
    from a electronic cloud

    @authors: F. A. Iglesias, Y. Machuca and F. Cisterna. 2023. Mendoza Group for Heloiphysics Studies (GEHMe), Argentina
    Contact: franciscoaiglesias@gmail.com

    test::Bool # Set to True to ignore input_data and run the example saved in test_save_file 
    input_data:: dict with all the inputs, see below (extracted from raytrace.cpp in libraytrace.so):
        int *pmodelid; //!< Model number
        int *pis; //!< X size of the image
        int *pjs; //!< Y size of the image
        float *pfovpix; //!< Pixel resolution [rad]
        float *pobspos; //!< Observer position [Rsun]
        float *pobsang; //!< Observer camera orientation [rad]
        float *pnepos; //!< Electron density center positon [Rsun]
        float *pneang; //!< Electron density orientation [rad]
        int *plosnbp; //!< LOS number of integration steps
        float *plosrange; //!< LOS integration range [Rsun]
        float *pmparam; //!< Model parameter array pointer
                 	pparam[0]; // 2.55; //dist to bottom of structure // 
                    pparam[1]; // *DTOR; //30.*DTOR; // angle between axis and foot [rad]
                    pparam[2]; // leg height [Rsun]
                    pparam[3]; // ratio of tube radius to height //.2 // this is K
                    pparam[4]; // electron density 
                    pparam[5]; // thickness of the skeleton: set to 0. if no display requested 
                    pparam[8]; // Thickness of the skin: pseudo-gaussian profile -- inner sigma 
                    pparam[9]; // Thickness of the skin: pseudo-gaussian profile-- front sigma        
        int *pquiet; //!< Flag: Disable display of program progression
        float *pbtot; //!< Total brightness image pointer
        float *pbpol; //!< Polarized brightness image pointer
        float *pnetot; //!< Integrated electron density image pointer
        float *pcrpix; //!< Position of the Sun center on the image
        float *prhoim; //!< Image giving the impact parameter
        float *pmmlon; //!< Min and max longitude
        float *pmmlat; //!< Min and max latitude
        float *prrr; //!< Projected radius on the Earth plane of the sky
        int *pneonly; //!< Flag to set for Ne only calculation
        int *proi; //!< Region of interest map
        int *ppofinteg; //!< Force integration centered on a plane of integration
        float *ppoiang; //!< Plane of integration angle orientation
        float *Hlonlat; //!< Heliographic lon and lat of disk center in rad
        float *poccrad; //!< Radius of the occulter.
        float *padapthres; //!< Adaptative Simpson's integration difference threshold. No Simpson integration if = 0. 
        int *pmaxsubdiv; //!< Adaptative Simpson's integration maximun number of interval subdivision recursion. 
        float *plimbdark; //!< User limb darkening: default 0.58 
        float *protmat; //!< output final Ne rotation matrix
        float *obslonlat; //!< observer lon lat and height in carrington coordinate
        int *obslonlatflag; //!< set to 1 if user defined obslonlat: then obspos is ignored
        int *projtypecode; //!< type of projection: 1:ARC, 2:TAN, 3:SIN, 4:AZP
        float *pv2_1; //!< mu parameter for the AZP projection
        int *pfrontinteg; //!< Integration centered on the observer
        unsigned int uvinteg; //!< Use UV emission instead of Thomson scattering
        float disttofracmax; //!< Set to the fraction of max B to evaluate the integration distance to that threshold: the distance is returned in bpol
        float *pnerotcntr; //!< Electron density center positon within the nps coord system defined by nepos and neang [Rsun]
        float *pnerotang; //!< Electron density orientation within the nps coord system defined by nepos and neang [rad]
        float *pnetranslation; //!< Shift of the electron density, in the density coordinate system
        int *pnerotaxis; //!< Electron density axis ids corresponding to the nerotang rotation angles

    Special thanks to:
        https://torroja.dmt.upm.es/media/files/ctypes.pdf
        https://www.youtube.com/watch?v=p_LUzwylf-Y&ab_channel=C%2FC%2B%2BDublinUserGroup
        https://stackoverflow.com/questions/38839525/ctypes-to-int-conversion-with-variable-assigned-by-reference
        https://blog.opencore.ch/posts/python-cpp-numpy-zero-copy/        
    """
    # Constants (system dependant)
    
    exec_path = os.getcwd()
    #test_save_file = exec_path+'/rtraytracewcs_wrapper_test_input.sav' # used to
    test_save_file = '/gehme/projects/2020_gcs_with_ml/data/gcs_idl/input_flor_a.sav'
    os.environ['RT_PATH'] = '/usr/local/ssw/stereo/secchi'
    os.environ['RT_SOFILENAME'] = 'libraytrace.so'
    os.environ['RT_SOTHREADFILENAME'] = 'libraytracethread.so'
    os.environ['RT_SOEXTENSION'] = 'so'
    os.environ['RT_RUNFROM'] = 'local'
    os.environ['RT_DATAPATH'] = '/usr/local/ssw/stereo/secchi/data/scraytrace'
    os.environ['SSW'] = '/usr/local/ssw'
    os.environ['RT_SOSUBPATH'] = 'lib/linux/x86_64'
    os.environ['RT_FORCELIBFILE'] = ' '
    os.environ['RT_LIBFILE'] = '/usr/local/ssw/stereo/secchi/lib/linux/x86_64/libraytrace.so'
    os.environ['RT_FORCELIBTHREAD'] = ''
    os.environ['RT_LIBFILETHREAD'] = ''
    os.environ['RT_UVEMISSIONPATH'] =  '/usr/local/ssw/stereo/secchi/cpp/scraytrace/data' #added on 222.12.26
    
    # importing libraytrace.so from C++
    c_lib = load_library('libraytrace.so', '/usr/local/ssw/stereo/secchi/lib/linux/x86_64/')

   

    # test case, reads input_data from IDL save file of an example
    if test:
        input_data = readsav(test_save_file, python_dict=True)
        file = '/gehme/projects/2020_gcs_with_ml/data/test_true.pickle'
    else:
        file = '/gehme/projects/2020_gcs_with_ml/data/test_false.pickle'
        
    # with open(file, 'wb') as handle:
    #     pickle.dump(input_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # with open(file, 'rb') as handle:
    #     data = pickle.load(handle)

    if compare:
        input_data_sav = readsav(test_save_file, python_dict=True)
        for i in input_data:
            if np.any(input_data[i] != input_data_sav[i]):
                print(i," es distinto")
                print("input data: ",input_data[i],type(input_data[i]))
                print("input data sav: ",input_data_sav[i], type(input_data_sav[i]))
        breakpoint()

    # clase tipo estructura con los tipo de punteros
    # queda más bonito y es más rápido cuando se llama a la función
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
    imsize_0 = cast(pointer(c_int32(input_data['imsize'][0])), c_void_p)
    imsize_1 = cast(pointer(c_int32(input_data['imsize'][1])), c_void_p)
    fovpix = cast(pointer(c_float(input_data['fovpix'])), c_void_p)
    obspos = cast(np.ctypeslib.as_ctypes(input_data['obspos'].astype('float32').copy()), c_void_p)
    obsang = cast(np.ctypeslib.as_ctypes(input_data['obsang'].astype('float32').copy()), c_void_p)
    nepos = cast(np.ctypeslib.as_ctypes(input_data['nepos'].astype('float32').copy()), c_void_p)
    neang = cast(np.ctypeslib.as_ctypes(input_data['neang'].astype('float32').copy()), c_void_p)
    losnbp = cast(pointer(c_int32(input_data['losnbp'])), c_void_p)
    losrange = cast(np.ctypeslib.as_ctypes(input_data['losrange'].astype('float32').copy()), c_void_p)
    modelid = cast(pointer(c_int32(input_data['modelid'])), c_void_p)
    btot = cast(np.ctypeslib.as_ctypes(input_data['btot'].astype('float32').copy()), c_void_p)
    bpol = cast(np.ctypeslib.as_ctypes(input_data['bpol'].astype('float32').copy()), c_void_p)
    netot = cast(np.ctypeslib.as_ctypes(input_data['netot'].astype('float32').copy()), c_void_p)
    modparam = cast(np.ctypeslib.as_ctypes(input_data['modparam'].astype('float32').copy()), c_void_p)
    crpix = cast(np.ctypeslib.as_ctypes(input_data['crpix'].astype('float32').copy()), c_void_p)
    rho = cast(np.ctypeslib.as_ctypes(input_data['rho'].astype('float32').copy()), c_void_p)
    mmlon = cast(np.ctypeslib.as_ctypes(input_data['mmlon'].astype('float32').copy()), c_void_p)
    mmlat = cast(np.ctypeslib.as_ctypes(input_data['mmlat'].astype('float32').copy()), c_void_p)
    rrr = cast(np.ctypeslib.as_ctypes(input_data['rrr'].astype('float32').copy()), c_void_p)
    pofinteg = cast(pointer(c_int32(input_data['pofinteg'])), c_void_p)
    quiet = cast(pointer(c_int32(input_data['quiet'])), c_void_p)
    neonly = cast(pointer(c_int32(input_data['neonly'])), c_void_p)
    roi = cast(np.ctypeslib.as_ctypes(input_data['roi'].astype('int32').copy()), c_void_p)
    poiang = cast(np.ctypeslib.as_ctypes(input_data['poiang'].astype('float32').copy()), c_void_p)
    hlonlat = cast(np.ctypeslib.as_ctypes(input_data['hlonlat'].astype('float32').copy()), c_void_p)
    occrad = cast(pointer(c_float(input_data['occrad'])), c_void_p)
    adapthres = cast(pointer(c_float(input_data['adapthres'])), c_void_p)
    maxsubdiv = cast(pointer(c_int32(input_data['maxsubdiv'])), c_void_p)
    limbdark = cast(pointer(c_float(input_data['limbdark'])), c_void_p)
    rotmat = cast(np.ctypeslib.as_ctypes(input_data['rotmat'].astype('float32').copy()), c_void_p)
    obslonlat = cast(np.ctypeslib.as_ctypes(input_data['obslonlat'].astype('float32').copy()), c_void_p)
    obslonlatflag = cast(pointer(c_int32(input_data['obslonlatflag'])), c_void_p)  # ver
    projtypecode = cast(pointer(c_int32(input_data['projtypecode'])), c_void_p)
    pv2_1 = cast(pointer(c_double(input_data['pv2_1'])), c_void_p)  # ver
    pc = cast(np.ctypeslib.as_ctypes(input_data['pc'].astype('float32').copy()), c_void_p)
    frontinteg = cast(pointer(c_int32(input_data['frontinteg'])), c_void_p)
    uvinteg = cast(pointer(c_int32(input_data['uvinteg'])), c_void_p)
    nerotcntr = cast(np.ctypeslib.as_ctypes(input_data['nerotcntr'].astype('float32').copy()), c_void_p)
    nerotang = cast(np.ctypeslib.as_ctypes(input_data['nerotang'].astype('float32').copy()), c_void_p)
    netranslation = cast(np.ctypeslib.as_ctypes(input_data['netranslation'].astype('float32').copy()), c_void_p)
    nerotaxis = cast(np.ctypeslib.as_ctypes(input_data['nerotaxis'].astype('int32').copy()), c_void_p)

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
    
    c_lib.rtraytracewcs.restype = c_bool
    c_lib.rtraytracewcs.argtypes = [c_int, POINTER(InputStructure)]
    c_lib.rtraytracewcs(41, input_obj)
    

    # reads btot from mem and optionally pots it
    btot = np.ctypeslib.as_array((c_float * (input_data['imsize'][0]*input_data['imsize'][1])).from_address(input_obj.btot))
    btot = btot.newbyteorder('<')
    btot = np.array(np.reshape(btot, (input_data['imsize'][0], input_data['imsize'][1])))
    if test:
        os.makedirs(exec_path+'/../output', exist_ok=True)
        fig = plt.figure(figsize=(4,4), facecolor='black') 
        m = np.nanmean(btot)
        sd = np.nanstd(btot)
        plt.imshow(btot, vmax=m+3*sd, vmin=m-3*sd, origin='lower')
        plt.savefig(exec_path+'/../output/rtraytracewcs_wrapper_test_output_btot.png')
        plt.close(fig)

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
    
    return (btot)


if __name__ == "__main__":
    rtraytracewcs_wrapper(None, test=True)
