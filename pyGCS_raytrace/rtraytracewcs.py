""" PURPOSE: 
Main raytracing routine that calls the python wrapper of the compiled c raytracing library (shared object)

@authors: F. A. Iglesias, Y. Machuca and F. Cisterna. 2023. Mendoza Group for Heloiphysics Studies (GEHMe), Argentina
Contact: franciscoaiglesias@gmail.com

CATEGORY:
raytracing, simulation, 3d
INPUTS:
 all the distances must be given in Rsun
 and the angles in rad
 imsize : [xs,ys] size of the output image
 fovpix : fov angle of one pixel in rad 
 -- observer position and attitude
 obspos : [x,y,z] position of the observer in the Sun basis
 obslonlat : [lon,lat,height] position of the observer in Carrington
             coordinate. If set, then obspos is ignored. The optical
             axis always points toward the Sun center. Use obsang to
             change telescope orientation. Note that obslonlat=[0,0,215]
             correspond to obspos=[0,0,215] and obsang=[!pi,0,0]: this
             means that the Carrington coordinate origin on the Solar
             sphere (lon,lat,height)=(0,0,1) is located at (x,y,z)=(0,0,1), with
             Ox pointing to solar north and Oy pointing to (lon,lat)=(3*!pi/2,0)
 obsang : [ax,ay,az] orientation of the observer, 
          z is the optical axis 
 rollang : allow to set the roll angle of the virtual instrument. 
           Works only if a preset instrument is requested.
 -- Ne position and attitude
 nepos : [x,y,z] position of the Ne reference in the Sun basis
 neang : [ax,ay,az] orientation of the Ne
 nerotcntr : [x,y,z] center of rotation of the Ne model, in the Ne basis
 nerotang : [ax,ay,az] rotation of the Ne model around the nerotcntr, in the Ne basis
 nerotaxis : [axid1,axid2,axid3] axis id corresponding to the nerotang rotation angles. 1: X, 2: Y, 3: Z. Default is [3,2,1].
 netranslation : [tx,ty,tz] translation vector of the Ne model, in the Ne basis
 -- POI (central Plane of Integration) orientation
 poiang : [ax,ay,az] orientation of the POI z axis: note that az
          rotation has no effect.
 -- LOS params
 losnbp : number of step for the integration along the LOS
 losrange : [lstart,lend] range for the integration along the LOS
            in Rsun. The origin of the LOS is the orthogonal
            projection of the Sun cntr on that LOS.
 modelid : model id number
 modparam : parameters of the model
 save : put path and filename in that variable (without extention) 
        if you want to save the results in a .fits binary table.
 fakelasco : put fake lasco header information in the fits header
 pofinteg : the raytracing LOS center is taken in the plane of the sky
          containing the Sun center instead of the Sun center 
          projection on the LOS (impact distance projection)
 frontinteg : set so that the origin of the LOS is taken at the observer: 
              if used, the losrange parameters must both be positive.
 uvinteg : use UV emission instead of thomson scattering. If used, the 
           model selected has to return a temperature in addition to the electron density.
           Value should be [1,2,3, or 4] for O VI 1032, Si XII 499, LyB, and Fe XVIII 974
 quiet : disable display of raytracing parameters
 neonly : set to compute only the Ne along the LOS
 roi : region of interest map: int image same size than the requested
       output image. 0 pixels won't be calculated to speed up.
 hlonlat : [Hlon,Hlat,Hrot] heliographic lon and lat of the center of
 the disk, rotation angle corresponding to the projection of the
 north pole, counterclockwise
 secchiab : 'A' or 'B', to select Ahead or Behind spacecraft, for
            secchi only
 occrad : occulter radius. The integration in not performed within
          that disk. [Rsun]
 adapthres : adapthres=maxdiff [Ne]: Set to allow adaptative simpson
             integration. Set to the maximum difference allowed
             between two consecutive samples. If the difference is
             bigger then the algorithm will subdivide the interval
             until the difference falls below the limit.
 maxsubdiv : only with adapthres: maximum recursive subdivision of an
             interval. Stop subdivision in case that maximum number
             of recusion is met. (default : 4)
 xdr : save into xdr format instead of fits table. 'save' keyword
       must be set for xdr to take effect.
 projtype : projection type: (see Calabretta and Greisen,
            Representations of celestial coordinates in FITS, A&A
            395, 1077-1122(2002))
             ARC : Zenithal equidistant (default)
             TAN : Gnomonic
             SIN : Slant orthographic
             AZP : Zenithal perspective
            If an instrument preset is requested then this keyword
            will overwrite the projection type of the selected
            instrument.
 pv2_1 : mu parameter for the AZP projection
 pcin : force the fits PCi_j matrix. Must be a 4 elements array

 dateobs : observation date that will be copied in the image header
           and used to compute the observer position in the different
           coordinate systems.
 instr : txt instrument preset, to select from the list above:
 scchead : secchi structure header: raytrace will use the
           positionning info of the header to generate the view
 progessonly : show only the computation progression if set. No
               effect if quiet is set.
 nbthreads : [default = 0] set to the number of processors you want 
             to use in parallel to speed up the computation. This is only useful 
             if you have a multi-core processor. Note that the following
             keywords are not used if you use nbthreads: rho,mmlon,mmlat,rr,rotmat,
             adapthres, maxsubdiv, roi, uvinteg, pofinteg, poiang.
 nbchunks : [default = 0] use with nbthread. If set to a value less than 2, the threads are 
           launched by lines of sight. If nbchunks >= 2, the threads are launched by chunk
           of the image. Ballancing nbthreads and nbchunks allow optimizing the performances.

 -- Instrument FOV preset
 c1, c2, c3 : lasco C1, C2, C3
 cor1, cor2 : Secchi Cor1, Cor2
 hi1, hi2 : Secchi Hi1, Hi2
 limbdark : limb darkening coeff: default 0.58
 usedefault : we use the default parameters for the selected model.

 OUTPUTS:
  sbtot : structure with image of the total brightness
  sbpol : polarized brightness
  snetot : integrated electron density along the LOSes
  rho : impact parameter for each LOS
  mmlon : min and max longitude
  mmlat : min and max latitude
  rrr : dist instersection LOS - plane of the sky containing the Sun cntr
  rotmat : final rotation matrix of the density cube """

# adds parent dir to syspath
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pyGCS_raytrace.rtraytracewcs_wrapper import rtraytracewcs_wrapper
import numpy as np
import math
from ctypes import *

def rtsccguicloud_calcneang(CMElon, CMElat, CMEtilt, carrlonshiftdeg= -0.0000000, carrstonyshiftdeg= 0.00000):
    return np.array([CMElon + carrlonshiftdeg, CMElat, CMEtilt], dtype='float32')

def rotatemat(crval, x): #rotmat.pro
    if x==1:#X
        r = np.array([[1.,0.,0.], 
                        [0.,math.cos(crval),-math.sin(crval)], 
                        [0.,math.sin(crval),math.cos(crval)]]).transpose()
    if x==2:#Y
        r = np.array([[math.cos(crval),0.,math.sin(crval)], 
                        [0.,1.,0.], 
                        [-math.sin(crval),0.,math.cos(crval)]]).transpose()
    if x==3:#Z
        r = np.array([[math.cos(crval),-math.sin(crval),0.], 
                        [math.sin(crval),math.cos(crval),0.], 
                          [0.,0.,1]]).transpose()
    return r

def rtrotmat2rxryrz(r):
    ry = math.asin(r[2,0])
    c1 = math.cos(ry)
    if (abs(c1) < 1.0e-6):
        rz = -math.atan2(r[1,2], r[0,2])
        rx = 0.0
    else:
        rz = -math.atan2(r[1,0], r[0,0])
        rx = -math.atan2(r[2,1], r[2,2])
    return np.array([rx, ry, rz])

def piximchangereso(pixim,reso):
    pixccd = pixim * ( 2.**reso) + (2.**reso)/2.
    pixsidesize = 1.
    return pixccd/pixsidesize - 0.5

def arcsec2rad(x):
#arseconds to radians convertion
    return math.radians(x/3600.) 

def rtsccguicloud_calcfeetheight(height,k,ang):
    # convert from distance of nose to Thernisien h (length of leg)
    return height*(1.-k)*np.cos(ang)/(1.+np.sin(ang))


def rtraytracewcs(header, CMElon=60, CMElat=20, CMEtilt=70, height=6, k=3, ang=30, in_sig=0.1, out_sig=0.1, nel=1e5, modelid=54, imsize=np.array([512, 512], dtype='int32'), occrad=0, losrange=np.array([-10., 10.], dtype='float32'), losnbp=64):
    obslonlatflag = 1
    CMElon = math.radians(CMElon)
    CMElat = math.radians(CMElat)
    CMEtilt = math.radians(CMEtilt)
    ang = math.radians(ang)
    
    neang = rtsccguicloud_calcneang(CMElon, CMElat, CMEtilt)
    leg_height = rtsccguicloud_calcfeetheight(height, k, ang)

    modparam = np.array([1.5, ang, leg_height, k, nel, 0., 0., 0., in_sig, out_sig], dtype='float32')
    try:
        pv2_1 = header['PV2_1']
    except:
        pv2_1 = 0.0

    obslonlat = np.array([math.radians(header['CRLN_OBS']),math.radians(header['CRLT_OBS']),1./arcsec2rad(header['RSUN'])], dtype='float32') # carr lon, lat, distance from Sun center in Sr
    rollang = 0.00000
    imszratio = header['NAXIS1']/imsize[0]
    fovpix = np.array(arcsec2rad((header['CDELT1']+header['CDELT2'])*imszratio/2.), dtype='float32') # mean plate scale # np.array(math.radians(1./64.), dtype='float32')
    obspos = np.array([0., 0, -214], dtype='float32')
    rmat = rotatemat(arcsec2rad(header['CRVAL2']),2) @ rotatemat(arcsec2rad(-header['CRVAL1']),1) @ rotatemat(arcsec2rad(-rollang),3)
    obsang = rtrotmat2rxryrz(rmat)
    nepos = np.array([0., 0, 0], dtype='float32')
    nerotcntr = np.array([0., 0, 0], dtype='float32')
    nerotang = np.array([0., 0, 0], dtype='float32')
    nerotaxis = np.array([3, 2, 1], dtype='int32')
    netranslation = np.array([0., 0, 0], dtype='float32')
    pofinteg = 0
    frontinteg = 0
    uvinteg = 0
    quiet = 1
    neonly = 0
    roi = np.ones((imsize[0], imsize[1]), dtype='int32')
    poiang = np.array([0, 0, 0], dtype='float32')
    hlonlat = np.array([0., 0, 0], dtype='float32')
    adapthres = 0.
    maxsubdiv = 4
    limbdark = 0.58
    crpix = piximchangereso(np.array([header['CRPIX1'] -1, header['CRPIX2'] -1], dtype='float32'),-math.log(imszratio)/math.log(2) )
    pc = np.array([1, 8.687118e-14, -8.687118e-14, 1], dtype='float32')

    # set projection type
    projtypecode = 2

    # init the outputs
    btot = np.zeros((imsize[0], imsize[1]), dtype='float32')
    bpol = np.zeros((imsize[0], imsize[1]), dtype='float32')
    netot = np.zeros((imsize[0], imsize[1]), dtype='float32')
    rho = np.zeros((imsize[0], imsize[1]), dtype='float32')
    mmlon = np.zeros(2, dtype='float32')
    mmlat = np.zeros(2, dtype='float32')
    rrr = np.zeros((imsize[0], imsize[1]), dtype='float32')
    rotmat = np.zeros((3, 3), dtype='float32')

    # assebmles rtraytrace input
    data_input = {'imsize': imsize,
                  'fovpix': fovpix,
                  'obspos': obspos,
                  'obsang': obsang,
                  'nepos': nepos,
                  'neang': neang,
                  'losnbp': losnbp,
                  'losrange': losrange,
                  'modelid': modelid,
                  'btot': btot,
                  'bpol': bpol,
                  'netot': netot,
                  'modparam': modparam,
                  'crpix': crpix,
                  'rho': rho,
                  'mmlon': mmlon,
                  'mmlat': mmlat,
                  'rrr': rrr,
                  'pofinteg': pofinteg,
                  'quiet': quiet,
                  'neonly': neonly,
                  'roi': roi,
                  'poiang': poiang,
                  'hlonlat': hlonlat,
                  'occrad': occrad,
                  'adapthres': adapthres,
                  'maxsubdiv': maxsubdiv,
                  'limbdark': limbdark,
                  'rotmat': rotmat,
                  'obslonlat': obslonlat,
                  'obslonlatflag': obslonlatflag,
                  'projtypecode': projtypecode,
                  'pv2_1': pv2_1,
                  'pc':pc,
                  'frontinteg': frontinteg,
                  'uvinteg': uvinteg,
                  'nerotcntr': nerotcntr,
                  'nerotang': nerotang,
                  'netranslation': netranslation,
                  'nerotaxis': nerotaxis
                  }

    # calls rtraytrace wrapper   
    return (rtraytracewcs_wrapper(data_input, test=False))
