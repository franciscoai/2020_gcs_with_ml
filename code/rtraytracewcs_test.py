""" PURPOSE: 
Main raytracing routine that calls the C raytracing routine

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

from pyGCS_ import *
from curses import is_term_resized
import numpy as np
import math
import numpy.ma as ma
import datetime
import struct
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame
import sunpy
from sunpy.coordinates.ephemeris import get_horizons_coord
import sunpy.map
from sunpy.map.maputils import all_coordinates_from_map
from sunpy.sun.constants import radius as _RSUN
from datetime import datetime

#rtraytracewcs,sbta,sbpa,snea,modelid=54,imsize=sgui.imdispsize,losrange=sgui.losrange,modparam=mp,neang=neang,scchead=sgui.hdra,losnbp=sgui.losnbp,/progressonly

#sgui : returns a structure containing all the different parameters of the GUI.
#sgui.lon : longitude Carrington.
#sgui.lat : latitude.
#sgui.rot : tilt angle or rotation around the model axis of symmetry. 0 is parallel to the equator.
#sgui.han : half angle between the model's feet.
#sgui.hgt : height, in Rsun.
#sgui.rat : aspect ratio = k
#sigin=0.1
#sigout=0.1
#sgui.nel=100000.

def rtsccguicloud_calcneang(CMElon,CMElat,CMEtilt,carrlonshiftdeg=-0.0,carrstonyshiftdeg=0.0):
    return [CMElon+carrlonshiftdeg*!dtor,CMElat,CMEtilt]
def rtsccguicloud_calcfeetheight(height,k,ang): 
    return height*(1.-k)*cos(ang)/(1.+sin(ang))
neang = rtsccguicloud_calcneang(CMElon,CMElat,CMEtilt,carrlonshiftdeg,carrstonyshiftdeg)
height=rtsccguicloud_calcfeetheight(height,k,ang)
nel = 100000.
mp = [1.5,ang,height,k,nel,0.,0.,0.,0.1,0.1]
def rtraytracewcs(modelid=54, imsize=[512,512], losrange=[-10.,10.], modparam=mp, neang=neang, header, losnbp=[64], progressonly):
    pv2_1in = header['PV2_1']
    if header['INSTRUME']=='LASCO':
        flagsoho='SOHO'
        instr='c3'
    else:
        flagsoho=False
        instr='cor2'
    dateobs=header['DATEOBS']
    if not instr: instr=header['DETECTOR']
    if not secchiab and ~flagsoho:
        secchiab = header['OBSRVTRY'][-1]
        if obslonlat == 0:
            obslonlat=float(header['CRLN_OBS']*dtor),float(header['CRLT_OBS']*dtor),float(header['DSUN_OBS']/_RSUN.value)
            obslonlatflag=1
            obslonlatheaderflag=True
        if rollang == 0:
            rollang=0.
            rollangheaderflag=True
    if fovpix == 0:
        fovpix=2./64.*dtor
        flagfovpix=False
    else:
        fovpix=float(fovpix)
        flagfovpix=True
    if not obspos:
        obspos=[0.,0,-214]
        obsposflag=False
    else:
        obspos=float(obspos)
        obsposflag=True
    if not obsang:
        obsang=[0.,0,0]
        obsangflag=False
    else:
        obsang=float(obsang)
        obsangflag=True

    if not nepos:nepos=[0.,0,0]
    else: nepos=float(nepos)
    if not neang: neang=[0.,0,0]  
    else: neang=float(neang)
    if not nerotcntr: nerotcntr=[0.,0,0]
    else: nerotcntr=float(nerotcntr)
    if not nerotang: nerotang=[0.,0,0]
    else: nerotang=float(nerotang)
    if not nerotaxis: nerotaxis=[3,2,1] 
    else: nerotaxis=nerotaxis
    if not netranslation: netranslation=[0.,0,0]
    else: netranslation=float(netranslation)
    if losnbp==0: losnbp=64
    else: losnbp=losnbp
    if not losrange: losrange=[-3.2,3.2]
    else: losrange=float(losrange)
    if modelid==0: modelid=1
    else: modelid=modelid
    if modparam==0: modparam=0.
    else: modparam=float(modparam)
    if pofinteg==0: pofinteg=0
    else: pofinteg=pofinteg
    if frontinteg==0: frontinteg=0
    else: frontinteg=frontinteg
    if uvinteg==0: uvinteg=0
    else: uvinteg=uvinteg
    if quiet==0: quiet=0
    else: quiet=2
    if progressonly!=0 and quiet==0: quiet=1
    if neonly==0: neonly=0
    else: neonly=1
    if not roi: roi=lonarr(imsize[0],imsize[1])+1
    else:
        sroi=len(roi)
        if sroi[0]!=imsize[0] or sroi[1]!=imsize[1]:
        print('The ROI image must be the same size than the output image !')
        roi=roi
    if not poiang: poiang=[0.,0,0]
    else: poiang=float(poiang)
    if not hlonlat: hlonlat=[0.,0,0]
    else: hlonlat=float(hlonlat)
    if not secchiab: secchiab='A' 
    else:
        secchiab=secchiab.upper()
        if secchiab != 'A' and secchiab != 'B': print('secchiab keyword must be either ''A'' or ''B''')

    if occrad==0: occrad=0
    else: occrad=float(occrad)
    if adapthres==0: adapthres=0.
    else: adapthres=float(adapthres)
    if maxsubdiv==0: maxsubdiv=4
    else: maxsubdiv=maxsubdiv
    if limbdark==0: limbdark=0.58
    else: limbdark=float(limbdark)
    if nbthreads==0: nbthreads=0
    else: nbthreads=nbthreads
    if nbchunks==0: nbchunks=0
    else: nbchunks=nbchunks
    if obslonlat!=0 and not obslonlatflag:
        obslonlat=float(obslonlat)
        obspos=obslonlat[2]*[sin(obslonlat[1]), sin(obslonlat[0])*cos(obslonlat[1]),-cos(obslonlat[0])*cos(obslonlat[1])]
        obslonlatflag=1
    if not obslonlat:
        obslonlat=[-atan(obspos[1],obspos[2]), asin(obspos[1]/norm(obspos)), norm(obspos)]
        obslonlatflag=0
    if not dateobs: dateobs=''
    if rollang==0: rollang=0.

    xdr=keyword_set(xdr)

#instrument presets if requested
#rtgetinstrwcsparam extract pointing parameters from wcs header to initialize raytrace
    rtgetinstrwcsparam,instr,imsize,scchead,fovpix,crpix,obsangpreset,pc,projtypepreset=projtypepreset,pv2_1=pv2_1,rollang=rollang,crval=crval,pcin=pcin,flagfovpix=flagfovpix

    if not obsangflag and instr1=0: obsang=obsangpreset
    if pv2_1in1=0: pv2_1=pv2_1in
    if not pv2_1: pv2_1=0.

    #set projection type
    if not projtype:
        if not projtypepreset: projtype='ARC' 
        else:
            projtype=projtypepreset

    projtype= projtype.upper()
    match projtype:
        case 'ARC': projtypecode=1
        case 'TAN': projtypecode=2
        case 'SIN': projtypecode=3
        case 'AZP': projtypecode=4
        else: print('Bad projtype keyword !')

    #init the outputs
    btot=np.zeros(imsize[0],imsize[1])
    bpol=np.zeros(imsize[0],imsize[1])
    netot=np.zeros(imsize[0],imsize[1])
    rho=np.zeros(imsize[0],imsize[1])
    mmlon=np.zeros(2)
    mmlat=np.zeros(2)
    rrr=np.zeros(imsize[0],imsize[1])
    rotmat=np.zeros(3,3)

    #init environment variable
    rtinitenv

    #start raytracing
    starttime=datetime.now().strftime("%H:%M:%S")
    if nbthreads==0:
        #SAVE, rtraytracewcs,imsize[0],imsize[1],fovpix,obspos,obsang,nepos,neang,losnbp,losrange,modelid,btot,bpol,netot,modparam,crpix,rho,mmlon,mmlat,rrr,pofinteg,quiet,neonly,roi,poiang,hlonlat,occrad,adapthres,maxsubdiv,limbdark,rotmat,obslonlat,obslonlatflag,projtypecode,pv2_1,pc,frontinteg,uvinteg,nerotcntr,nerotang,netranslation,nerotaxis,/unload), FILENAMES = '/gehme/projects/2020_gcs_with_ml/repo/gcs_idl/arguments/inputs.sav'
        s=call_external(getenv('RT_LIBFILE'),$
                    'rtraytracewcs',$
                    imsize[0],imsize[1],$
                    fovpix,$
                    obspos,obsang,$
                    nepos,neang,$
                    losnbp,losrange,modelid,$
                    btot,bpol,netot,modparam,$
                    crpix,rho,mmlon,mmlat,rrr,pofinteg,quiet,neonly,$
                    roi,poiang,hlonlat,occrad,adapthres,maxsubdiv,limbdark,$
                    rotmat,obslonlat,obslonlatflag,projtypecode,pv2_1,pc,frontinteg,uvinteg,nerotcntr,nerotang,netranslation,nerotaxis,/unload)
        #SAVE, rtraytracewcs,imsize[0],imsize[1],fovpix,obspos,obsang,nepos,neang,losnbp,losrange,modelid,btot,bpol,netot,modparam,crpix,rho,mmlon,mmlat,rrr,pofinteg,quiet,neonly,roi,poiang,hlonlat,occrad,adapthres,maxsubdiv,limbdark,rotmat,obslonlat,obslonlatflag,projtypecode,pv2_1,pc,frontinteg,uvinteg,nerotcntr,nerotang,netranslation,nerotaxis,/unload), FILENAMES = '/gehme/projects/2020_gcs_with_ml/repo/gcs_idl/arguments/outputs.sav'
    else:
