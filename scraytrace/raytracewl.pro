
; ---- wrapper that calls the C raytracing program

pro raytracewl,sbtot,sbpol,snetot,imsize=imsize,$
               fovpix=fovpix,$
               obspos=obspos,obsang=obsang,$
               nepos=nepos,neang=neang,$
               losnbp=losnbp,losrange=losrange,$
               modelid=modelid,modparam=modparam,$
               save=save,fakelasco=fakelasco,$
               c2image=c2image,c3image=c3image,$
               rho=rho,mmlon=mmlon,mmlat=mmlat,rrr=rrr,$
               pofinteg=pofinteg,quiet=quiet,$
               cor1=cor1,cor2=cor2,hi1=hi1,hi2=hi2,c1fov=c1fov,$
               neonly=neonly,roi=roi,poiang=poiang,$
               hlonlat=hlonlat,secchiab=secchiab,occrad=occrad,$
               adapthres=adapthres,maxsubdiv=maxsubdiv,xdr=xdr,$
               instr=instr,limbdark=limbdark,rotmat=rotmat,$
               usedefault=usedefault,obslonlat=obslonlat,$
               projtype=projtype,dateobs=dateobs,rollang=rollang,$
               scchead=scchead,pv2_1=pv2_1,frontinteg=frontinteg,$
               uvinteg=uvinteg,disttofracmaxinteg=disttofracmaxinteg,$
               nbthreads=nbthreads,pc=pc,nbchunks=nbchunks,$
               nerotcntr=nerotcntr,nerotang=nerotang,nerotaxis=nerotaxis,netranslation=netranslation,$
               time=time,physics=physics,phyparam=phyparam

;+
;  $Id: raytracewl.pro,v 1.18 2010/09/08 15:58:04 thernis Exp $
; 
; PURPOSE: 
;  Main raytracing routine that calls the C raytracing routine
;
; CATEGORY:
;  raytracing, simulation, 3d
;
; INPUTS:
; ! all the distances must be given in Rsun
; ! and the angles in rad
; imsize : [xs,ys] size of the output image
; fovpix : fov angle of one pixel in rad 
; -- observer position and attitude
; obspos : [x,y,z] position of the observer in the Sun basis
; obslonlat : [lon,lat,height] position of the observer in Carrington
;             coordinate. If set, then obspos is ignored. The optical
;             axis always points toward the Sun center. Use obsang to
;             change telescope orientation. Note that obslonlat=[0,0,215]
;             correspond to obspos=[0,0,215] and obsang=[!pi,0,0]: this
;             means that the Carrington coordinate origin on the Solar
;             sphere (lon,lat,height)=(0,0,1) is located at (x,y,z)=(0,0,1), with
;             Ox pointing to solar north and Oy pointing to (lon,lat)=(3*!pi/2,0)
; obsang : [ax,ay,az] orientation of the observer, 
;          z is the optical axis 
; rollang : allow to set the roll angle of the virtual instrument. 
;           Works only if a preset instrument is requested.
; -- Ne position and attitude
; nepos : [x,y,z] position of the Ne reference in the Sun basis
; neang : [ax,ay,az] orientation of the Ne
; nerotcntr : [x,y,z] center of rotation of the Ne model, in the Ne basis
; nerotang : [ax,ay,az] rotation of the Ne model around the nerotcntr, in the Ne basis
; nerotaxis : [axid1,axid2,axid3] axis id corresponding to the nerotang rotation angles. 1: X, 2: Y, 3: Z. Default is [3,2,1].
; netranslation : [tx,ty,tz] translation vector of the Ne model, in the Ne basis
; -- POI (central Plane of Integration) orientation
; poiang : [ax,ay,az] orientation of the POI z axis: note that az
;          rotation has no effect.
; -- LOS params
; losnbp : number of step for the integration along the LOS
; losrange : [lstart,lend] range for the integration along the LOS
;            in Rsun. The origin of the LOS is the orthogonal
;            projection of the Sun cntr on that LOS.
; modelid : model id number
; modparam : parameters of the model
; save : put path and filename in that variable (without extention) 
;        if you want to save the results in a .fits binary table.
; fakelasco : put fake lasco header information in the fits header
; pofinteg : the raytracing LOS center is taken in the plane of the sky
;          containing the Sun center instead of the Sun center 
;          projection on the LOS (impact distance projection)
; frontinteg : set so that the origin of the LOS is taken at the observer: 
;              if used, the losrange parameters must both be positive.
; uvinteg : use UV emission instead of thomson scattering. If used, the 
;           model selected has to return a temperature in addition to the electron density.
;           Value should be [1,2,3, or 4] for O VI 1032, Si XII 499, LyB, and Fe XVIII 974
; disttofracmaxinteg : set to the fraction of the maximum total B per LOS in order to 
;                      compute the distance to that fraction of brightness. Disabled if set to 0 [default].
;                      The distance is returned in the bpol image, in Rsun.
; quiet : disable display of raytracing parameters
; neonly : set to compute only the Ne along the LOS
; roi : region of interest map: int image same size than the requested
;       output image. 0 pixels won't be calculated to speed up.
; hlonlat : [Hlon,Hlat,Hrot] heliographic lon and lat of the center of
; the disk, rotation angle corresponding to the projection of the
; north pole, counterclockwise
; secchiab : 'A' or 'B', to select Ahead or Behind spacecraft, for
;            secchi only
; occrad : occulter radius. The integration in not performed within
;          that disk. [Rsun]
; adapthres : adapthres=maxdiff [Ne]: Set to allow adaptative simpson
;             integration. Set to the maximum difference allowed
;             between two consecutive samples. If the difference is
;             bigger then the algorithm will subdivide the interval
;             until the difference falls below the limit.
; maxsubdiv : only with adapthres: maximum recursive subdivision of an
;             interval. Stop subdivision in case that maximum number
;             of recusion is met. (default : 4)
; xdr : save into xdr format instead of fits table. 'save' keyword
;       must be set for xdr to take effect.
; projtype : projection type: (see Calabretta and Greisen,
;            Representations of celestial coordinates in FITS, A&A
;            395, 1077-1122(2002))
;             ARC : Zenithal equidistant (default)
;             TAN : Gnomonic
;             SIN : Slant orthographic
;             AZP : Zenithal perspective
;            If an instrument preset is requested then this keyword
;            will overwrite the projection type of the selected
;            instrument.
; pv2_1 : mu parameter for the AZP projection
; pc : wcs pc[2,2] matrix: default is unit matrix
; dateobs : observation date that will be copied in the image header
;           and used to compute the observer position in the different
;           coordinate systems.
; instr : txt instrument preset, to select from the list above:
; scchead : secchi structure header: raytrace will use the
;           positionning info of the header to generate the view
; nbthreads : [default = 0] set to the number of processors you want 
;             to use in parallel to speed up the computation. This is only useful 
;             if you have a multi-core processor. Note that the following
;             keywords are not used if you use nbthreads: rho,mmlon,mmlat,rr,rotmat,
;             adapthres, maxsubdiv, roi, uvinteg, pofinteg, poiang.
; nbchunks : [default = 0] use with nbthread. If set to a value less than 2, the threads are 
;           launched by lines of sight. If nbchunks >= 2, the threads are launched by chunk
;           of the image. Ballancing nbthreads and nbchunks allow optimizing the performances.
; physics : type of physics to perform the raytracing:
;           0 : Thomson scattering (default), 1 : UV emission, 2 : F Corona (experimental)
;           for the UV emission and F Corona, phyparam have to be passed too. See C source code for more info.
; phyparam : extra parameters required by the chosen physics.
;
; -- Instrument FOV preset
; c1, c2, c3 : lasco C1, C2, C3
; cor1, cor2 : Secchi Cor1, Cor2
; hi1, hi2 : Secchi Hi1, Hi2
; limbdark : limb darkening coeff: default 0.58
; usedefault : we use the default parameters for the selected model.
;
; OUTPUTS:
;  sbtot : structure with image of the total brightness
;  sbpol : polarized brightness
;  snetot : integrated electron density along the LOSes
;  rho : impact parameter for each LOS
;  mmlon : min and max longitude
;  mmlat : min and max latitude
;  rrr : dist instersection LOS - plane of the sky containing the Sun cntr
;  rotmat : final rotation matrix of the density cube
; 
; CALL EXEMPLE:
;
;raytracewl,sbt,imsize=[256,256],modelid=54,/usedefault,losrange=[-15,15],losnbp=128L,neang=[60.,20.,0]*!dtor,/cor2
;wnd,0,alog10(sbt.im >1e-14)
;
;-


; ---- define keyword default values
;      and cast every inputs to avoid C crash with bad type entry

; -- use header information if passed by user
obslonlatheaderflag=0B
;pv2_1headerflag=0B
rollangheaderflag=0B

if n_elements(pv2_1) ne 0 then pv2_1in=float(pv2_1)

if n_elements(scchead) ne 0 then begin
    if scchead.instrume ne 'SECCHI' then message,'The header passed in scchead should be a SECCHI header.'

    wcs=fitshead2wcs(scchead)

    if n_elements(dateobs) eq 0 then dateobs=scchead.date_obs
    if n_elements(instr) eq 0 then instr=scchead.detector
    if n_elements(secchiab) eq 0 then secchiab=strmid(scchead.obsrvtry,0,1,/reverse_offset)

    if n_elements(obslonlat) eq 0 then begin 
        obslonlat=[wcs.position.crln_obs*!dtor,wcs.position.crlt_obs*!dtor,scchead.dsun_obs/(onersun()*1e3)]
        obslonlatheaderflag=1B
    endif
    if n_elements(pv2_1) eq 0 then begin
        pv2_1=float(scchead.pv2_1)
        ;pv2_1headerflag=1b
    endif

    if n_elements(pc) eq 0 then pc=float(wcs.pc)

    if n_elements(rollang) eq 0 then begin
        rollang=scchead.crota*!dtor
        rollangheaderflag=1B
    endif

endif

if n_elements(pc) eq 0 then pc=float(identity(2)) else pc=float(pc)

if n_elements(imsize) eq 0 then imsize=[64L,64] else imsize=long(imsize)
if n_elements(fovpix) eq 0 then begin
	fovpix=2./(64.)*!dtor
	flagfovpixin=0b
endif else begin
	fovpix=float(fovpix)
	flagfovpixin=1b
	fovpixin=fovpix
endelse
if n_elements(obspos) eq 0 then obspos=[0.,0,-214] else obspos=float(obspos)
if n_elements(obsang) eq 0 then begin
    obsang=[0.,0,0] 
    obsangflag=0b
endif else begin
    obsang=float(obsang)
    obsangflag=1b
endelse
if n_elements(nepos) eq 0 then nepos=[0.,0,0] else nepos=float(nepos)
if n_elements(neang) eq 0 then neang=[0.,0,0] else neang=float(neang)
if n_elements(nerotcntr) eq 0 then nerotcntr=[0.,0,0] else nerotcntr=float(nerotcntr)
if n_elements(nerotang) eq 0 then nerotang=[0.,0,0] else nerotang=float(nerotang)
if n_elements(nerotaxis) eq 0 then nerotaxis=long([3,2,1]) else nerotaxis=long(nerotaxis)
if n_elements(netranslation) eq 0 then netranslation=[0.,0,0] else netranslation=float(netranslation)
if n_elements(losnbp) eq 0 then losnbp=64L else losnbp=long(losnbp)
if n_elements(losrange) eq 0 then losrange=[-3.2,3.2] else losrange=float(losrange)
if n_elements(modelid) eq 0 then modelid=1L else modelid=long(modelid)
if n_elements(modparam) eq 0 then modparam=0. else modparam=float(modparam)
if n_elements(pofinteg) eq 0 then pofinteg=0L else pofinteg=long(pofinteg)
if n_elements(frontinteg) eq 0 then frontinteg=0L else frontinteg=long(frontinteg)
if n_elements(uvinteg) eq 0 then uvinteg=0L else uvinteg=long(uvinteg)
if n_elements(disttofracmaxinteg) eq 0 then disttofracmaxinteg=0. else disttofracmaxinteg=float(disttofracmaxinteg)
if n_elements(quiet) eq 0 then quiet=0L else quiet=1L
if n_elements(neonly) eq 0 then neonly=0L else neonly=1L
if n_elements(roi) eq 0 then roi=lonarr(imsize[0],imsize[1])+1 else begin
    sroi=size(roi,/dim)
    if sroi[0] ne imsize[0] or sroi[1] ne imsize[1] then begin
        message,'The ROI image must be the same size than the output image !'
    endif
    roi=long(roi)
endelse
if n_elements(poiang) eq 0 then poiang=[0.,0,0] else poiang=float(poiang)
if n_elements(hlonlat) eq 0 then hlonlat=[0.,0,0] else hlonlat=float(hlonlat)
if n_elements(secchiab) eq 0 then secchiab='A' else begin
    secchiab=strupcase(secchiab)
    if secchiab ne 'A' and secchiab ne 'B' then message,'secchiab keyword must be either ''A'' or ''B'''
endelse
if n_elements(occrad) eq 0 then occrad=0. else occrad=float(occrad)
if n_elements(adapthres) eq 0 then adapthres=0. else adapthres=float(adapthres)
if n_elements(maxsubdiv) eq 0 then maxsubdiv=4L else maxsubdiv=long(maxsubdiv)
if n_elements(limbdark) eq 0 then limbdark=0.58 else limbdark=float(limbdark)
if n_elements(nbthreads) eq 0 then nbthreads=0l else nbthreads=long(nbthreads)
if n_elements(nbchunks) eq 0 then nbchunks=0l else nbchunks=long(nbchunks)
if n_elements(physics) eq 0 then physics=0l else physics=long(physics)
if n_elements(phyparam) eq 0 then phyparam=0. else phyparam=float(phyparam)

if n_elements(obslonlat) ne 0 then begin 
    obslonlat=float(obslonlat)
    obspos=obslonlat[2]*[sin(obslonlat[1]),$
                         sin(obslonlat[0])*cos(obslonlat[1]),$
                         -cos(obslonlat[0])*cos(obslonlat[1])]
    obslonlatflag=1L
endif else begin
    obslonlat=[-atan(obspos[1],obspos[2]),$
               asin(obspos[1]/norm(obspos)),$
               norm(obspos)]
    obslonlatflag=0L
endelse
if n_elements(dateobs) eq 0 then dateobs=''
if n_elements(rollang) eq 0 then rollang=0.

; ---- get the default parameters if requested
if keyword_set(usedefault) then begin
    getmoddefparam,modelid,sdefault
    modparam=parsemoddefparam(sdefault,svdefault,filepro='buildmodel'+strtrim(modelid,2)+'param.pro')
endif

; ---- detect instrument, if any
c2image=keyword_set(c2image)
if c2image then instr='c2'
c3image=keyword_set(c3image)
if c3image then instr='c3'
c1fov=keyword_set(c1fov)
if c1fov then instr='c1'
cor1=keyword_set(cor1)
if cor1 then instr='cor1'
cor2=keyword_set(cor2)
if cor2 then instr='cor2'
hi1=keyword_set(hi1)
if hi1 then instr='hi1'
hi2=keyword_set(hi2)
if hi2 then instr='hi2'
xdr=keyword_set(xdr)

; ---- instrument presets if requested
crval=[-obsang[0]*!radeg,0.]
rtinstrpresets,instr,imsize,fovpix,obsangpreset,secchiab=secchiab,projtypepreset=projtypepreset,pv2_1=pv2_1,scchead=scchead,rollang=rollang,crpixin=crpixin,crval=crval
if flagfovpixin then fovpix=fovpixin

; -- obsang and rollang is forced if passed by user
if not obsangflag and n_elements(instr) ne 0 then obsang=obsangpreset
if n_elements(pv2_1in) ne 0 then pv2_1=pv2_1in
if n_elements(pv2_1) eq 0 then pv2_1=0.
;if rollang ne 0 then obsang[2]=rollang

; ---- set projection type
if n_elements(projtype) eq 0 then begin
    if n_elements(projtypepreset) eq 0 then projtype='ARC' else projtype=projtypepreset
endif

projtype=strupcase(projtype)
case projtype of
    'ARC' : projtypecode=1L
    'TAN' : projtypecode=2L
    'SIN' : projtypecode=3L
    'AZP' : projtypecode=4L
    else : message,'Bad projtype keyword !'
endcase


; ---- init the outputs
btot=fltarr(imsize[0],imsize[1])
bpol=fltarr(imsize[0],imsize[1])
netot=fltarr(imsize[0],imsize[1])
rho=fltarr(imsize[0],imsize[1])
mmlon=fltarr(2)
mmlat=fltarr(2)
rrr=fltarr(imsize[0],imsize[1])
rotmat=fltarr(3,3)

crpix=fltarr(2)

; -- init environment variables
rtinitenv
; ---- start raytracing
starttime=systime(1)
if nbthreads eq 0 then begin
s=call_external(getenv('RT_LIBFILE'),$
                'raytracewl',$
                imsize[0],imsize[1],$
                fovpix,$
                obspos,obsang,$
                nepos,neang,$
                losnbp,losrange,modelid,$
                btot,bpol,netot,modparam,$
                crpix,rho,mmlon,mmlat,rrr,pofinteg,quiet,neonly,$
                roi,poiang,hlonlat,occrad,adapthres,maxsubdiv,limbdark,$
                rotmat,obslonlat,obslonlatflag,projtypecode,pv2_1,frontinteg,uvinteg,disttofracmaxinteg,nerotcntr,nerotang,netranslation,nerotaxis,/unload)
endif else begin
; ---- case of nbthreads > 0
;      See User''s Guide for more instructions using this feature.
crpix=crpixin
s=call_external(getenv('RT_LIBFILETHREAD'),$
                'rtthreadidl',$
                imsize[0],imsize[1],$
                fovpix,$
                obspos,obsang,$
                nepos,neang,$
                losnbp,losrange,modelid,$
                btot,bpol,netot,modparam,$
                crpix,quiet,neonly,$
                hlonlat,occrad,limbdark,$
                obslonlat,obslonlatflag,projtypecode,pv2_1,pc,frontinteg,$
                nbthreads,nbchunks,$
                nerotcntr,nerotang,netranslation,nerotaxis,physics,phyparam,/unload)
endelse
time=systime(1)-starttime
if quiet eq 0 then begin
    print,'Seconds ellapsed :'
    print,time
endif

; -- sun radius in pix 
solar_r=abs(atan(1./obspos[2]))/fovpix

; ---- fill in the fits header
mkhdr,h0,btot
sxaddpar,h0,'COMMENT','scraytrace'
sxaddpar,h0,'DESC', 'model '+strtrim(modelid,2)
sxaddpar,h0,'AUTHOR','scraytrace'
sxaddpar,h0,'PROGRAM','scraytrace'
sxaddpar,h0,'DATE',anytim(systim(),/ccsds)

case 1 of
    c2image : rtdetec='c2'
    c3image : rtdetec='c3'
    c1fov : rtdetec='c1'
    cor1 : rtdetec='cor1'
    cor2 : rtdetec='cor2'
    hi1 : rtdetec='hi1'
    hi2 : rtdetec='hi2'
    else : rtdetec=''
endcase
sxaddpar,h0,'RTDETEC',rtdetec,'raytrace detector option'
sxaddpar,h0,'DETECTOR',rtdetec 
case 1 of
    rtdetec eq 'c1' or rtdetec eq 'c2' or rtdetec eq 'c3' : begin
        sxaddpar,h0,'obsrvtry','LASCO'
        break
    end
    rtdetec eq '' : sxaddpar,h0,'obsrvtry',''
    else : sxaddpar,h0,'obsrvtry','STEREO_'+secchiab
endcase


; ---- WCS compliant header here
sxaddpar,h0,'WCSAXES',2
sxaddpar,h0,'WCSname','Helioprojective-Cartesian'

sxaddpar,h0,'CRPIX1',crpix[0]+1,'X Sun center (pix) (FORTRAN convention)'
sxaddpar,h0,'CRPIX2',crpix[1]+1,'Y Sun center (pix)'

sxaddpar,h0,'CTYPE1','HPLN-'+projtype
sxaddpar,h0,'CTYPE2','HPLT-'+projtype

sxaddpar,h0,'PV2_1',pv2_1

sxaddpar,h0,'CUNIT1','deg'
sxaddpar,h0,'CUNIT2','deg'

sxaddpar,h0,'CDELT1', fovpix*!radeg
sxaddpar,h0,'CDELT2', fovpix*!radeg

sxaddpar,h0,'crval1',crval[0]
sxaddpar,h0,'crval2',crval[1]


; -- WARNING: pci_j implementation not tested
sxaddpar,h0,'pc1_1',cos(obsang[2])
sxaddpar,h0,'pc2_1',sin(obsang[2])
sxaddpar,h0,'pc1_2',-sin(obsang[2])
sxaddpar,h0,'pc2_2',cos(obsang[2])

sxaddpar,h0,'SOLAR_R',solar_r,'pix'
sxaddpar,h0,'FOVPIX',fovpix,'rad'

sxaddpar,h0,'history','$Id: raytracewl.pro,v 1.18 2010/09/08 15:58:04 thernis Exp $'

sxaddpar,h0,'p1col',1
sxaddpar,h0,'p2col',imsize[1]
sxaddpar,h0,'p1row',1
sxaddpar,h0,'p2row',imsize[0]
sxaddpar,h0,'r1col',1
sxaddpar,h0,'r2col',imsize[1]
sxaddpar,h0,'r1row',1
sxaddpar,h0,'r2row',imsize[0]
sxaddpar,h0,'naxis1',imsize[0]
sxaddpar,h0,'naxis2',imsize[1]


if dateobs eq '' then sxaddpar,h0,'date-obs',anytim(systim(),/ccsds) else sxaddpar,h0,'date-obs',dateobs
;sxaddpar,h0,'date_avg',dateobs
;sxaddpar,h0,'date_cmd',dateobs
;sxaddpar,h0,'date_end',dateobs

sxaddpar,h0,'rsun',atan(1.,abs(obspos[2]))*!radeg*3600,'arcsec' ; solar radius in arcsec

if obslonlatflag eq 1 then begin
    sxaddpar,h0,'dsun_obs',obslonlat[2]*696000e3,'meter'
    sxaddpar,h0,'crln_obs',obslonlat[0]*!radeg,'deg' ; -- carrington lon
    sxaddpar,h0,'crlt_obs',obslonlat[1]*!radeg,'deg' ; -- carrington lat
    
    ; ---- CONVERT HERE FROM CARRINGTON TO STONYHURST
    if dateobs eq '' then begin
        message,'WARNING : cannot convert Carrington coordinates to other coordinate systems without a dateobs !',/info
    endif else begin

        if obslonlatheaderflag then begin
            ; -- use the position given by the fitsheader
            
            sxaddpar,h0,'hgln_obs',wcs.position.hgln_obs
            sxaddpar,h0,'hglt_obs',wcs.position.hglt_obs

            sxaddpar,h0,'HCIX_OBS',wcs.position.hci_obs[0]
            sxaddpar,h0,'HCIY_OBS',wcs.position.hci_obs[1]
            sxaddpar,h0,'HCIZ_OBS',wcs.position.hci_obs[2]
            
            sxaddpar,h0,'HAEX_OBS',wcs.position.hae_obs[0]
            sxaddpar,h0,'HAEY_OBS',wcs.position.hae_obs[1]
            sxaddpar,h0,'HAEZ_OBS',wcs.position.hae_obs[2]
            
            sxaddpar,h0,'HEEX_OBS',wcs.position.hee_obs[0]
            sxaddpar,h0,'HEEY_OBS',wcs.position.hee_obs[1]
            sxaddpar,h0,'HEEZ_OBS',wcs.position.hee_obs[2]
            
            sxaddpar,h0,'HEQX_OBS',wcs.position.heq_obs[0]
            sxaddpar,h0,'HEQY_OBS',wcs.position.heq_obs[1]
            sxaddpar,h0,'HEQZ_OBS',wcs.position.heq_obs[2]
            
            sxaddpar,h0,'dsun_obs',wcs.position.dsun_obs
            
            sxaddpar,h0,'solar_b0',wcs.position.solar_b0
            sxaddpar,h0,'CARR_EARTH',wcs.position.CARR_EARTH


        endif else begin
            ; -- use the date_obs to compute the position in the other
            ;    coordinate systems

            ;carrlong,anytim2utc(dateobs),cr,clong
       	    coord=get_stereo_lonlat(dateobs,'earth',system='car',/degrees)
            clong=coord[1]

            lonstony=obslonlat[0]*!radeg-clong
            if lonstony lt 0 then lonstony+=360.
            sxaddpar,h0,'hgln_obs',lonstony,'deg' ; -- stonyhurst lon 
            sxaddpar,h0,'hglt_obs',obslonlat[1]*!radeg,'deg' ; -- stonyhurst lat: 
                                                  ;    same as Carrington lat
            ; ---- polar to cartesian
            coord0=latlonheight2xyz([obslonlat[1],obslonlat[0],sxpar(h0,'dsun_obs')])

            ; ----- NOTE -----
            ; Using this conversion method,
            ; starting with the Stonyhurst coordinates, I still
            ; find some discrepancies between what
            ; I find in a SECCHI header and what I calculate here.
            ; Bug #14

            ; ---- CAR to HCI
            coord=coord0
            convert_stereo_coord,dateobs,coord,'CAR','HCI'
            sxaddpar,h0,'HCIX_OBS',coord[0]
            sxaddpar,h0,'HCIY_OBS',coord[1]
            sxaddpar,h0,'HCIZ_OBS',coord[2]
        
            ; ---- CAR to HAE
            coord=coord0
            convert_stereo_coord,dateobs,coord,'CAR','HAE'
            sxaddpar,h0,'HAEX_OBS',coord[0]
            sxaddpar,h0,'HAEY_OBS',coord[1]
            sxaddpar,h0,'HAEZ_OBS',coord[2]

            ; ---- CAR to HEE
            coord=coord0
            convert_stereo_coord,dateobs,coord,'CAR','HEE'
            sxaddpar,h0,'HEEX_OBS',coord[0]
            sxaddpar,h0,'HEEY_OBS',coord[1]
            sxaddpar,h0,'HEEZ_OBS',coord[2]

            ; ---- CAR to HEQ
            coord=coord0
            convert_stereo_coord,dateobs,coord,'CAR','HEQ'
            sxaddpar,h0,'HEQX_OBS',coord[0]
            sxaddpar,h0,'HEQY_OBS',coord[1]
            sxaddpar,h0,'HEQZ_OBS',coord[2]
        endelse

    endelse
endif else begin
    sxaddpar,h0,'dsun_obs',norm(obspos)*onersun()*1e3 ; - dist to sun center in m
endelse

; ---- simulation parameters
sxaddpar,h0,'OBSPOSX',obspos[0],'Rsun'
sxaddpar,h0,'OBSPOSY',obspos[1],'Rsun'
sxaddpar,h0,'OBSPOSZ',obspos[2],'Rsun'

sxaddpar,h0,'OBSLON',obslonlat[0],'rad'
sxaddpar,h0,'OBSLAT',obslonlat[1],'rad'
sxaddpar,h0,'OBSHEIGHT',obslonlat[2],'Rsun'

sxaddpar,h0,'OBSANGX',obsang[0],'rad'
sxaddpar,h0,'OBSANGY',obsang[1],'rad'
sxaddpar,h0,'OBSANGZ',obsang[2],'rad'

sxaddpar,h0,'NEPOSX',nepos[0],'Rsun'
sxaddpar,h0,'NEPOSY',nepos[1],'Rsun'
sxaddpar,h0,'NEPOSZ',nepos[2],'Rsun'

sxaddpar,h0,'NEANGX',neang[0],'rad'
sxaddpar,h0,'NEANGY',neang[1],'rad'
sxaddpar,h0,'NEANGZ',neang[2],'rad'


sxaddpar,h0,'NEROTCNTRX',nerotcntr[0],'Rsun'
sxaddpar,h0,'NEROTCNTRY',nerotcntr[1],'Rsun'
sxaddpar,h0,'NEROTCNTRZ',nerotcntr[2],'Rsun'

sxaddpar,h0,'NEROTANGX',nerotang[0],'rad'
sxaddpar,h0,'NEROTANGY',nerotang[1],'rad'
sxaddpar,h0,'NEROTANGZ',nerotang[2],'rad'

sxaddpar,h0,'NEROTAXIS1',nerotaxis[0]
sxaddpar,h0,'NEROTAXIS2',nerotaxis[1]
sxaddpar,h0,'NEROTAXIS3',nerotaxis[2]

sxaddpar,h0,'NETRANSX',netranslation[0],'rad'
sxaddpar,h0,'NETRANSY',netranslation[1],'rad'
sxaddpar,h0,'NETRANSZ',netranslation[2],'rad'

sxaddpar,h0,'LOSNBP',losnbp,'points'

sxaddpar,h0,'LOSRNG1',losrange[0],'Rsun'
sxaddpar,h0,'LOSRNG2',losrange[1],'Rsun'

sxaddpar,h0,'MODELID',modelid

if n_elements(file) ne 0 then $
  sxaddpar,h0,'DENSFILE',file else $
  sxaddpar,h0,'DENSFILE',''

sxaddpar,h0,'HLONLAT0',hlonlat[0],'rad'
sxaddpar,h0,'HLONLAT1',hlonlat[1],'rad'
sxaddpar,h0,'HLONLAT2',hlonlat[2],'rad'

if cor1 or cor2 or hi1 or hi2 then sxaddpar,h0,'SECCHIAB',secchiab,'Spacecraft A or B'
sxaddpar,h0,'PROJTYPE',projtype


hbtot=h0
sxaddpar,hbtot,'BUNIT','Bsun'
sxaddpar,hbtot,'COMMENT','Total brightness'

hbpol=h0
sxaddpar,hbpol,'BUNIT','Bsun'
sxaddpar,hbpol,'COMMENT','Polarized brightness'

hnetot=h0
sxaddpar,hnetot,'BUNIT','cm^(-3)'
sxaddpar,hnetot,'COMMENT','Integrated electron density'


revision=getcvsrevision('$Revision: 1.18 $')

sbtot={imsize:imsize,fovpix:fovpix,obspos:obspos,obsang:obsang,nepos:nepos,neang:neang,losnbp:losnbp,losrange:losrange,modelid:modelid,crpixprg:crpix,im:btot,fitshead:hbtot,mmlon:mmlon,mmlat:mmlat,neonly:neonly,roi:roi,pofinteg:pofinteg,poiang:poiang,hlonlat:hlonlat,secchiab:secchiab,occrad:occrad,revision:revision,adapthres:adapthres,maxsubdiv:maxsubdiv,rtdetec:rtdetec,densfile:(n_elements(file) ne 0 ? file : ''),limbdark:limbdark,rotmat:rotmat,obslonlat:obslonlat,projtype:projtype,dateobs:dateobs,frontinteg:frontinteg,nerotcntr:nerotcntr,nerotang:nerotang,netranslation:netranslation,nerotaxis:nerotaxis}

sbpol={imsize:imsize,fovpix:fovpix,obspos:obspos,obsang:obsang,nepos:nepos,neang:neang,losnbp:losnbp,losrange:losrange,modelid:modelid,crpixprg:crpix,im:bpol,fitshead:hbpol,mmlon:mmlon,mmlat:mmlat,neonly:neonly,roi:roi,pofinteg:pofinteg,poiang:poiang,hlonlat:hlonlat,secchiab:secchiab,occrad:occrad,revision:revision,adapthres:adapthres,maxsubdiv:maxsubdiv,rtdetec:rtdetec,densfile:(n_elements(file) ne 0 ? file : ''),limbdark:limbdark,rotmat:rotmat,obslonlat:obslonlat,projtype:projtype,dateobs:dateobs,frontinteg:frontinteg,nerotcntr:nerotcntr,nerotang:nerotang,netranslation:netranslation,nerotaxis:nerotaxis}

snetot={imsize:imsize,fovpix:fovpix,obspos:obspos,obsang:obsang,nepos:nepos,neang:neang,losnbp:losnbp,losrange:losrange,modelid:modelid,crpixprg:crpix,im:netot,fitshead:hnetot,mmlon:mmlon,mmlat:mmlat,neonly:neonly,roi:roi,pofinteg:pofinteg,poiang:poiang,hlonlat:hlonlat,secchiab:secchiab,occrad:occrad,revision:revision,adapthres:adapthres,maxsubdiv:maxsubdiv,rtdetec:rtdetec,densfile:(n_elements(file) ne 0 ? file : ''),limbdark:limbdark,rotmat:rotmat,obslonlat:obslonlat,projtype:projtype,dateobs:dateobs,frontinteg:frontinteg,nerotcntr:nerotcntr,nerotang:nerotang,netranslation:netranslation,nerotaxis:nerotaxis}

srhorrr={imsize:imsize,fovpix:fovpix,obspos:obspos,obsang:obsang,nepos:nepos,neang:neang,losnbp:losnbp,losrange:losrange,crpixprg:crpix,rho:rho,rrr:rrr,mmlon:mmlon,mmlat:mmlat,hlonlat:hlonlat,secchiab:secchiab,occrad:occrad,revision:revision,rtdetec:rtdetec,nerotcntr:nerotcntr,nerotang:nerotang,netranslation:netranslation,nerotaxis:nerotaxis}


if n_elements(save) ne 0 then begin
    if (save ne '0') then begin
    
        print,'Saving the results'
        
        if xdr then begin
            savestruct,save+'btot.xdr',sbtot
            savestruct,save+'bpol.xdr',sbpol
            savestruct,save+'netot.xdr',snetot
            savestruct,save+'rhorrr.xdr',srhorrr
        endif else begin
            mwrfits,sbtot,save+'btot.fits',/create
            mwrfits,sbpol,save+'bpol.fits',/create
            mwrfits,snetot,save+'netot.fits',/create
            mwrfits,srhorrr,save+'rhorrr.fits',/create
        endelse
    endif
endif

return
end
;
; CVSLOG:
;  $Log: raytracewl.pro,v $
;  Revision 1.18  2010/09/08 15:58:04  thernis
;  - use disttofracmax
;  - implement physics
;  - use crval
;
;  Revision 1.17  2009-07-09 19:11:36  thernis
;  Cleanup unuseful code
;
;  Revision 1.16  2009-04-15 20:48:39  thernis
;  Add time keyword for benchmarking
;
;  Revision 1.15  2009/04/13 21:18:41  thernis
;  Implement extra positioning parameters for the models.
;
;  Revision 1.14  2009/04/07 18:59:32  thernis
;  Change a print to a comment if rtthread
;
;  Revision 1.13  2009/03/06 22:14:41  thernis
;  Implement neshift
;
;  Revision 1.12  2009/02/10 16:45:37  thernis
;  - Implement uv integration (experimental for now)
;  - Implement multi threading raytracing
;
;  Revision 1.11  2008/09/23 14:13:36  thernis
;  implement frontinteg kw
;
;  Revision 1.10  2008/04/09 15:41:13  thernis
;  Fix problem with pv2_1 user value
;
;  Revision 1.9  2007/05/11 20:49:24  thernis
;  Update WCS compliant positioning and header info
;
;  Revision 1.8  2007/03/15 20:22:25  thernis
;  Implement AZP projection
;
;  Revision 1.7  2006/11/29 22:10:09  thernis
;  Implement scchead keyword
;
;  Revision 1.6  2006/11/27 22:14:03  thernis
;  Change date_obs into date-obs
;
;  Revision 1.5  2006/11/22 16:16:55  thernis
;  - Put proper units in the WCS header
;  - Add position in the different coordinate systems present in standard SECCHI headers
;
;  Revision 1.4  2006/11/14 21:55:35  thernis
;  - Implement WCS projection type
;  - Change fitsheader to comply with WCS standard
;  - Different small modifs: see with diff
;
;  Revision 1.3  2006/11/09 19:43:30  thernis
;  Implement obslonlat keyword to allow positioning the observer using Carrington spherical coordinates.
;
;  Revision 1.2  2006/10/30 21:52:40  thernis
;  Change the environment variable that points to the shared object library.
;
;
;
