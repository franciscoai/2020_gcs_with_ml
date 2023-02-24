pro rtgetcarposondisk,pixpos,carlonlat,flagintersection,imsize=imsize,$
  fovpix=fovpix,$
  obspos=obspos,obsang=obsang,$
  c2image=c2image,c3image=c3image,$
  quiet=quiet,$
  cor1=cor1,cor2=cor2,hi1=hi1,hi2=hi2,c1fov=c1fov,$
  hlonlat=hlonlat,secchiab=secchiab,$
  instr=instr,$
  obslonlat=obslonlat,$
  projtype=projtype,rollang=rollang,$
  scchead=scchead,pv2_1=pv2_1,pcin=pcin

;+
;  $Id: rtgetcarposondisk.pro,v 1.1 2007/07/19 19:49:33 thernis Exp $
; 
; PURPOSE: 
;  Gives the Carrington lon lat on the limb from a pix position
;
; CATEGORY:
;  raytracing, simulation, 3d
;
; INPUTS:
; ! all the distances must be given in Rsun
; ! and the angles in rad
; plist : list of points : [3,N]
; imsize : [xs,ys] size of the output image
; fovpix : fov angle of one pixel in rad 
; -- observer position and attitude
; obspos : [x,y,z] position of the observer in the Sun basis
; obslonlat : [lon,lat,height] position of the observer in Carrington
;             coordinate. If set, then obspos is ignored. The optical
;             axis always points toward the Sun center. Use obsang to
;             change telescope orientation.
; obsang : [ax,ay,az] orientation of the observer, 
;          z is the optical axis 
; rollang : allow to set the roll angle of the virtual instrument. 
;           Works only if a preset instrument is requested.
; -- Ne position and attitude
; quiet : disable display of raytracing parameters
; hlonlat : [Hlon,Hlat,Hrot] heliographic lon and lat of the center of
;           the disk, rotation angle corresponding to the projection of the
;           north pole, counterclockwise
; secchiab : 'A' or 'B', to select Ahead or Behind spacecraft, for
;            secchi only
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
; pcin : force the fits PCi_j matrix. Must be a 4 elements array
;
; instr : txt instrument preset, to select from the list above:
; scchead : secchi structure header: raytrace will use the
;           positionning info of the header to generate the view
;
; -- Instrument FOV preset
; c1, c2, c3 : lasco C1, C2, C3
; cor1, cor2 : Secchi Cor1, Cor2
; hi1, hi2 : Secchi Hi1, Hi2
;
; OUTPUTS:
;  carlonlat : carrington lon and lat in rad
;  flagintersection : 0 if no intersection with solar disk
;
; CALL EXAMPLE:
;
;-


; ---- define keyword default values
;      and cast every inputs to avoid C crash with bad type entry
; -- use header information if passed by user
;obslonlatheaderflag=0B
obslonlatflag=0B
pv2_1headerflag=0B
rollangheaderflag=0B
if n_elements(scchead) ne 0 then begin
    if scchead.instrume ne 'SECCHI' then message,'The header passed in scchead should be a SECCHI header.'

    wcs=fitshead2wcs(scchead)

    if n_elements(instr) eq 0 then instr=scchead.detector
    if n_elements(secchiab) eq 0 then secchiab=strmid(scchead.obsrvtry,0,1,/reverse_offset)

    if n_elements(obslonlat) eq 0 then begin
        ; ---- position of the virtual spacecraft
        obslonlat=float([wcs.position.crln_obs*!dtor,wcs.position.crlt_obs*!dtor,scchead.dsun_obs/(onersun()*1e3)])
        obslonlatflag=1B
    endif
    if n_elements(pv2_1) ne 0 then begin
        pv2_1=scchead.pv2_1
        pv2_1headerflag=1b
    endif

    if n_elements(rollang) eq 0 then begin
        rollang=0.
        rollangheaderflag=1B
    endif

endif

if n_elements(plist) eq 0 then plist=[0.,0.,0] else plist=float(plist)
if n_elements(imsize) eq 0 then imsize=[64L,64] else imsize=long(imsize)
if n_elements(fovpix) eq 0 then begin
    fovpix=2./(64.)*!dtor 
    flagfovpix=0B
endif else begin
    fovpix=float(fovpix)
    flagfovpix=1B
endelse
if n_elements(obspos) eq 0 then begin
    obspos=[0.,0,-214] 
    obsposflag=0b
endif else begin
    obspos=float(obspos)
    obsposflag=1b
endelse
if n_elements(obsang) eq 0 then begin
    obsang=[0.,0,0] 
    obsangflag=0b
endif else begin
    obsang=float(obsang)
    obsangflag=1b
endelse
if n_elements(quiet) eq 0 then quiet=0L else quiet=1L
if n_elements(secchiab) eq 0 then secchiab='A' else begin
    secchiab=strupcase(secchiab)
    if secchiab ne 'A' and secchiab ne 'B' then message,'secchiab keyword must be either ''A'' or ''B'''
endelse
if n_elements(obslonlat) ne 0 and not obslonlatflag then begin 
    obslonlat=float(obslonlat)
    obspos=obslonlat[2]*[sin(obslonlat[1]),$
                         sin(obslonlat[0])*cos(obslonlat[1]),$
                         -cos(obslonlat[0])*cos(obslonlat[1])]
    obslonlatflag=1L
endif 
if n_elements(obslonlat) eq 0 then begin
    obslonlat=[-atan(obspos[1],obspos[2]),$
               asin(obspos[1]/norm(obspos)),$
               norm(obspos)]
    obslonlatflag=0L
endif
if n_elements(rollang) eq 0 then rollang=0.

if n_elements(carlonlat) ne 2 then carlonlat=[0.,0.] else carlonlat=float(carlonlat)
if n_elements(flagintersection) ne 1 then flagintersection=0L else flagintersection=long(flagintersection)


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
rtgetinstrwcsparam,instr,imsize,scchead,fovpix,crpix,obsangpreset,pc,projtypepreset=projtypepreset,pv2_1=pv2_1,rollang=rollang,crval=crval,pcin=pcin,flagfovpix=flagfovpix

; ---- compute incerse of pc matrix
pc=float(pc)
pcinv=float(invert(pc))

; ---- size of the list of points
if size(plist,/n_dim) eq 1 then nbp=1L else nbp=long((size(plist,/dim))[1])

; -- obsang and rollang is forced if passed by user
if not obsangflag and n_elements(instr) ne 0 then obsang=obsangpreset
if n_elements(pv2_1) eq 0 then pv2_1=0.

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
rotmat=fltarr(3,3)

rtinitenv
starttime=systime(1)
s=call_external(getenv('RT_LIBFILE'),$
                'rtGetCarPosOnDisk',$
                imsize[0],imsize[1],$
                fovpix,$
                obspos,obsang,$
                crpix,quiet,$
                obslonlat,obslonlatflag,projtypecode,pv2_1,pc,$
                pixpos,pcinv,carlonlat,flagintersection,/unload)


if quiet eq 0 then begin
    print,'Seconds ellapsed :'
    print,systime(1)-starttime
endif

revision=getcvsrevision('$Revision: 1.1 $')


return
end
;
; CVSLOG:
;  $Log: rtgetcarposondisk.pro,v $
;  Revision 1.1  2007/07/19 19:49:33  thernis
;  First commit
;
;  Revision 1.1  2007/07/10 21:12:19  thernis
;  First commit.
;
;
