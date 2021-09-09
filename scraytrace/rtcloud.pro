pro rtcloud,plist,sbtot,imsize=imsize,$
            fovpix=fovpix,$
            obspos=obspos,obsang=obsang,$
            nepos=nepos,neang=neang,$
            c2image=c2image,c3image=c3image,$
            quiet=quiet,$
            cor1=cor1,cor2=cor2,hi1=hi1,hi2=hi2,c1fov=c1fov,$
            hlonlat=hlonlat,$
            instr=instr,rotmat=rotmat,$
            obslonlat=obslonlat,$
            projtype=projtype,rollang=rollang,$
            scchead=scchead,pv2_1=pv2_1,pcin=pcin,flistout=flistout,$
            fclip=fclip,listout=listout,listbehind=listbehind,$
            nerotcntr=nerotcntr,nerotang=nerotang,nerotaxis=nerotaxis,$
            netranslation=netranslation,unload=unload

;+
;  $Id: rtcloud.pro,v 1.11 2013/01/29 17:50:58 thernis Exp $
; 
; PURPOSE: 
;  Generate a view using a cloud of points
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
; nepos : [x,y,z] position of the Ne reference in the Sun basis
; neang : [ax,ay,az] orientation of the Ne
; nerotcntr : [x,y,z] center of rotation of the Ne model, in the Ne basis
; nerotang : [ax,ay,az] rotation of the Ne model around the nerotcntr, in the Ne basis
; nerotaxis : [axid1,axid2,axid3] axis id corresponding to the nerotang rotation angles. 1: X, 2: Y, 3: Z. Default is [3,2,1].
; netranslation : [tx,ty,tz] translation vector of the Ne model, in the Ne basis
; quiet : disable display of raytracing parameters
; hlonlat : [Hlon,Hlat,Hrot] heliographic lon and lat of the center of
;           the disk, rotation angle corresponding to the projection of the
;           north pole, counterclockwise
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
; flistout : set to 1 if you want a list of points instead of an image
; fclip : set to 1 if you want not to compute points masked by the sun
; 
; -- Instrument FOV preset
; c1, c2, c3 : lasco C1, C2, C3
; cor1, cor2 : Secchi Cor1, Cor2
; hi1, hi2 : Secchi Hi1, Hi2
;
; OUTPUTS:
;  sbtot : structure with image of the total brightness
;  rotmat : final rotation matrix of the density cube
;  listout : will contain the list of pixels if requested with /flistout
;  listbehind : 1 if point masked by disk, else 0
;
;-



; -- use header information if passed by user
obslonlatflag=0L
if n_elements(scchead) ne 0 then begin
	if scchead.instrume ne 'SECCHI' then flagsoho=scchead.telescop eq 'SOHO' else flagsoho=0b

    wcs=fitshead2wcs(scchead,KEYWORD_NULL_VALUE=0)

    if n_elements(instr) eq 0 then instr=scchead.detector
;     if ~flagsoho && n_elements(secchiab) eq 0 then secchiab=strmid(scchead.obsrvtry,0,1,/reverse_offset)

    if n_elements(obslonlat) eq 0 then begin
        ; ---- position of the virtual spacecraft
;         crln_obs_rad = wcschangeunits(wcs.cunit[0],'rad',wcs.position.crln_obs)
;         crlt_obs_rad = wcschangeunits(wcs.cunit[1],'rad',wcs.position.crlt_obs)

;         obslonlat = float([crln_obs_rad, crlt_obs_rad, wcs.position.dsun_obs/(onersun()*1e3)])
        obslonlat = float([(wcs.position.carr_earth + wcs.position.hgln_obs) * !dtor, wcs.position.crlt_obs * !dtor, wcs.position.dsun_obs / (onersun() * 1e3)])
;         obslonlat = float([wcs.position.crln_obs*!dtor,wcs.position.crlt_obs*!dtor,wcs.position.dsun_obs/(onersun()*1e3)])
        obslonlatflag=1L
    endif
    if n_elements(pv2_1) ne 0 then begin
        pv2_1=float(scchead.pv2_1)
    endif

    if n_elements(rollang) eq 0 then begin
        rollang=0.
    endif

endif

; ---- define keyword default values
;      and cast every inputs to avoid C crash with bad type entry
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
if n_elements(nepos) eq 0 then nepos=[0.,0,0] else nepos=float(nepos)
if n_elements(neang) eq 0 then neang=[0.,0,0] else neang=float(neang)
if n_elements(nerotcntr) eq 0 then nerotcntr=[0.,0,0] else nerotcntr=float(nerotcntr)
if n_elements(nerotang) eq 0 then nerotang=[0.,0,0] else nerotang=float(nerotang)
if n_elements(nerotaxis) eq 0 then nerotaxis=long([3,2,1]) else nerotaxis=long(nerotaxis)
if n_elements(netranslation) eq 0 then netranslation=[0.,0,0] else netranslation=float(netranslation)
if n_elements(quiet) eq 0 then quiet=0L else quiet=1L
if n_elements(hlonlat) eq 0 then hlonlat=[0.,0,0] else hlonlat=float(hlonlat)
; if n_elements(secchiab) eq 0 then secchiab='A' else begin
;     secchiab=strupcase(secchiab)
;     if secchiab ne 'A' and secchiab ne 'B' then message,'secchiab keyword must be either ''A'' or ''B''',/info
; endelse
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
if n_elements(rollang) eq 0 then rollang=0. else rollang=float(rollang)
if n_elements(flistout) eq 0 then flistout=0L else flistout=1L
if n_elements(fclip) eq 0 then fclip=0L else fclip=1L


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
listout=fltarr(2,nbp)
listbehind=lonarr(nbp)

; -- obsang and rollang is forced if passed by user
if not obsangflag and n_elements(instr) ne 0 then obsang=obsangpreset
if n_elements(pv2_1) eq 0 then pv2_1=0. else pv2_1=float(pv2_1)

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
                'rtcloud',$
                imsize[0],imsize[1],$
                fovpix,$
                obspos,obsang,$
                nepos,neang,$
                btot,$
                crpix,quiet,$
                hlonlat,$
                rotmat,obslonlat,obslonlatflag,projtypecode,pv2_1,pc,$
                plist,nbp,pcinv,flistout,listout,fclip,listbehind,nerotcntr,$
                nerotang,netranslation,nerotaxis,unload=unload)

if quiet eq 0 then begin
    print,'Seconds ellapsed :'
    print,systime(1)-starttime
endif

revision=getcvsrevision('$Revision: 1.11 $')
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
sbtot={imsize:imsize,fovpix:fovpix,obspos:obspos,obsang:obsang,nepos:nepos,neang:neang,im:btot,hlonlat:hlonlat,revision:revision,rtdetec:rtdetec,rotmat:rotmat,obslonlat:obslonlat,projtype:projtype,pc:pc,pv2_1:pv2_1,wcs:wcs,nerotcntr:nerotcntr,nerotang:nerotang,netranslation:netranslation,nerotaxis:nerotaxis}



return
end
;
; CVSLOG:
;  $Log: rtcloud.pro,v $
;  Revision 1.11  2013/01/29 17:50:58  thernis
;  Change computation of obslonlat variable.
;
;  Revision 1.10  2011-08-09 21:49:30  thernis
;  Remove unuseful keyword secchiab
;
;  Revision 1.9  2011-08-03 14:40:49  thernis
;  - cast pv2_1 to float to avoid problem with HI (AZP projection)
;  - cast obslonlatflag to long
;  - clean-up unneeded variables
;
;  Revision 1.8  2010-08-26 13:23:59  mcnutt
;  added KEYWORD_NULL_VALUE to call to fitshead2wcs
;
;  Revision 1.7  2009/04/13 21:18:41  thernis
;  Implement extra positioning parameters for the models.
;
;  Revision 1.6  2009/03/06 22:14:42  thernis
;  Implement neshift
;
;  Revision 1.5  2008/09/23 14:16:20  thernis
;  add unload kw
;
;  Revision 1.4  2008/06/06 17:38:25  thernis
;  Modif to make it work with SOHO data.
;
;  Revision 1.3  2007/07/19 22:24:26  thernis
;  Implement listbehind keyword
;
;  Revision 1.2  2007/07/19 19:20:18  thernis
;  Fix bug with obslonlat when it was forced by the user
;
;  Revision 1.1  2007/07/10 21:12:19  thernis
;  First commit.
;
;
