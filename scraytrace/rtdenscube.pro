
; ---- wrapper that calls the C raytracing program optimized for a
;      density cube

pro rtdenscube,sbtot,sbpol,snetot,imsize=imsize,$
               fovpix=fovpix,$
               obspos=obspos,obsang=obsang,$
               nepos=nepos,neang=neang,$
               denscube=denscube,$
               voxsize=voxsize,dccenter=dccenter,$
               save=save,file=file,fakelasco=fakelasco,$
               c2image=c2image,c3image=c3image,$
               rho=rho,mmlon=mmlon,mmlat=mmlat,rrr=rrr,$
               pofinteg=pofinteg,quiet=quiet,$
               cor1=cor1,cor2=cor2,hi1=hi1,hi2=hi2,c1fov=c1fov,$
               neonly=neonly,roi=roi,poiang=poiang,$
               hlonlat=hlonlat,secchiab=secchiab,occrad=occrad,$
               xdr=xdr,$
               instr=instr,limbdark=limbdark,denslistfn=denslistfn,$
               obslonlat=obslonlat,$
               projtype=projtype,dateobs=dateobs,rollang=rollang,$
               scchead=scchead

;+
;  $Id: rtdenscube.pro,v 1.4 2007/05/11 20:02:06 thernis Exp $
;
; PURPOSE
;  Density cube main raytracing program: splatting method
; 
; CATEGORY:
;  raytracing, 3d
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
;             change telescope orientation.
; obsang : [ax,ay,az] orientation of the observer, 
;          z is the optical axis 
; rollang : allow to set the roll angle of the virtual instrument. 
;           Works only if a preset instrument is requested.
; -- Ne position and attitude
; nepos : [x,y,z] position of the Ne reference in the Sun basis
; neang : [ax,ay,az] orientation of the Ne
; -- POI (central Plane of Integration) orientation
; poiang : [ax,ay,az] orientation of the POI z axis: note that az
;          rotation has no effect.
; -- LOS params
; denscube : density cube
; voxsize : size of the edge of a voxel, in Rsun. 
;               Voxels are assumed to be cubic
; dccenter : [xc,yc,zc] center of the density cube in pix. Relative 
;            to the origin that is at the center of the voxel [0,0,0]:
;            the first vertice of the density cube is at -[0.5,0.5,0.5]
; save : put path and filename in that variable (without extention) 
;        if you want to save the results in a .fits binary table.
; file : density cube filename for density 4 and 5
; fakelasco : put fake lasco header information in the fits header
; pofinteg : the raytracing LOS center is taken in the plane of the sky
;          containing the Sun center instead of the Sun center 
;          projection on the LOS (impact distance projection)
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
; xdr : save into xdr format instead of fits table. 'save' keyword
;       must be set for xdr to take effect.
; projtype : projection type: (see Calabretta and Greisen,
;            Representations of celestial coordinates in FITS, A&A
;            395, 1077-1122(2002))
;             ARC : Zenithal equidistant (default)
;             TAN : Gnomonic
;             SIN : Slant orthographic
;            If an instrument preset is requested then this keyword
;            will overwrite the projection type of the selected
;            instrument.
; dateobs : observation date that will be copied in the image header
;           and used to compute the observer position in the different
;           coordinate systems.
; instr : txt instrument preset, to select from the list above:
; scchead : secchi structure header: raytrace will use the
;           positionning info of the header to generate the view
;
; -- Instrument FOV preset
; c1, c2, c3 : lasco C1, C2, C3
; cor1, cor2 : Secchi Cor1, Cor2
; hi1, hi2 : Secchi Hi1, Hi2
; limbdark : limb darkening coeff: default 0.58
; denslistfn : density list filename. A density list can be generated
;              by buildcloud using the option outputtype=3
;
; OUTPUTS:
;  sbtot : structure with image of the total brightness
;  sbpol : polarized brightness
;  snetot : integrated electron density along the LOSes
;  rho : impact parameter for each LOS
;  mmlon : min and max longitude
;  mmlat : min and max latitude
;  rrr : dist instersection LOS - plane of the sky containing the Sun cntr
;
;
;
;-


; ---- define keyword default values
;      and cast every inputs to avoid C crash with bad type entry

; -- use header information if passed by user
if n_elements(scchead) ne 0 then begin
    if scchead.instrume ne 'SECCHI' then message,'The header passed in scchead should be a SECCHI header.'
    if n_elements(dateobs) eq 0 then dateobs=scchead.date_obs
    if n_elements(instr) eq 0 then instr=scchead.detector
    if n_elements(secchiab) eq 0 then secchiab=strmid(scchead.obsrvtry,0,1,/reverse_offset)

    if n_elements(obslonlat) eq 0 then begin
        ; -- transform from stonyhurst to carrington
        carrlong,anytim2utc(dateobs),cr,clong
    
        loncarr=scchead.hgln_obs + clong
        if loncarr gt 360 then loncarr-=360.
        
        obslonlat=[loncarr*!dtor,scchead.hglt_obs*!dtor,scchead.dsun_obs/696000e3]
    endif
endif


if n_elements(imsize) eq 0 then imsize=[64L,64] else imsize=long(imsize)
if n_elements(fovpix) eq 0 then fovpix=2./(64.)*!dtor else fovpix=float(fovpix)
if n_elements(obspos) eq 0 then obspos=[0.,0,-214] else obspos=float(obspos)
if n_elements(obsang) eq 0 then obsang=[0.,0,0] else obsang=float(obsang)
if n_elements(nepos) eq 0 then nepos=[0.,0,0] else nepos=float(nepos)
if n_elements(neang) eq 0 then neang=[0.,0,0] else neang=float(neang)
if n_elements(denscube) eq 0 then denscube=1. else denscube=float(denscube)


if size(denscube,/type) eq 7 then begin
                                ; ---- denscube is a string: it should
;                                      contain the filename of a
;                                      density cube: well, not taken
;                                      into account yet...

    flagdenscubetype=2L      
    

endif else begin
    if size(denscube,/n_dimension) eq 0 then cubesize=[1L,1,1] else cubesize=long((size(denscube,/dim)))
    if n_elements(voxsize) eq 0 then voxsize=1. else voxsize=float(voxsize)
    if n_elements(dccenter) eq 0 then dccenter=[0.,0,0] else dccenter=float(dccenter)
    flagdenscubetype=0L         ; -- the different dim of the cubes are in the arguments: default
    if size(denscube,/n_dimension) eq 1 then begin
        flagdenscubetype=1L     ; -- the cube is formated as defined in loaddenscube.pro
        cubesize=long(denscube[0:2])
        dccenter=float(denscube[3:5])
        voxsize=float(denscube[6])

    endif
endelse
if n_elements(denslistfn) eq 0 then bytedenslistfn=byte('') else begin
    bytedenslistfn=[byte(denslistfn),0]
    flagdenscubetype=3L 
endelse



if n_elements(pofinteg) eq 0 then pofinteg=0L else pofinteg=long(pofinteg)
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
if n_elements(limbdark) eq 0 then limbdark=0.58 else limbdark=float(limbdark)
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
rtinstrpresets,instr,imsize,fovpix,obsang,secchiab=secchiab;,projtypepreset=projtypepreset

; ---- set projection type
;if n_elements(projtype) eq 0 then begin
;    if n_elements(projtypepreset) eq 0 then projtype='ARC' else projtype=projtypepreset
;endif

;projtype=strupcase(projtype)
;case projtype of
;    'ARC' : projtypecode=1L
;    'TAN' : projtypecode=2L
;    'SIN' : projtypecode=3L
;    else : message,'Bad projtype keyword !'
;endcase

; ---- init the outputs
btot=fltarr(imsize[0],imsize[1])
bpol=fltarr(imsize[0],imsize[1])
netot=fltarr(imsize[0],imsize[1])
rho=fltarr(imsize[0],imsize[1])
mmlon=fltarr(2)
mmlat=fltarr(2)
rrr=fltarr(imsize[0],imsize[1])
crpix=fltarr(2)


rtinitenv
starttime=systime(1)
s=call_external(getenv('RT_LIBFILE'),$
                'rtdenscube',$
                imsize[0],imsize[1],$
                fovpix,$
                obspos,obsang,$
                nepos,neang,$
                btot,bpol,netot,denscube,$
                crpix,rho,mmlon,mmlat,rrr,pofinteg,quiet,neonly,$
                roi,poiang,hlonlat,occrad,limbdark,$
                cubesize,voxsize,dccenter,flagdenscubetype,$
                bytedenslistfn,obslonlat,obslonlatflag,/unload)

if quiet eq 0 then begin
    print,'Seconds ellapsed :'
    print,systime(1)-starttime
endif

; -- sun radius in pix 
solar_r=abs(atan(1./obspos[2]))/fovpix

; ---- fill in the fits header
mkhdr,h0,btot
sxaddpar,h0,'COMMENT','scraytrace'
sxaddpar,h0,'AUTHOR','scraytrace'
sxaddpar,h0,'PROGRAM','scraytrace'
sxaddpar,h0,'DATE',anytim(systim(),/ccsds)


sxaddpar,h0,'SOLAR_R',solar_r,'pix'
sxaddpar,h0,'SOLAR_B0',0.
sxaddpar,h0,'FOVPIX',fovpix,'rad'
case 1 of
    c2image : rtdetec='c2image'
    c3image : rtdetec='c3image'
    c1fov : rtdetec='c1fov'
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

sxaddpar,h0,'CRPIX1',crpix[0]+1,'X Sun center (pix) (FORTRAN convention)'
sxaddpar,h0,'CRPIX2',crpix[1]+1,'Y Sun center (pix)'

sxaddpar,h0,'CTYPE1','HPLN-'+projtype
sxaddpar,h0,'CTYPE2','HPLT-'+projtype

sxaddpar,h0,'CUNIT1','deg'
sxaddpar,h0,'CUNIT2','deg'

sxaddpar,h0,'CDELT1',-fovpix*!radeg,'Long. -> E (rad)'
sxaddpar,h0,'CDELT2', fovpix*!radeg,'Lat. -> N (rad)'

sxaddpar,h0,'crval1',0.
sxaddpar,h0,'crval2',-90.

; -- WARNING: pci_j implementation not tested
sxaddpar,h0,'pc1_1',cos(obsang[2])
sxaddpar,h0,'pc2_1',-sin(obsang[2])
sxaddpar,h0,'pc1_2',sin(obsang[2])
sxaddpar,h0,'pc2_2',cos(obsang[2])

sxaddpar,h0,'SOLAR_R',solar_r,'pix'
sxaddpar,h0,'FOVPIX',fovpix,'rad'

sxaddpar,h0,'history','$Id: rtdenscube.pro,v 1.4 2007/05/11 20:02:06 thernis Exp $'

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


sxaddpar,h0,'rsun',atan(1./abs(obspos[2])) ; solar radius in rad

if obslonlatflag eq 1 then begin
    sxaddpar,h0,'dsun_obs',obslonlat[2]*696000e3,'meter'
    sxaddpar,h0,'crln_obs',obslonlat[0]*!radeg,'deg' ; -- carrington lon
    sxaddpar,h0,'crlt_obs',obslonlat[1]*!radeg,'deg' ; -- carrington lat
    
    ; ---- CONVERT HERE FROM CARRINGTON TO STONYHURST
    if dateobs eq '' then begin
        message,'WARNING : cannot convert Carrington coordinates to other coordinate systems without a dateobs !',/info
    endif else begin

        carrlong,anytim2utc(dateobs),cr,clong
        
        lonstony=obslonlat[0]*!radeg-clong
        if lonstony lt 0 then lonstony+=360.
        sxaddpar,h0,'hgln_obs',lonstony,'deg'     ; -- stonyhurst lon 
        sxaddpar,h0,'hglt_obs',obslonlat[1]*!radeg,'deg' ; -- stonyhurst lat: 
                                                  ;    same as Carrington lat
        ; ---- polar to cartesian
        coord0=latlonheight2xyz([obslonlat[1],obslonlat[0],sxpar(h0,'dsun_obs')])

        ; ----- NOTE -----
        ; Using this conversion method,
        ; starting with the Stonyhurst coordinates, I still
        ; find some discrepancies between what
        ; I find in a SECCHI header and what I calculate here.

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
endif else begin
    sxaddpar,h0,'dsun_obs',norm(obspos)*696000e3 ; - dist to sun center in m
endelse





sxaddpar,h0,'OBSPOSX',obspos[0],'Rsun'
sxaddpar,h0,'OBSPOSY',obspos[1],'Rsun'
sxaddpar,h0,'OBSPOSZ',obspos[2],'Rsun'
sxaddpar,h0,'OBSANGX',obsang[0],'rad'
sxaddpar,h0,'OBSANGY',obsang[1],'rad'
sxaddpar,h0,'OBSANGZ',obsang[2],'rad'
sxaddpar,h0,'NEPOSX',nepos[0],'Rsun'
sxaddpar,h0,'NEPOSY',nepos[1],'Rsun'
sxaddpar,h0,'NEPOSZ',nepos[2],'Rsun'
sxaddpar,h0,'NEANGX',neang[0],'rad'
sxaddpar,h0,'NEANGY',neang[1],'rad'
sxaddpar,h0,'NEANGZ',neang[2],'rad'
if n_elements(file) ne 0 then $
  sxaddpar,h0,'DENSFILE',file else $
  sxaddpar,h0,'DENSFILE',''
sxaddpar,h0,'HLONLAT0',hlonlat[0],'rad'
sxaddpar,h0,'HLONLAT1',hlonlat[1],'rad'
sxaddpar,h0,'HLONLAT2',hlonlat[2],'rad'
if cor1 or cor2 or hi1 or hi2 then sxaddpar,h0,'SECCHIAB',secchiab,'Spacecraft A or B'



hbtot=h0
sxaddpar,hbtot,'UNITS','Bsun'
sxaddpar,hbtot,'COMMENT','Total brightness'

hbpol=h0
sxaddpar,hbpol,'UNITS','Bsun'
sxaddpar,hbpol,'COMMENT','Polarized brightness'

hnetot=h0
sxaddpar,hnetot,'UNITS','e- . cm-3'
sxaddpar,hnetot,'COMMENT','Integrated electron density'


revision=getcvsrevision('$Revision: 1.4 $')

sbtot={imsize:imsize,fovpix:fovpix,obspos:obspos,obsang:obsang,nepos:nepos,neang:neang,crpixprg:crpix,im:btot,fitshead:hbtot,mmlon:mmlon,mmlat:mmlat,neonly:neonly,roi:roi,pofinteg:pofinteg,poiang:poiang,hlonlat:hlonlat,secchiab:secchiab,occrad:occrad,revision:revision,rtdetec:rtdetec,densfile:(n_elements(file) ne 0 ? file : ''),limbdark:limbdark,rotmat:rotmat,obslonlat:obslonlat,projtype:projtype,dateobs:dateobs}

sbpol={imsize:imsize,fovpix:fovpix,obspos:obspos,obsang:obsang,nepos:nepos,neang:neang,crpixprg:crpix,im:bpol,fitshead:hbpol,mmlon:mmlon,mmlat:mmlat,neonly:neonly,roi:roi,pofinteg:pofinteg,poiang:poiang,hlonlat:hlonlat,secchiab:secchiab,occrad:occrad,revision:revision,rtdetec:rtdetec,densfile:(n_elements(file) ne 0 ? file : ''),limbdark:limbdark,rotmat:rotmat,obslonlat:obslonlat,projtype:projtype,dateobs:dateobs}

snetot={imsize:imsize,fovpix:fovpix,obspos:obspos,obsang:obsang,nepos:nepos,neang:neang,crpixprg:crpix,im:netot,fitshead:hnetot,mmlon:mmlon,mmlat:mmlat,neonly:neonly,roi:roi,pofinteg:pofinteg,poiang:poiang,hlonlat:hlonlat,secchiab:secchiab,occrad:occrad,revision:revision,rtdetec:rtdetec,densfile:(n_elements(file) ne 0 ? file : ''),limbdark:limbdark,rotmat:rotmat,obslonlat:obslonlat,projtype:projtype,dateobs:dateobs}

srhorrr={imsize:imsize,fovpix:fovpix,obspos:obspos,obsang:obsang,nepos:nepos,neang:neang,crpixprg:crpix,rho:rho,rrr:rrr,mmlon:mmlon,mmlat:mmlat,hlonlat:hlonlat,secchiab:secchiab,occrad:occrad,revision:revision,rtdetec:rtdetec}


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
;  $Log: rtdenscube.pro,v $
;  Revision 1.4  2007/05/11 20:02:06  thernis
;  Implement wcs header. NOT TESTED !
;
;  Revision 1.3  2006/11/09 20:27:18  thernis
;  Implement positioning of the observer with Carrington spherical coordinates
;
;  Revision 1.2  2006/10/30 21:52:40  thernis
;  Change the environment variable that points to the shared object library.
;
;
