; ---- wrapper that calls the C raytracing program

pro rtwlcirc,sbtot,sbpol,snetot,radius=radius,cntr=cntr,imsize=imsize,$
             fovpix=fovpix,nbang=nbang,$
             obspos=obspos,obsang=obsang,$
             nepos=nepos,neang=neang,$
             losnbp=losnbp,losrange=losrange,$
             modelid=modelid,modparam=modparam,$
             save=save,file=file,fakelasco=fakelasco,$
             c2image=c2image,c3image=c3image,quiet=quiet,$
             cor1=cor1,cor2=cor2,hi1=hi1,hi2=hi2,c1fov=c1fov,$
             unload=unload,hlonlat=hlonlat,xdr=xdr,secchiab=secchiab,$
             instr=instr,limbdark=limbdark,obslonlat=obslonlat

;+
;  $Id: rtwlcirc.pro,v 1.3 2006/11/09 19:43:30 thernis Exp $
; 
; PURPOSE: 
;  Raytracing program to generate a circular profile
;
; CATEGORY:
;  raytracing, simulation, 3d
;
; INPUTS:
; ! all the distances must be given in Rsun
; ! and the angles in rad
; radius : radius of the circular profile in pix
; cntr : [x0,y0] center of the circle in pix, default is sun center
; nbang : size of the output profile
; imsize : [xs,ys] size of the image (not calculated here, just for
; the scale)
; fovpix : fov angle of one pixel in rad 
; -- observer position and attitude
; obspos : [x,y,z] position of the observer in the Sun basis
; obslonlat : [lon,lat,height] position of the observer in Carrington
;             coordinate. If set, then obspos is ignored. The optical
;             axis always points toward the Sun center. Use obsang to
;             change telescope orientation.
; obsang : [ax,ay,az] orientation of the observer, 
;          z is the optical axis 
; -- Ne position and attitude
; nepos : [x,y,z] position of the Ne reference in the Sun basis
; neang : [ax,ay,az] orientation of the Ne
; -- LOS params
; losnbp : number of step for the integration along the LOS
; losrange : [lstart,lend] range for the integration along the LOS
;            in Rsun. The origin of the LOS is the orthogonal
;            projection of the Sun cntr on that LOS.
; modelid : model id number
; modparam : parameters of the model
; save : put path and filename in that variable (without extention) 
;        if you want to save the results in a .fits binary table.
; file : density cube filename for density 4 and 5
; fakelasco : put fake lasco header information in the fits header
; quiet : disable display of raytracing parameters
; unload : set to force unloading the C++ program (actualy .so lib):
;          useful when recompiling the C program
; hlonlat : [Hlon,Hlat,Hrot] heliographic lon and lat of the center of
; the disk, rotation angle corresponding to the projection of the
; north pole, counterclockwise.
; xdr : save into xdr format instead of fits table. 'save' keyword
;       must be set for xdr to take effect.
; secchiab : 'A' or 'B', to select Ahead or Behind spacecraft, for
;            secchi only
; instr : txt instrument preset:
; -- Instrument FOV preset
; c1, c2, c3 : lasco C1, C2, C3
; cor1, cor2 : Secchi Cor1, Cor2
; hi1, hi2 : Secchi Hi1, Hi2
; limbdark : limb darkening coeff: default 0.58
;
; OUTPUTS:
;  sbtot : structure for the total brightness profile
;  sbpol : structure for the polarized brightness profile
;  snetot : structure for the LOS integrated electron density profile
;
;-


; ---- define keyword default values
;      and cast every inputs to avoid C crash with bad type entry
if n_elements(imsize) eq 0 then imsize=[64L,64] else imsize=long(imsize)
if n_elements(radius) eq 0 then radius=30. else radius=float(radius)
if n_elements(cntr) ne 0 then cntr=float(cntr)
if n_elements(nbang) eq 0 then nbang=360L else nbang=long(nbang)
if n_elements(fovpix) eq 0 then fovpix=2./(64.)*!dtor else fovpix=float(fovpix)
if n_elements(obspos) eq 0 then obspos=[0.,0,-214] else obspos=float(obspos)
if n_elements(obsang) eq 0 then obsang=[0.,0,0] else obsang=float(obsang)
if n_elements(nepos) eq 0 then nepos=[0.,0,0] else nepos=float(nepos)
if n_elements(neang) eq 0 then neang=[0.,0,0] else neang=float(neang)
if n_elements(losnbp) eq 0 then losnbp=64L else losnbp=long(losnbp)
if n_elements(losrange) eq 0 then losrange=[-3.2,3.2] else losrange=float(losrange)
if n_elements(modelid) eq 0 then modelid=1L else modelid=long(modelid)
if n_elements(modparam) eq 0 then modparam=0. else modparam=float(modparam)
if n_elements(quiet) eq 0 then quiet=0L else quiet=1L
if n_elements(hlonlat) eq 0 then hlonlat=[0.,0,0] else hlonlat=float(hlonlat)
if n_elements(secchiab) eq 0 then secchiab='A' else begin
    secchiab=strupcase(secchiab)
    if secchiab ne 'A' and secchiab ne 'B' then message,'secchiab keyword must be either ''A'' or ''B'''
endelse
if n_elements(limbdark) eq 0 then limbdark=0.58 else limbdark=float(limbdark)
if n_elements(obslonlat) ne 0 then begin 
    obslonlat=float(obslonlat)
    obslonlatflag=1L
endif else begin
    obslonlat=[0.,0,0]
    obslonlatflag=0L
endelse


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
rtinstrpresets,instr,imsize,fovpix,obsang,secchiab=secchiab

; -- set center to the sun position if not specified by user
if n_elements(cntr) eq 0 then $
  cntr=rtcalccntr(fovpix,obsang,imsize)

; ---- init the outputs
btot=fltarr(nbang)
bpol=fltarr(nbang)
netot=fltarr(nbang)
crpix=fltarr(2)

case modelid of
    4 : begin
        print,'Model 4:'
        print,'Loading YM density...'
        print,'Raytracing will be done with trilinear intepolation'
        load_dens,dens,rco,phico,thetaco,file=file
        modparam=[rco,phico,thetaco,reform(dens,n_elements(dens))]
        dens=0b
    end
    5 : begin
        print,'Model 5:'
        print,'Loading YM density...'
        print,'Raytracing will be done with nearest neighbor'
        load_dens,dens,rco,phico,thetaco,file=file
        modparam=[rco,phico,thetaco,reform(dens,n_elements(dens))]
        dens=0b
    end
    else :
endcase

rtinitenv
starttime=systime(1)
s=call_external(getenv('RT_LIBFILE'),$
                'rtwlcirc',$
                imsize[0],imsize[1],$
                fovpix,$
                obspos,obsang,$
                nepos,neang,$
                losnbp,losrange,modelid,$
                btot,bpol,netot,modparam,$
                crpix,nbang,radius,quiet,hlonlat,$
                cntr,limbdark,obslonlat,obslonlatflag,unload=unload)

if not quiet then begin
    print,'Seconds ellapsed :'
    print,systime(1)-starttime
endif

; -- sun radius in pix 
solar_r=abs(atan(1./obspos[2]))/fovpix

mkhdr,h0,btot
sxaddpar,h0,'COMMENT','Author: A.Thernisien USRA-NRL'
; -- X simu -> Y image, Y simu -> X image
sxaddpar,h0,'CRPIX1',crpix[1]*solar_r+(float(imsize[0])-1)/2.,'X Sun center (pix)'
sxaddpar,h0,'CRPIX2',crpix[0]*solar_r+(float(imsize[1])-1)/2.,'Y Sun center (pix)'
sxaddpar,h0,'SOLAR_R',solar_r,'pix'
sxaddpar,h0,'CDELT1',fovpix*!radeg*3600.,'arcsec'
sxaddpar,h0,'CDELT2',fovpix*!radeg*3600.,'arcsec'
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
    else : rtdetec=0
endcase
sxaddpar,h0,'RTDETEC',rtdetec,'raytrace detector option'

; -- put fake date
;sxaddpar,h0,'TIME_OBS','00:00:00.000'
;sxaddpar,h0,'DATE_OBS','2004/01/22'

; -- put fake LASCO header
if keyword_set(fakelasco) then begin
    sxaddpar,h0,'FILEORIG','040124_023058.img'
   sxaddpar,h0,'DATE','2004/02/19 04:38:31.577'
   sxaddpar,h0,'DATE-OBS','2004/01/24'
   sxaddpar,h0,'TIME-OBS','02:30:05.451'
   sxaddpar,h0,'P1COL',20
   sxaddpar,h0,'P1ROW',1
   sxaddpar,h0,'P2COL',1043
   sxaddpar,h0,'P2ROW',1024
   sxaddpar,h0,'VERSION',2
   sxaddpar,h0,'EXPTIME',1.
   sxaddpar,h0,'EXP0',23.000000
   sxaddpar,h0,'EXPCMD',23.000000
   sxaddpar,h0,'EXP1',1.8842800
   sxaddpar,h0,'EXP2',2.0659200
   sxaddpar,h0,'EXP3',2.5966800
   sxaddpar,h0,'TELESCOP','SOHO'
   sxaddpar,h0,'INSTRUME','LASCO'
   sxaddpar,h0,'DETECTOR','C2'
   if keyword_set(c3image) then sxaddpar,h0,'DETECTOR','C3'
   sxaddpar,h0,'READPORT','C'
   sxaddpar,h0,'SUMROW',0
   sxaddpar,h0,'SUMCOL',0
   sxaddpar,h0,'LEBXSUM',1
   sxaddpar,h0,'LEBYSUM',1
   if imsize[0] eq 512 then begin
       sxaddpar,h0,'LEBXSUM',2
       sxaddpar,h0,'LEBYSUM',2
   endif
   if imsize[0] eq 256 then begin
       sxaddpar,h0,'LEBXSUM',4
       sxaddpar,h0,'LEBYSUM',4
   endif
   if imsize[0] eq 128 then begin
       sxaddpar,h0,'LEBXSUM',8
       sxaddpar,h0,'LEBYSUM',8
   endif

   sxaddpar,h0,'SHUTTR',0
   sxaddpar,h0,'LAMP',0
   sxaddpar,h0,'FILTER','Orange'
   if keyword_set(c3image) then sxaddpar,h0,'FILTER','Clear'
   sxaddpar,h0,'POLAR','Clear'
   sxaddpar,h0,'LP_NUM','Normal'
   sxaddpar,h0,'OS_NUM',3389
   sxaddpar,h0,'IMGCTR',433
   sxaddpar,h0,'IMGSEQ',0
   sxaddpar,h0,'COMPRSSN','X%'
   sxaddpar,h0,'HCOMP_SF',64
   sxaddpar,h0,'FP_WL_UP',0.0000000
   sxaddpar,h0,'FP_WL_CM',0.0000000
   sxaddpar,h0,'WAVELENG',0.0000000
   sxaddpar,h0,'FP_ORDER',0
   sxaddpar,h0,'M1_PZ1',0
   sxaddpar,h0,'M1_PZ2',0
   sxaddpar,h0,'M1_PZ3',0
   sxaddpar,h0,'MID_DATE',53028
   sxaddpar,h0,'MID_TIME',9018.2500
   sxaddpar,h0,'PLATESCL',11.900000
   if keyword_set(c3image) then sxaddpar,h0,'PLATESCL',56.
   sxaddpar,h0,'OFFSET',600
   sxaddpar,h0,'IMAGE_CT',433
   sxaddpar,h0,'SEQ_NUM',0
   sxaddpar,h0,'OBT_TIME',1.4536026e+09
   sxaddpar,h0,'R1COL',20
   sxaddpar,h0,'R1ROW',1
   sxaddpar,h0,'R2COL',1043
   sxaddpar,h0,'R2ROW',1024
   sxaddpar,h0,'BUNIT','       0'
   sxaddpar,h0,'EFFPORT','C'
   sxaddpar,h0,'RECTIFY','TRUE'
   sxaddpar,h0,'CRVAL1',0.0000000
   sxaddpar,h0,'CRVAL2',0.0000000
   sxaddpar,h0,'CROTA',0.0000000
   sxaddpar,h0,'XCEN',0.0000000
   sxaddpar,h0,'YCEN',0.0000000
   sxaddpar,h0,'CROTA1',180.00000
   sxaddpar,h0,'CROTA2',180.00000
   sxaddpar,h0,'CTYPE1','SOLAR-X'
   sxaddpar,h0,'CTYPE2','SOLAR-Y'
   sxaddpar,h0,'CUNIT1','ARCSEC'
   sxaddpar,h0,'CUNIT2','ARCSEC'
   sxaddpar,h0,'SECTOR','       0'
   sxaddpar,h0,'RSUN',0.0000000
   sxaddpar,h0,'NMISSING',0
   sxaddpar,h0,'MISSLIST','       0'
endif

sxaddpar,h0,'RADIUS',radius,'rad'
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



hbtot=h0
sxaddpar,hbtot,'UNITS','Bsun'
sxaddpar,hbtot,'COMMENT','Total brightness'

hbpol=h0
sxaddpar,hbpol,'UNITS','Bsun'
sxaddpar,hbpol,'COMMENT','Polarized brightness'

hnetot=h0
sxaddpar,hnetot,'UNITS','e- . cm-3'
sxaddpar,hnetot,'COMMENT','Integrated electron density'

revision=getcvsrevision('$Revision: 1.3 $')

sbtot={imsize:imsize,nbang:nbang,radius:radius,fovpix:fovpix,obspos:obspos,obsang:obsang,nepos:nepos,neang:neang,losnbp:losnbp,losrange:losrange,modelid:modelid,crpixprg:crpix,im:btot,fitshead:hbtot,hlonlat:hlonlat,rtdetec:rtdetec,densfile:(n_elements(file) ne 0 ? file : ''),revision:revision,secchiab:secchiab}

sbpol={imsize:imsize,nbang:nbang,radius:radius,fovpix:fovpix,obspos:obspos,obsang:obsang,nepos:nepos,neang:neang,losnbp:losnbp,losrange:losrange,modelid:modelid,crpixprg:crpix,im:bpol,fitshead:hbpol,hlonlat:hlonlat,rtdetec:rtdetec,densfile:(n_elements(file) ne 0 ? file : ''),revision:revision,secchiab:secchiab}

snetot={imsize:imsize,nbang:nbang,radius:radius,fovpix:fovpix,obspos:obspos,obsang:obsang,nepos:nepos,neang:neang,losnbp:losnbp,losrange:losrange,modelid:modelid,crpixprg:crpix,im:netot,fitshead:hnetot,hlonlat:hlonlat,rtdetec:rtdetec,densfile:(n_elements(file) ne 0 ? file : ''),revision:revision,secchiab:secchiab}

if n_elements(save) ne 0 then begin
    print,'Saving the results'

    if xdr then begin
        savestruct,save+'btot.xdr',sbtot
        savestruct,save+'bpol.xdr',sbpol
        savestruct,save+'netot.xdr',snetot
    endif else begin
        mwrfits,sbtot,save+'btot.fits',/create
        mwrfits,sbpol,save+'bpol.fits',/create
        mwrfits,snetot,save+'netot.fits',/create
    endelse
endif
return
end
;
; CVSLOG:
;  $Log: rtwlcirc.pro,v $
;  Revision 1.3  2006/11/09 19:43:30  thernis
;  Implement obslonlat keyword to allow positioning the observer using Carrington spherical coordinates.
;
;  Revision 1.2  2006/10/30 21:52:40  thernis
;  Change the environment variable that points to the shared object library.
;
;
