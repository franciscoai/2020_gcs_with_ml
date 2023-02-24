; ---- get the pixel position given a point in space and position and
;      orientation of the basis

pro rtgetprojection,posin,posout,imsize=imsize,$
                    fovpix=fovpix,$
                    obspos=obspos,obsang=obsang,$
                    nepos=nepos,neang=neang,$
                    c2image=c2image,c3image=c3image,$
                    cor1=cor1,cor2=cor2,$
                    hi1=hi1,hi2=hi2,c1fov=c1fov,$
                    quiet=quiet,hlonlat=hlonlat,$
                    clip=clip,instr=instr

;+
; PURPOSE:
;  Get the pixel position on the image given a point in space
;
; CATEGORY:
;  raytracing, 3d, visualization
;
; INPUTS:
; ! all the distances must be given in Rsun
; ! and the angles in rad
; posin : [*,3] position of the points in space, in Rsun
; imsize : [xs,ys] size of the output image
; fovpix : fov angle of one pixel in rad 
; -- observer position and attitude
; obspos : [x,y,z] position of the observer in the Sun basis
; obsang : [ax,ay,az] orientation of the observer, 
;          z is the optical axis 
; -- Ne position and attitude
; nepos : [x,y,z] position of the Ne reference in the Sun basis
; neang : [ax,ay,az] orientation of the Ne
; hlonlat : [Hlon,Hlat,Hrot] heliographic lon and lat of the center of
; the disk, rotation angle corresponding to the projection of the
; north pole, counterclockwise.
; clip : set if you don't want to compute the projection 
;        of the points behind the sun
;
; -- Instrument FOV preset
; c1, c2, c3 : lasco C1, C2, C3
; cor1, cor2 : Secchi Cor1, Cor2
; hi1, hi2 : Secchi Hi1, Hi2
; instr : can be one the instrument above, in txt format
;
; OUTPUTS:
; posout : [*,2] position of the points in the image, in pix
;
;  $Id: rtgetprojection.pro,v 1.2 2006/10/30 21:52:40 thernis Exp $
;-

if n_elements(posin) eq 0 then message,'posin variable cannot be empty !'
szposin=size(posin)
case szposin[0] of
    1 : begin
        if szposin[1] ne 3 then message,'Format of posin variable not good'
        posin=float(posin)
        nbpoints=1L    
    end
    2 : begin
        if szposin[1] ne 3 then message,'Format of posin variable not good'
        posin=float(posin)
        nbpoints=long(szposin[2])
    end
    else : message,'Format of posin variable not good'
endcase


if n_elements(imsize) eq 0 then imsize=[64L,64] else imsize=long(imsize)
if n_elements(fovpix) eq 0 then fovpix=2./(64.)*!dtor else fovpix=float(fovpix)
if n_elements(obspos) eq 0 then obspos=[0.,0,-214] else obspos=float(obspos)
if n_elements(obsang) eq 0 then obsang=[0.,0,0] else obsang=float(obsang)
if n_elements(nepos) eq 0 then nepos=[0.,0,0] else nepos=float(nepos)
if n_elements(neang) eq 0 then neang=[0.,0,0] else neang=float(neang)
if n_elements(quiet) eq 0 then quiet=0L else quiet=1L
if n_elements(hlonlat) eq 0 then hlonlat=[0.,0,0] else hlonlat=float(hlonlat)
if n_elements(clip) eq 0 then clip=0L else clip=1L



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
rtinstrpresets,instr,imsize,fovpix,obsang,secchiab=secchiab




;c2image=keyword_set(c2image)
;c3image=keyword_set(c3image)
;c1fov=keyword_set(c1fov)
;cor1=keyword_set(cor1)
;cor2=keyword_set(cor2)
;hi1=keyword_set(hi1)
;hi2=keyword_set(hi2)

;; ---- instrument FOV presets
;if c2image then begin
;    scale=t_param('C2','SCALE')
;    fovpix=scale*!dtor
;    ; -- the image is assumed to be square
;    imszratio=1024./float(imsize[0])
;    fovpix=scale*!dtor*imszratio

;    ; -- shift of the sun center
;    scntr=t_param('C2','CENTER')
;    shft=(scntr-[511.5,511.5])*scale*!dtor
;    obsang=[shft[0],-shft[1],0]
;endif

;if c3image then begin
;    scale=t_param('C3','SCALE')
;    fovpix=scale*!dtor
;    ; -- the image is assumed to be square
;    imszratio=1024./float(imsize[0])
;    fovpix=scale*!dtor*imszratio

;    ; -- shift of the sun center
;    scntr=t_param('C3','CENTER')
;    shft=(scntr-[511.5,511.5])*scale*!dtor
;    obsang=[shft[0],-shft[1],0]
;endif

;if c1fov then begin
;    scale=t_param('C1','SCALE')
;    fovpix=scale*!dtor
;    ; -- the image is assumed to be square
;    imszratio=1024./float(imsize[0])
;    fovpix=scale*!dtor*imszratio

;    ; -- shift of the sun center
;    scntr=t_param('C1','CENTER')
;    shft=(scntr-[511.5,511.5])*scale*!dtor
;    obsang=[shft[0],-shft[1],0]
;endif


;if cor1 then begin
;    ; -- assume 4 rsun radius FOV
;    ;    the image is assumed to be square
;    fovpix=atan(4.,abs(obspos[2]))*2./float(imsize[0])
;    ; -- the Sun center is assumed to be at the center of the image

;endif

;if cor2 then begin
;    ; -- assume 15 rsun radius FOV
;    ;    the image is assumed to be square
;    fovpix=atan(15.,abs(obspos[2]))*2./float(imsize[0])
;    ; -- the Sun center is assumed to be at the center of the image

;endif

;if hi1 then begin
;    ; -- assume 37.5 rsun radius FOV
;    ;    the image is assumed to be square
;    fovpix=atan(37.5,abs(obspos[2]))*2./float(imsize[0])

;    ; -- center : see Secchi Phase A concept study report, p3-38
;    ;obspos=[0,49.8,-204.01] ; see 8 nov 2004
;    obsang=[-13.28*!dtor,0,0] ; see 8 nov 2004
;endif

;if hi2 then begin
;    ; -- assume 129.6 rsun radius FOV
;    ;    the image is assumed to be square
;    fovpix=atan(129.6,abs(obspos[2]))*2./float(imsize[0])

;    ; -- center : see Secchi Phase A concept study report, p3-38
;    ;obspos=[0,200.,-64.0312] ; see 8 nov 2004
;    obsang=[-53.36*!dtor,0,0] ; see 8 nov 2004
;endif


posout=fltarr(2,nbpoints)


rtinitenv
starttime=systime(1)
s=call_external(getenv('RT_LIBFILE'),$
                'getprojectionidl',$
                imsize[0],imsize[1],$
                fovpix,$
                obspos,obsang,$
                nepos,neang,$
                posin,posout,hlonlat,nbpoints,clip,$
                /unload)

if quiet eq 0 then begin
    print,'Seconds ellapsed :'
    print,systime(1)-starttime
endif






return
end
;
; CVSLOG:
;  $Log: rtgetprojection.pro,v $
;  Revision 1.2  2006/10/30 21:52:40  thernis
;  Change the environment variable that points to the shared object library.
;
;
