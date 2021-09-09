;+
; NAME:
;  dispraytrace.pro
;
; PURPOSE:
;  display a raytracing image with the lon lat grid overplotted
;
; CATEGORY:
;  visualization, raytrace
;
; CALLING SEQUENCE:
;  dispraytrace,rebin(alog10(sbt1.im > 1e-17),512,512),mmlon1*!radeg,mmlat1*!radeg,10.,xstyle=1,ystyle=1
;
; INPUTS:
;  im : image to display
;  mmlon, mmlat : [min,max] longitude and latitude from raytracewl program 
;  angstep : step angle in the same units than mmlo and mmlat
; 
; KEYWORDS:
;  posxy : [x,y] position of the image on the output window
;  all keywords for plot
;
; OUTPUTS:
;  Display on screen
;
;  $Id: dispraytrace.pro,v 1.1 2006/09/08 16:03:20 nathan Exp $
;
;-
pro dispraytrace,im,mmlon,mmlat,angstep,_extra=e,posxy=posxy


sx=(size(im))(1)
sy=(size(im))(2)

if n_elements(posxy) eq 0 then posxy=[60.,50] else posxy=float(posxy)
posx=posxy(0)	;x offset
posy=posxy(1)	;y offset
swx=!d.x_size	;x win size
swy=!d.y_size	;y win size

xmold=!x.margin
ymold=!y.margin
prold=!p.region

!x.margin=0
!y.margin=0
!p.region=[(posx-1)/swx,(posy-1)/swy,(posx+sx)/swx,(posy+sy)/swy]

erase

tvscl,im,posx,posy,/device

oplotlonlatgrid,mmlon,mmlat,angstep,/noerase,_extra=e


;!x.margin=xmold
;!y.margin=ymold
;!p.region=prold



return
end
