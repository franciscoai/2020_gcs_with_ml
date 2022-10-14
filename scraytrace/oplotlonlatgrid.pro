;+
; NAME:
;  oplotlonlatgrid.pro
;
; PURPOSE:
;  Overplot a longitude latidute grid on images from raytracing
;
; CATEGORY:
;  visualization, raytracing
;
; CALLING SEQUENCE:
;  oplotlonlatgrid,mmlon,mmlat,10.
;
; INPUTS:
;  mmlon : [min,max] longitude (x axis)
;  mmlat : [min,max] latitude (y axis)
;  stepang : step angle of the grid in the same units than mmlon and mmlat
;
; OPTIONAL INPUTS:
;
; KEYWORDS:
;  all keywords for plot
;
; OUTPUTS:
;
; HISTORY:
;  A.T. , 05-Oct-2004
;
; CVS:
;  $Id: oplotlonlatgrid.pro,v 1.1 2006/09/08 16:03:20 nathan Exp $
;-
pro oplotlonlatgrid,mmlon,mmlat,stepang,_extra=e

cntrlon=(mmlon[0]+mmlon[1])/2.
cntrlat=(mmlat[0]+mmlat[1])/2.

nblonp=long((mmlon[1]-cntrlon)/stepang)+1
nblonn=long((cntrlon-mmlon[0])/stepang)+1

nblatp=long((mmlat[1]-cntrlat)/stepang)+1
nblatn=long((cntrlat-mmlat[0])/stepang)+1


;plot,mmlon,mmlat,xmargin=[0,0],ymargin=[0,0],/nodata,xstyle=nomargin,ystyle=nomargin,ticklen=-0.01,/noerase
plot,mmlon,mmlat,/nodata,_extra=e


lon=cntrlon
for i=0,nblonp-1 do begin
    plots,[[lon,mmlat[0]],[lon,mmlat[1]]],line=(i gt 0 ? 1 : 3)
    lon=lon+stepang
endfor
lon=cntrlon-stepang
for i=1,nblonn-1 do begin
    plots,[[lon,mmlat[0]],[lon,mmlat[1]]],line=1
    lon=lon-stepang
endfor


lat=cntrlat
for i=0,nblatp-1 do begin
    plots,[[mmlon[0],lat],[mmlon[1],lat]],line=(i gt 0 ? 1 : 3)
    lat=lat+stepang
endfor
lat=cntrlat-stepang
for i=1,nblatn-1 do begin
    plots,[[mmlon[0],lat],[mmlon[1],lat]],line=1
    lat=lat-stepang
endfor



return
end
