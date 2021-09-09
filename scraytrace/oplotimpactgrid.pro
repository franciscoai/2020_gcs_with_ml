;+
; NAME:
;  oplotimpactgrid.pro
;
; PURPOSE:
;  Overplot impact parameter or r parameter on images from raytracing
;
; CATEGORY:
;  visualization, raytracing
;
; CALLING SEQUENCE:
;  oplotimpactgrid,rebin(rrr1,512,512),levels=[1,(findgen(7)+2)*1],c_labels=replicate(1,10),ystyle=5,xstyle=5
;
; INPUTS:
;  rho : image of distance from the Sun center 
;  
; OPTIONAL INPUTS:
;
; KEYWORDS:
;  erase : erase befor plotting
;  all keywords for contour
;
; OUTPUTS:
;
; HISTORY:
;  A.T. , 05-Oct-2004
;
; CVS:
;  $Id: oplotimpactgrid.pro,v 1.1 2006/09/08 16:03:20 nathan Exp $
;
;-
pro oplotimpactgrid,rho,_extra=e,erase=erase


contour,rho,xmargin=[0,0],ymargin=[0,0],noerase=(keyword_set(erase) ? 0B : 1B),ticklen=-0.01,xstyle=1,ystyle=1,_extra=e


return
end
