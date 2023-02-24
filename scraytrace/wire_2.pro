;+
; $Id: wire_2.pro,v 1.1 2006/09/08 16:03:23 nathan Exp $
; 
; PURPOSE:
;  Generates a wire frame for model 2
; CATEGORY:
;  raytracing, visualization, 3d
;-

pro wire_2,x,y,z

;  Generates a wire frame for model 2
;
nr = 39			; number of radial points
nth = 13		; number of anglar points
delr = (20.-1.)/(nr-1)
scale = 0.2		; scale factor for display only (not used in ne calc)
rad = 0.5		; radius of circle at height r
r = findgen(nr)*delr+1
delth = 360/(nth-1)
th = findgen(nth)*delth*!dtor
x = fltarr(nth*nr)
y = x
z = x
f = x+1

FOR i=0,nr-1 DO BEGIN
    u = COS(th)*exp(-(r(i)-r(0))*scale)
    v = SIN(th)*exp(-(r(i)-r(0))*scale)
    x(i*nth) = u*rad
    y(i*nth) = v*rad
    z(i*nth) = REPLICATE(r(i),nth)
ENDFOR

; sphere = Obj_New('Orb')
; sphere->Scale, 1., 1., 1.
; oSymbol = OBJ_NEW('IDLgrSymbol', sphere)
; xplot3d,[0.,0.],[0.,0.],[0.,0.],line=0,symbol=oSymbol,$
;   neangx=neangx,neangy=neangy,neangz=neangz,$
;   XRANGE=[-20,20],YRANGE=[-20,20],ZRANGE=[-20,20]


; xplot3d,x,y,z,neangx=neangx,neangy=neangy,neangz=neangz, /OVERPLOT

; dragger,x,y,z

RETURN
END
