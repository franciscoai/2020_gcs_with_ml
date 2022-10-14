;+
; $Id: wire_6.pro,v 1.1 2006/09/08 16:03:23 nathan Exp $
; 
; PURPOSE:
;  Generates a wire frame for model 6
; CATEGORY:
;  raytracing, visualization, 3d
;-
pro wire_6,x,y,z  ;,neangx=neangx,neangy=neangy,neangz=neangz ;,x,y,z,f

;  Generates a wire frame for model 6 - simple streamer belt
;
nr = 9
nth = 100
nd = 7
d0 = 0.3
dr = (10.-1.)/nr
nd = 1
r = dr*findgen(nr)+1
delth = 360./(nth-1)
th = findgen(nth)*delth*!dtor
x = fltarr(nth,nd,nr)
y = x
z = x
f = x+1
d = (FINDGEN(nd)-nd/2)*0.1
FOR k=0,nth-1 DO BEGIN
FOR i=0,nr-1 DO BEGIN
    FOR j=0,nd-1 DO BEGIN
        phi0 = !pi/2+asin(0.5*cos(2*th(k)))
        phi = phi0;asin(d(j)/r(i))+phi0
        u = r(i)*cos(th(k))*sin(phi)
        v = r(i)*sin(th(k))*sin(phi)
        w = r(i)*cos(phi)
        x(k,j,i) = u
        y(k,j,i) = v
        z(k,j,i) = w
    ENDFOR
ENDFOR
ENDFOR
x = REFORM(x,nth*nr*nd)
y = REFORM(y,nth*nr*nd)
z = REFORM(z,nth*nr*nd)

;dragger,x,y,z,neangx=neangx,neangy=neangy,neangz=neangz
;
;sphere = Obj_New('Orb')
;sphere->Scale, 1., 1., 1.
;oSymbol = OBJ_NEW('IDLgrSymbol', sphere)
;xplot3d,[0.,0.],[0.,0.],[0.,0.],line=0,symbol=oSymbol,$
;  neangx=neangx,neangy=neangy,neangz=neangz,$
;  XRANGE=[-20,20],YRANGE=[-20,20],ZRANGE=[-20,20]
;
;xplot3d,x,y,z,neangx=neangx,neangy=neangy,neangz=neangz, /OVERPLOT

RETURN
END
