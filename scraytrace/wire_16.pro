;+
; $Id: wire_16.pro,v 1.1 2006/09/08 16:03:22 nathan Exp $
; 
; PURPOSE:
;  Generates a wire frame for model 16
; CATEGORY:
;  raytracing, visualization, 3d
;-

pro wire_16,x,y,z

;  Generates a wire frame for model 16
;
nr = 19
nth = 13
d0 = 0.2
dr = (5.-1.)/nr
r = dr*findgen(nr)+1
delth = 360/(nth-1)
th = findgen(nth)*delth*!dtor
x = fltarr(nth*nr)
y = x
z = x
f = x+1
one_over_e = 1+d0
coef = [-1,-2,-3,-4]
c=[1.34e4,1.15e6,-6.022e6,5.577e7]
nc = N_ELEMENTS(c)-1
nel = 0
FOR j=0,nc-1 DO nel = nel+c(nc-1-j)*EXP(r(0)*coef(nc-1-j))
nel0 = nel
FOR i=0,nr-1 DO BEGIN
    nel = 0.
    FOR j=0,nc-1 DO nel = nel+c(nc-1-j)*r(i)^coef(nc-1-j)
    nel = EXP(-nel/nel0)
    nel = 2/r(i)^2
    u = d0*COS(th)
    v = d0*SIN(th)
    x(i*nth) = u*nel
    y(i*nth) = v*nel
    z(i*nth) = REPLICATE(r(i),nth)
ENDFOR

z = -z

RETURN
END
