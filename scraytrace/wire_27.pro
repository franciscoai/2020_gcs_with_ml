;+
; $Id: wire_27.pro,v 1.1 2006/09/08 16:03:22 nathan Exp $
; 
; PURPOSE:
;  Generates a wire frame for model 27
; CATEGORY:
;  raytracing, visualization, 3d
;-

pro wire_27,xx,yy,zz,modparam=modparam
;
;  Spherical Shell Model
;
; Parameters:
if n_elements(modparam) eq 0 then begin
    r0 = 8.                     ; height of center of sphere
    d  = 2.                     ; radius of sphere
    d0 = 0.3                    ; 1/2 thickness of shell
endif else begin
    r0 = modparam[0]
    d = modparam[1]
    d0 = modparam[2]
end

;
;  electron density calculation:
;  rr = sqrt(x*x+y*y+(z-r0)2)
;  if ABS(rr) LE d0 then nel=nemin + (nemax-nemin)*(z-r0-d-d0)/(2*d+2*d0)
;                   else nel=0
;

nd = 20		; number of radial slices
nt = 29		; number of theta values in wire model
delR = 2*(d0+d)/nd
th = !DTOR*FINDGEN(nt+1)*360/nt
h0 = -(d+d0)

h0in = d0-d
r2in = (d-d0)^2
delRin = 2*(d-d0)/nd

FOR i=0,nd DO BEGIN
    h = h0+i*delR
    rr = SQRT (((d+d0)^2)-h*h)
    z = rr*SIN(th(0:29))+r0
    y = rr*COS(th(0:29))
    x = y-y+h
    IF i EQ 0  THEN BEGIN
       xx = [x]
       yy = [y]
       zz = [z]
    ENDIF ELSE BEGIN
       xx = [xx,x]
       yy = [yy,y]
       zz = [zz,z]
   ENDELSE

    hin = h0in+i*delRin
    rrin = SQRT (r2in-hin*hin)
    yin = rrin*COS(th(0:29))
    zin = rrin*SIN(th(0:29))+r0
    xin = yin-yin+hin
    xx = [xx,reverse(xin)]
    yy = [yy,reverse(yin)]
    zz = [zz,reverse(zin)]

ENDFOR

zz = -zz

; h0in = d0-d
; r2in = (d-d0)^2
; delRin = 2*(d-d0)/nd
; FOR i=0,nd DO BEGIN
;     hin = h0in+i*delRin
;     rrin = SQRT (r2in-hin*hin)
;     yin = rrin*COS(th(0:29))
;     zin = rrin*SIN(th(0:29))+r0in
;     xin = yin-yin+hin
;     xx = [xx,reverse(x)]
;     yy = [yy,reverse(y)]
;     zz = [zz,reverse(z)]
; ENDFOR


RETURN
END


