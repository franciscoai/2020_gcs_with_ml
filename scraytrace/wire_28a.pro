;+
; $Id: wire_28a.pro,v 1.1 2006/09/08 16:03:23 nathan Exp $
;
; PURPOSE:
;  Generates a wire frame for model 28
; CATEGORY:
;  raytracing, visualization, 3d
;-
pro wire_22,xx,yy,zz
;
;  Curved Cylindrical Shell Model
;
;  electron density calculation:
;  rr = sqrt(x*x+(z-h0)^2)
;  If (ABS(y) LE len) THEN BEGIN
;     IF (ABS(rr) LE d0 then nel=nemin + (nemax-nemin)*(z-h0-d-d0)/(2*d+2*d0)
;                       else nel=0
;  ENDIF ELSE nel=0
;
;
; Parameters:
h0 = 8.		; height of leading edge of cylinder at the nose
d  = 1.		; radius of cylinder
d0 = 0.3	; 1/2 thickness of shell
r0 = 8.		; radius of curvature of the cylinder
phi = 45.	; 1/2 angle of the CME
len = 7.	; length of cylinder

ny = 21		; number of radial slices
nt = 61		; number of theta values in wire model
nd = 3		; number of radii in the shell
y0 = len/2.
h1 = h0-2*d0-d	; height of radius of cylinder
p  = -r0+SQRT(2*r0*r0+y0*y0)		;
h2 = h1-p							; height of center of cylinder curvature
;IF (h2 LE r0)  THEN BEGIN
;	PRINT,'ERROR:  Center of cylinder curvature must be higher than radius of curvature'
;	PRINT,'INPUTS: h0, d0, r0: ',h0,d0,r0
;	RETURN
;END
;ph = phi*!DTOR
;a = h2*(1.-cos(ph)*cos(ph))
;a = SQRT(a*a+r0*r0)-a
;psi = acos(a/r0)			; angular span of cylinder around center of curvature
;delwidth = 2*psi/(ny-1)	; spacing of wire along cylinder
;ps0 = -psi
beta = 180*!dtor-2*atan(y0,r0)
print,beta/!DTOR
delR = d0/(nd-1)	; spacing of wire in shell
th = !DTOR*FINDGEN(nt+1)*360/nt
dely = len/(ny-1)
hh0 = h0-2*d0-d
hh1 = hh0-r0
delhh = 0.5*(hh0-hh1)/(ny-1.)
halfny = .5*(ny-1)
delbeta = 0.5*beta/(ny-1)
hght = fltarr(ny)
ang = fltarr(ny)
fac = COS(delbeta*halfny)
IF (fac LT 0.0001) THEN FAC=1
FOR i=0,ny-1 DO BEGIN	; loop on each x-section of cylinder
;	ps = ps0+i*delwidth		; angle of wire section
;	g0 = d-d0
	hh=hh0-abs(i-halfny)*delhh
	hh = hh0-p*(1-COS(abs(i-halfny)*delbeta))/(1-fac)
	hght(i) = hh
	ang(i) = abs(i-halfny)*delbeta
	;print,hh,ang(i),COS(abs(i-halfny)*delbeta)/fac
	y1 = -y0+i*dely
	FOR j=0,nd-1 DO BEGIN	; loop on each radius in cylinder shell
    	rr = d+j*delR
        x = rr*COS(th)
        z = rr*SIN(th)+hh
        y = x-x+y1
        IF i EQ 0  THEN BEGIN
           xx = [x]
           yy = [y]
           zz = [z]
        ENDIF ELSE BEGIN
           xx = [xx,x]
           yy = [yy,y]
           zz = [zz,z]
        ENDELSE
    ENDFOR
ENDFOR
RETURN
END
