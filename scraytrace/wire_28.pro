;+
; $Id: wire_28.pro,v 1.1 2006/09/08 16:03:23 nathan Exp $
; 
; PURPOSE:
;  Generates a wire frame for model 28
; CATEGORY:
;  raytracing, visualization, 3d
;-

pro wire_28,xx,yy,zz,desc, modparam=modparam
;
;  Cylindrical Shell Model
;
; Parameters:
if n_elements(modparam) eq 0 then begin
    r0 = 8.                     ; height of center of sphere
    d  = 2.                     ; radius of cylinder
    len  = 2.                   ; length of cylinder in each direction
    d0 = .3                    ; 1/2 thickness of shell
endif else begin
    r0 = modparam[0]
    d = modparam[1]
    len = modparam[2]
    d0 = modparam[3]
end
;
;  electron density calculation:
;  rr = sqrt(x*x+(z-r0)2)
;  If (ABS(y) LE len) THEN BEGIN
;     IF (ABS(rr) LE d0 then nel=nemin + (nemax-nemin)*(z-r0-d-d0)/(2*d+2*d0)
;                       else nel=0
;  ENDIF ELSE nel=0
;

ny = 10		; number of radial slices
nt = 20		; number of theta values in wire model
nd = 2		; number of radii in the shell
dely = 2.*len/(ny-1.)
IF (nd GE 1) THEN delR = d0/nd ELSE delR=0
th = !DTOR*FINDGEN(nt)*360/(nt-1)
h0 = -len
first = 1
FOR i=0,ny-1 DO BEGIN
    h = h0+i*dely
    g0 = d-d0
    FOR j=0,nd DO BEGIN
    	rr = g0+j*delR
        x = rr*COS(th)
        z = rr*SIN(th)+r0
        y = x-x+h
        IF first EQ 1  THEN BEGIN
           xx = [x]
           yy = [y]
           zz = [z]
           first = 0
        ENDIF ELSE BEGIN
           xx = [xx,x]
           yy = [yy,y]
           zz = [zz,z]
        ENDELSE
    ENDFOR
ENDFOR

desc=intarr(ny*nt*(nd+1),5)
ij=0
nd=nd+1
FOR k=0,nd-1 DO BEGIN
    FOR i=0,ny-2 DO BEGIN
        FOR j=0,nt-2 DO BEGIN
            desc(ij,0) = 4
            desc(ij,1) = i*nt*nd+j+k*nt
            desc(ij,2) = i*nt*nd+j+1+k*nt
            desc(ij,4) = (i+1)*nt*nd+j+k*nt
            desc(ij,3) = (i+1)*nt*nd+j+1+k*nt
            ij = ij+1
        ENDFOR
        ;desc(ij,0) = 4
        ;desc(ij,1) = i*nt*nd+k*nt
        ;desc(ij,2) = i*nt*nd+nt-1        +k*nt
        ;desc(ij,4) = (i+1)*nt*nd+nt        +k*nt
        ;desc(ij,3) = (i+1)*nt*nd+nt-1      +k*nt
        ;ij = ij+1
    ENDFOR
ENDFOR

zz = -zz
desc=transpose(desc)

RETURN
END


.r wire_28.pro
wire_28,xx,yy,zz,dd
dragger,xx,yy,zz,/solid,polygons=dd
