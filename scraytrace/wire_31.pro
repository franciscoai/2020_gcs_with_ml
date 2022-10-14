;+
; $Id: wire_31.pro,v 1.1 2006/09/08 16:03:23 nathan Exp $
; 
; PURPOSE:
;  Generates a wire frame for model 31
; CATEGORY:
;  raytracing, visualization, 3d
;-

pro wire_31,x,y,z,grid,modparam=modparam
;
;       Graduated Curved Cylindrical Shell Model
;       Flux rope model
;		8/19/04.
;		Problems: The joint from the cone to the circular front doesn't match;
;		The top is now a hemisphere attached to the cone legs.
;
;       The size of the circular cross section of the shell increases uniformly
;       from the base to the top
;
;  electron density calculation:  (needs to be modified for this new model)
;  rr = sqrt(x*x+(z-h0)^2)
;  If (ABS(y) LE len) THEN BEGIN
;     IF (ABS(rr) LE d0 then nel=nemin + (nemax-nemin)*(z-h0-d-d0)/(2*d+2*d0)
;                       else nel=0
;  ENDIF ELSE nel=0
;
;
; Input Parameters:

if n_elements(modparam) eq 0 then begin
    rh = 14.          ; height of leading edge of cylinder at the nose
    dh  = 4.          ; diameter of cylinder at the top
    d0 = 0.3          ; 1/2 thickness of shell
    rc = 3.           ; radius of curvature of the cylinder
    rb = 2.           ; height of base of cylinder
    db = .1           ; diameter of cylinder at the base
endif else begin
    rh=modparam[2]
    dh=modparam[3]
    d0=modparam[4]
    rc=modparam[5]
    rb=modparam[6]
    db=modparam[7]
end

nh = 80        ; number of radial slices
nz = 40         ; number of angular values (zeta) around the cylinder
nd = 1          ; number of radii in the shell
;y0 = len/2.


r0 = rh-dh*0.5-rc        ; height of center of curvature of flux rope
r1 = SQRT (r0*r0-rc*rc)  ; height of transition of conical legs to hemisphere
theta = atan(rc,r1)       ; half angle of width of curved cylinder
r2 = r1;*COS(theta)       ; projected height of transition


IF (nd EQ 1)  THEN delR = 0 ELSE delR = d0/(nd-1)   ; spacing of wire in shell
del_h = (rh-.5*dh-rb)/(nh-1)
zeta = !DTOR*FINDGEN(nz)*360/nz
del_d = FLOAT(dh-db)/FLOAT(rh-.5*dh-rb)
ifirst = 1
FOR j=0,nd-1 DO BEGIN   ; loop on each radius in cylinder shell
rcinc = -d0+j*delR
rcinc=0
;
; do left side of cylinder
;
nhleft = nh
FOR i=0,nh-1 DO BEGIN   ; loop on each x-section of cylinder
    hgt = rb+i*del_h
    r = 0.5*(db+(hgt-rb)*del_d) ; radius of cylinder cross section
    r = r+rcinc ; shell thickness
    IF (hgt LT r2)  THEN BEGIN  ; process legs
        ;help,r
        u = -r*cos(zeta) + hgt*tan(theta)
           v = r*sin(zeta)
           w = hgt+FLTARR(nz)
           nwh = 0
        ENDIF ELSE BEGIN
           phi = -ACOS((hgt-r0)/rc)
           ;print,r,hgt,phi,hgt-r0,rc,(hgt-r0)/rc
           u = r*COS(zeta)
           v = r*SIN(zeta)
           w = fltarr(nz);
           rotmat =[[sin(phi),cos(phi)],[-cos(phi),sin(phi)]]
           rotvec = rotmat##[[u],[w]]
           u=REFORM(rotvec[*,0])- rc*SIN(phi)
           w=REFORM(rotvec[*,1])+r0+rc*cos(phi)
           wh = WHERE (w LE r1,nwh)
        ENDELSE
        IF (nwh EQ 0)  THEN BEGIN
           IF (ifirst EQ 1)  THEN BEGIN
              x = u
              y = v
              z = w
              ifirst = 0
           ENDIF ELSE BEGIN
              x = [x,u]
              y = [y,v]
              z = [z,w]
           ENDELSE
        ENDIF ELSE nhleft=nhleft-1
ENDFOR
;
; now do right side of cylinder
;
;print,'right side'
nhright = nh-1
FOR ii=1,nh-1 DO BEGIN  ; loop on each x-section of cylinder
    i = nh-1-ii                 ; do it backwards from other side (go down from top)
        hgt = rb+i*del_h
    r = 0.5*(db+(hgt-rb)*del_d) ; radius of cylinder cross section
    r = r+rcinc
    IF (hgt LE r2)  THEN BEGIN  ; process legs
       u = r*cos(zeta) - hgt*tan(theta);-r
           v = r*sin(zeta)
           w = hgt+FLTARR(nz)
           nwh=0
        ENDIF ELSE BEGIN
           phi = ACOS((hgt-r0)/rc)
           ;print,phi,hgt-r0,rc,(hgt-r0)/rc
           u = r*COS(zeta)
           v = r*SIN(zeta)
           w = FLTARR(nz)
           rotmat =[[sin(phi),cos(phi)],[-cos(phi),sin(phi)]]
           rotvec = rotmat##[[u],[w]]
           u=REFORM(rotvec[*,0]) - rc*SIN(phi)
           w=REFORM(rotvec[*,1])+r0+rc*cos(phi)
		   wh = WHERE (w LE r1,nwh)

        ENDELSE
        IF (nwh eq 0)  THEN BEGIN
            x = [x,u]
            y = [y,v]
            z = [z,w]
        ENDIF ELSE BEGIN
            nhright=nhright-1
            ;print,r1,nwh,i,w(wh)
        ENDELSE
ENDFOR

ENDFOR
xx = z
yy = x
zz = y

x=xx
y=yy
z=zz

;
;  Generate the solid points
;
grid = LONARR(nd*2*nh*nz,5)
np = 0
nh = nhleft+nhright
FOR k=0,nd-1 DO BEGIN
    FOR i=0,nh-2 DO BEGIN
        FOR j=0,nz-2 DO BEGIN
            grid(np,0)=4
            grid(np,1)=i*nz+j
            grid(np,2)=i*nz+j+1
            grid(np,3)=(i+1)*nz+j+1
            grid(np,4)=(i+1)*nz+j
            np = np+1
        ENDFOR
            grid(np,0)=4
            grid(np,1)=i*nz
            grid(np,2)=i*nz+nz-1
            grid(np,3)=(i+1)*nz+nz-1
            grid(np,4)=(i+1)*nz
            np = np+1

    ENDFOR
ENDFOR


grid = TRANSPOSE(grid)
RETURN
END
