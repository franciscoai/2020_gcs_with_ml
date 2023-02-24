;+
;  $Id: rtrotmat2rxryrz.pro,v 1.1 2007/05/11 20:55:00 thernis Exp $
;
; PURPOSE:
;  Get the rx,ry,rz angles from a rotation matrix 
;
; CATEGORY:
;  raytracing, mathematics, 3d, geometry
;
; INPUTS:
;  m : 3x3 rotation matrix
;
; OUTPUTS:
;  [rx,ry,rz] in rad
;
; DESCRIPTION:
;  - See function angle3123 in dragger.pro
;
;  - The rotation matrix is assumed to be computed in that order:
;    R(-lon,x).R(lat,y).R(-roll,z)
;    with R(angle,axis) the rotation matrix around axis "axis" with an
;    angle "angle".
;  - Coordinate system orientation is assumed to follow 
;    raytracewl software package convention, that, I know, is not
;    usual:
;    X : vertical axis
;    Y : horizontal axis
;    Z : depth axis, perpendicular to the plane of the sky
;    
;-
function rtrotmat2rxryrz,m


ry=asin(m[0,2])
c1=cos(ry)
if (ABS(c1) lt 1.0e-6) then begin
    rz  = -ATAN(m[2,1], m[2,0])
    rx = 0.0
endif else begin
    rz  = -ATAN( m[0,1], m[0,0])
    rx = -ATAN( m[1,2], m[2,2])
endelse


RETURN, [rx,ry,rz]
end
