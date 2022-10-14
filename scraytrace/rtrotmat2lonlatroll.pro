;+
;  $Id: rtrotmat2lonlatroll.pro,v 1.1 2006/09/08 16:03:22 nathan Exp $
;
; PURPOSE:
;  Get the lon, lat and roll angle from a rotation matrix 
;
; CATEGORY:
;  raytracing, mathematics, 3d, geometry
;
; INPUTS:
;  m : 3x3 rotation matrix
;
; OUTPUTS:
;  [longitude,latitude,roll] in rad
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
function rtrotmat2lonlatroll,m


lat=asin(-m[2,0])
c1=cos(lat)
if (ABS(c1) lt 1.0e-6) then begin
    lon  = -ATAN(m[0,1], m[0,2])
    roll = 0.0
endif else begin
    lon  = -ATAN( m[2,1], m[2,2])
    roll = -ATAN( m[1,0], m[0,0])
endelse


RETURN, [lon,lat,roll]
end
