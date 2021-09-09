;+
;  $Id: latlonheight2xyz.pro,v 1.1 2006/11/22 16:17:17 thernis Exp $
; 
; PURPOSE: 
;  Convert lat lon height coordinate into cartesian
;
; CATEGORY:
;  3d
;
; INPUTS:
;  latlonr : [latitude,longitude,height]
;  degrees : set if the angles are in degrees. Default is in rad
;
; OUTPUTS:
;  [x,y,z]
;
;-

function latlonheight2xyz,latlonr,degrees=degrees

if keyword_set(degrees) then conv=!dpi/180d else conv=1d

xxx=latlonr[2]*cos(latlonr[0]*conv)*cos(latlonr[1]*conv)
yyy=latlonr[2]*cos(latlonr[0]*conv)*sin(latlonr[1]*conv)
zzz=latlonr[2]*sin(latlonr[0]*conv)

return,[xxx,yyy,zzz]
end
