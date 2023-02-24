;+
; $Id: neutlinecarposabs.pro,v 1.1 2007/07/31 15:19:35 thernis Exp $
; PURPOSE:
;  Returns the position of a point of the neutral line in the scraytrace abs coordinate system
;
; CVSLOG:
;  $Log: neutlinecarposabs.pro,v $
;  Revision 1.1  2007/07/31 15:19:35  thernis
;  De-embed from cmecloud.pro
;
;-


function neutlinecarposabs,lon,lat,halfang,tilt

c=cos(halfang)
d=2*sin(halfang)

OM=[(c*sin(lat)+0.5*d*cos(lat)*sin(tilt)),$
    -(sin(lon)*(c*cos(lat)-0.5*d*sin(lat)*sin(tilt))+0.5*d*cos(lon)*cos(tilt)),$
    (cos(lon)*(c*cos(lat)-0.5*d*sin(lat)*sin(tilt))+0.5*d*sin(-lon)*cos(tilt))]

return,OM
end
