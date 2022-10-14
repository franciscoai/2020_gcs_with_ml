;+
; $Id: rtcarr2abs.pro,v 1.2 2007/07/19 19:42:40 thernis Exp $
;
; PURPOSE: 
;  Convert carrington position into raytrace absolute coord system
;
; CVSLOG:
;  $Log: rtcarr2abs.pro,v $
;  Revision 1.2  2007/07/19 19:42:40  thernis
;  Just put the cvslog in the prog header
;
;-

function rtcarr2abs,carrlonlatheight

x=carrlonlatheight[2]*sin(carrlonlatheight[1])
y=-carrlonlatheight[2]*sin(carrlonlatheight[0])*cos(carrlonlatheight[1])
z=carrlonlatheight[2]*cos(carrlonlatheight[0])*cos(carrlonlatheight[1])

return,[x,y,z]
end
