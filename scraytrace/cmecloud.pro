;+
; $Id: cmecloud.pro,v 1.6 2008/06/26 15:26:29 thernis Exp $
; PURPOSE:
;  build a cloud of the model 54 cme: the points of the clouds are
;  located only on the surface of the max of NE-> on the shell
;
; INPUTS:
;  distjuncisleadingedge : set if distjuncin input is meant to be the height of the leading edge, not the height of the feet
;
; EXAMPLE:
;  oc=cmecloud(30.*!dtor,2.,5,20,0.4,50)
;  plot_3dbox,oc[0,*],oc[1,*],oc[2,*],psym=1
;
; CVSLOG:
;  $Log: cmecloud.pro,v $
;  Revision 1.6  2008/06/26 15:26:29  thernis
;  Use long for the loop to avoid problem when user wants a lot of points
;
;  Revision 1.5  2008/05/16 20:10:43  thernis
;  Fix the height parameter slider to the true height of the croissant model
;
;  Revision 1.4  2008/03/27 18:44:01  thernis
;  Fix artefacts in the wireframe model.
;
;  Revision 1.3  2007/07/25 19:41:46  thernis
;  Implement visu of neutral line orientation in the EUVI views
;
;  Revision 1.2  2007/07/25 14:46:27  thernis
;  Clean up the comments
;
;-


; ----- returns the position of a point of the neutral line in the
;       scraytrace abs coordinate system
function neutlinecarposabs,lon,lat,halfang,tilt

c=cos(halfang)
d=2*sin(halfang)

OM=[(c*sin(lat)+0.5*d*cos(lat)*sin(tilt)),$
    -(sin(lon)*(c*cos(lat)-0.5*d*sin(lat)*sin(tilt))+0.5*d*cos(lon)*cos(tilt)),$
    (cos(lon)*(c*cos(lat)-0.5*d*sin(lat)*sin(tilt))+0.5*d*sin(-lon)*cos(tilt))]

return,OM
end


; ------ main function
function cmecloud,hang,distjuncin,nbvertsl,nbvertcirc,k,nbvertcircshell,distjuncisleadingedge=distjuncisleadingedge

if keyword_set(distjuncisleadingedge) then distjunc=distjuncin*(1.-k)*cos(hang)/(1.+sin(hang)) else distjunc=distjuncin

; ---- compute the skeleton axis
p=shellskeleton(hang,distjunc,nbvertsl,nbvertcirc,k,rrr,ca)

nbp=(size(p,/dim))[1]

theta=lgen(nbvertcircshell,0.,360.-360./nbvertcircshell)*!dtor
sintheta=sin(theta)

OC=fltarr(3,nbvertcircshell*nbp)


for i=0L,nbp-1 do begin
	OCtmp=rrr[i]*transpose([[cos(theta)],[sintheta*cos(ca[i])],[sintheta*sin(ca[i])]])
    
	for j=0,nbvertcircshell-1 do OCtmp[*,j]=OCtmp[*,j]+p[*,i]

	OC[*,(i*nbvertcircshell):((i+1)*nbvertcircshell-1)]=OCtmp

endfor
;stop

;plot,ca-!pi/2,rrr
; ---- for each point of the axis, compute the cross section circle
;      don't use first and last point since we need the derivative

; OC=findgen(3,nbvertcircshell*(nbp-2))
; 
; for i=1,nbp-2 do begin
    
    ; -- Compute the vector normal to the plane of the circle
    ;    That vector is the axis orientation at the current point
    ;    note that the axis is in the y z plane
    ;    v is (0, -sin(xi), cos(xi))
;     v=p[*,i+1]-p[*,i-1]
;     v=v/norm(v)

    ; -- compute radius
;     r=norm(reform(p[*,i]))*k

    ; -- circle
    ;    note that OC is OOPrime+r*(cos(theta),sin(theta)cos(xi),sin(theta)sin(xi))
;     sintheta=sin(theta)
;     OCtmp=r*transpose([[cos(theta)],[sintheta*v[2]],[-sintheta*v[1]]])
;     
;     for j=0,nbvertcircshell-1 do OCtmp[*,j]=OCtmp[*,j]+p[*,i]
; 
;     OC[*,((i-1)*nbvertcircshell):(i*nbvertcircshell-1)]=OCtmp
;    
; endfor

return,OC
end
