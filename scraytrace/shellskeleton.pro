;+
; $Id: shellskeleton.pro,v 1.3 2011/05/02 18:47:26 thernis Exp $
;
; PURPOSE:
;  Compute the axis of the GCS CME model (model 54)
;
; CVSLOG:
;  $Log: shellskeleton.pro,v $
;  Revision 1.3  2011/05/02 18:47:26  thernis
;  Fix leg to front part junction
;
;  Revision 1.2  2008-03-27 18:44:01  thernis
;  Fix artefacts in the wireframe model.
;
;  Revision 1.1  2007/07/25 14:46:51  thernis
;  first commit
;
;-

function shellskeleton,alpha,distjunc,nbvertsl,nbvertcirc,k,r,ca

; ---- compute entire loop lenght
looplenght=distjunc*(1.+(alpha+!pi/2)*tan(alpha))

; ---- compute circular part half lenght
hcirclenght=distjunc*tan(alpha)*(2.*alpha+!pi)/2

; ---- calculate the points of the straight line part
pRstart=[sin(alpha),cos(alpha)]   ; start on the limb
pLstart=[-sin(alpha),cos(alpha)]
pslR=fltarr(3,nbvertsl)
pslL=fltarr(3,nbvertsl)
rsl=fltarr(nbvertsl) ; shell radius for the feet
casl=fltarr(nbvertsl) ; tilt angle of the plane of the circle
step=(distjunc-1)/(nbvertsl-1) ; start on the limb

gamma=asin(k)

for i=0,nbvertsl-1  do begin

	xxx=i*step*sin(alpha)+pRstart[0]
	yyy=i*step*cos(alpha)+pRstart[1]
    pslR[1,i]=xxx
    pslR[2,i]=yyy

    pslL[1,i]=-pslR[1,i]
    pslL[2,i]=pslR[2,i]

; 	rsl[i]=k*sqrt(xxx*xxx+yyy*yyy)
	rsl[i]=tan(gamma)*sqrt(xxx*xxx+yyy*yyy)
	casl[i]=-alpha
endfor

; ---- calculate the points of the circular part
;rc=fltarr(nbvertcirc) ; radius of the shell for the circular part
;cac=fltarr(nbvertcirc) ; tilt angle of the shell circle
pcR=fltarr(3,nbvertcirc)
pcL=fltarr(3,nbvertcirc)
step=(alpha+!pi/2)/(nbvertcirc-1)

beta=lgen(nbvertcirc,-alpha,!pi/2)
hf=distjunc
h=hf/cos(alpha)
rho=hf*tan(alpha)

X0=(rho+h*k^2*sin(beta))/(1-k^2)
rc=sqrt((h^2*k^2-rho^2)/(1-k^2)+X0^2)
cac=beta

pcR[1,*]=X0*cos(beta) ;    1/(1-k^2)*(rho*cos(beta)+h*k^2*sin(beta)*cos(beta))
pcR[2,*]=h+X0*sin(beta) ; 1/(1-k^2)*(rho*sin(beta)+h*k^2*sin(beta)^2)+h

pcL[1,*]=-pcR[1,*]
pcL[2,*]=pcR[2,*]

r=[rsl,rc[1:*],(reverse(rc))[1:*],(reverse(rsl))[1:*]]
ca=[casl,cac[1:*],!pi-(reverse(cac))[1:*],!pi-(reverse(casl))[1:*]]
p=[[pslR],[pcR[*,1:*]],[(reverse(pcL,2))[*,1:*]],[(reverse(pslL,2))[*,1:*]]]


return,p
end
