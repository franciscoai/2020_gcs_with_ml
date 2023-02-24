pro spheroidshock_cloud,e,kappa,h, xx,yy,zz, nr = nr, nth = nth

;
;Input: e - eccentricity if e>1 then prolate if e<1 oblate
;       kappa - self-similar constant
;       h - height of spheroid from solar center in unit of solar radians
;
;       nr - number of radial points (default 20)
;       nt - number of azimuthal points (defalut 20)
;
;Output: xx,yy,zz
;
;Author: Oscar Olmedo Jul 25, 2013

	IF ~keyword_set(nr) THEN nr = 20 ;# of radial elements
	IF ~keyword_set(nth) THEN nth = 20 ; # of azimuthal elements
       
        s=kappa*(h-1.0)
        case 1 of
           (e lt 0.):d=s*sqrt(1-e^2.)
           (e gt 0.):d=s/sqrt(1-e^2.)
           (e eq 0.):d=s
           else: stop,'e ERROR in spheroidshock_cloud.pro'
        endcase
 
        th = MAKEN(0,360.,nth+1)
        rth = MAKEN(0,180.,nr)
        sth=sin(rth*!dtor)
        x = (s)*sth#cos(th*!dtor)
        y = (s)*sth#sin(th*!dtor)
        z=(d)*(cos(rth*!dtor)-1.)#(fltarr(nth+1)+1)+h
;
	xx = REFORM(transpose(x),nr*(nth+1))
	yy = REFORM(transpose(y),nr*(nth+1))
	zz = REFORM(transpose(z),nr*(nth+1))
;
	;zz = h-d/s*(sqrt(xx^2+yy^2)/d)^s
	;plot_3dbox,xx,yy,zz,charsize=2,psym=3
;
;modparam=[d,s,h,thickness,density]

; 

RETURN
END

