PRO bowshock_cloud,d,s,h, xx,yy,zz, nr = nr, nth = nth
;
;
	IF ~keyword_set(nr) THEN nr = 20 ;# of radial elements
	IF ~keyword_set(nth) THEN nth = 20 ; # of azimuthal elements
;
	r = MAKEX(1,nr,1.)#replicate(1,nth)
	th = MAKEN(0,360.,nth)#replicate(1,nr)
	th = TRANSPOSE(th)
	x = r*cos(th*!dtor)
	y = r*sin(th*!dtor)
;
	yy = REFORM(transpose(y),nr*nth)                                   
	xx = REFORM(transpose(x),nr*nth)
;
	zz = h-d/s*(sqrt(xx^2+yy^2)/d)^s
;	plot_3dbox,xx,yy,zz,charsize=2,psym=3
;
;modparam=[d,s,h,thickness,density]

; 

RETURN
END

