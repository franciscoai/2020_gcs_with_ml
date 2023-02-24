;+
;  $Id: rtinstrpresets.pro,v 1.7 2010/09/08 15:47:12 thernis Exp $
;
; PURPOSE:
;  output the FOV presets for the requested instrument
; CATEGORY:
;  simulation, raytracing, calibration
; INPUTS:
;  instr : instrument name
;  imsize : image size in pix
;  projtypepassed : set to avoid overwriting the projtype
;
; OUTPUTS:
;  fovpix : field of view of a pixel in rad
;  obsang : observer angle
;  imszratio : rebin factor
;  secchiab : 'A' for spacecraft Ahead and 'B' for Behind
;  projtypecode : TAN or SIN depending on the instrument
;  pv2_1 : return the mu parameter for the AZP projection
;  crval : position pointed by the instrument optical axis in helioprojective cartesian coordinates
;-

pro rtinstrpresets,instr,imsize,fovpix,obsang,imszratio,crval=crval,secchiab=secchiab,projtypepreset=projtypepreset,pv2_1=pv2_1,scchead=scchead,rollang=rollang,crpixin=crpixin

; ---- overread instrument if instr KW used
if n_elements(instr) eq 0 then begin
	crpixin=float(imsize)/2.-0.5
	return
endif

instr=strlowcase(instr)
flagfound=0b
imszratio=1.


; ---- use secchi header if present
if n_elements(scchead) ne 0 then begin
    rtgetinstrwcsparam,instr,imsize,scchead,fovpix,crpix,obsang,pc,imszratio,projtypepreset=projtypepreset,pv2_1=pv2_1,crval=crval
    
    ; -- compute spacecraft attitude without pc matrix
    rmat=rotmat(crval[1],2)##rotmat(-crval[0],1)##rotmat(-rollang,3)
    obsang=float(rtrotmat2rxryrz(rmat))

    return
endif



; ---- instrument FOV presets
case instr of
    'c2' : begin
        flagfound=1b
        scale=rt_t_param('C2','SCALE')
        ; -- the image is assumed to be square
        imszratio=1024./float(imsize[0])
        fovpix=scale*!dtor*imszratio

        ; -- shift of the sun center
        scntr=rt_t_param('C2','CENTER')
        shft=(scntr-[511.5,511.5])*scale*!dtor
        obsang=[shft[0],-shft[1],0]
        
	crpixin=float(piximchangereso(scntr,-alog(imszratio)/alog(2)))
	crval=[0.,0.]

        ; -- projection type
        projtypepreset='TAN'
        pv2_1=0
    end
    'c3' : begin
        flagfound=1b
        scale=rt_t_param('C3','SCALE')
        ; -- the image is assumed to be square
        imszratio=1024./float(imsize[0])
        fovpix=scale*!dtor*imszratio

        ; -- shift of the sun center
        scntr=rt_t_param('C3','CENTER')
        shft=(scntr-[511.5,511.5])*scale*!dtor
        obsang=[shft[0],-shft[1],0]

	crpixin=float(piximchangereso(scntr,-alog(imszratio)/alog(2)))
	crval=[0.,0.]

        ; -- projection type
        projtypepreset='TAN'
        pv2_1=0
    end
    'c1' : begin
        flagfound=1b
        scale=rt_t_param('C1','SCALE')
        ; -- the image is assumed to be square
        imszratio=1024./float(imsize[0])
        fovpix=scale*!dtor*imszratio
        
        ; -- shift of the sun center
        scntr=rt_t_param('C1','CENTER')
        shft=(scntr-[511.5,511.5])*scale*!dtor
        obsang=[shft[0],-shft[1],0]

	crpixin=float(piximchangereso(scntr,-alog(imszratio)/alog(2)))
	crval=[0.,0.]

        ; -- projection type
        projtypepreset='TAN'
        pv2_1=0
    end
    'cor1' : begin
        flagfound=1b
        ; -- assume 4 rsun radius FOV
        ;    the image is assumed to be square
        imszratio=2048./float(imsize[0])
        fovpix=3.769*!dtor/3600.*imszratio
        ; -- the Sun center is assumed to be at the center of the image
        obsang=[0.,0,0]

	crpixin=float(piximchangereso([1023.5,1023.5],-alog(imszratio)/alog(2)))
	crval=[0.,0.]

        ; -- projection type
        projtypepreset='TAN'
        pv2_1=0
    end
    'cor2' : begin
        flagfound=1b
        ; -- assume 15 rsun radius FOV
        ;    the image is assumed to be square
        imszratio=2048./float(imsize[0])
        fovpix=14.5*!dtor/3600.*imszratio
        ;fovpix=float(atan(15.,oneau())*2./float(imsize[0]))
        ; -- the Sun center is assumed to be at the center of the image
        obsang=[0.,0,0]
	crval=[0.,0.]
 
	crpixin=float(piximchangereso([1023.5,1023.5],-alog(imszratio)/alog(2)))

       ; -- projection type
        projtypepreset='TAN'
        pv2_1=0
    end
    'hi1' : begin
        flagfound=1b
        ; -- assume 37.5 rsun radius FOV
        ;    the image is assumed to be square
        fovpix=float(atan(37.5,oneau())*2./float(imsize[0]))

        ; -- center : see Secchi Phase A concept study report, p3-38
        ;obspos=[0,49.8,-204.01] ; see 8 nov 2004
        if secchiab eq 'A' then obsang=[13.28*!dtor,0,0] else $ ; see 8 nov 2004
          obsang=[-13.28*!dtor,0,0]

	crpixin=float(imsize)/2.-0.5
	crval=[-obsang[0]*!radeg,0.]

        ; -- projection type
        projtypepreset='AZP'
        pv2_1=0.171830          ; mu parameter of the AZP projection
    end
    'hi2' : begin
        flagfound=1b
        ; -- assume 129.6 rsun radius FOV
        ;    the image is assumed to be square
        fovpix=float(atan(129.6,oneau())*2./float(imsize[0]))

        ; -- center : see Secchi Phase A concept study report, p3-38
        ;obspos=[0,200.,-64.0312] ; see 8 nov 2004
        if secchiab eq 'A' then obsang=[53.36*!dtor,0,0] else $ ; see 8 nov 2004
          obsang=[-53.36*!dtor,0,0]

	crpixin=float(imsize)/2.-0.5
	crval=[-obsang[0]*!radeg,0.]

        ; -- projection type
        projtypepreset='AZP'
        pv2_1=0.662969          ; mu parameter of the AZP projection
    end
    else : message,'Instrument '+instr+' not recognized',/info

endcase





return
end
