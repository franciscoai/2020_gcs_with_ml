;+
; PURPOSE: 
;  Extract from a fits header the size of the Sun in pixels 
; 
; CATEGORIES:
;  fits, data handling
;
;  $Log: getpixperrsun.pro,v $
;  Revision 1.1  2009/02/18 22:50:05  thernis
;  First commit.
;
;-


function getpixperrsun,hdr

radperrsun=atan(1,hdr.dsun_obs/onersun()/1e3)

wcs=fitshead2wcs(hdr)
radperpix=(wcschangeunits(wcs.cunit,['rad','rad'],wcs.cdelt))[0]

rsunpix=radperrsun/radperpix

return,rsunpix
end