;+
; PURPOSE: Filtering using fft
;
; CATEGORIES:
;  image processing, filtering
;
; INPUTS:
;  im : must be 512 x 512 image
;  ker : must be a 1024 x 1024 image
;
; OUTPUTS:
;  return : the filtered image
;
; $Id: fftconvol512.pro,v 1.1 2009/02/05 20:26:40 thernis Exp $
; $Log: fftconvol512.pro,v $
; Revision 1.1  2009/02/05 20:26:40  thernis
; First commit
;
;-
function fftconvol512,im,ker1024

im1024=fltarr(1024,1024)
im1024[256:767,256:767]=im

imconv=abs((fft(fft(im1024)*fft(ker1024),1))[256:767,256:767])

return,imconv/max(imconv)
end