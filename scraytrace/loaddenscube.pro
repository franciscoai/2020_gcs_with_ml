;+
; PURPOSE:
;  Load and format a density cube for the models 25 and 26
; CATEGORY:
;  raytracing, 3d, io
; INPUTS:
;  file : fits filename of the density cube
; OUTPUTS:
;  return : the modparam formated density cube
;
; ---- format of the model parameter array for a density cube
;//! -   0       : x size (sx)
;//! -   1       : y size (sy)
;//! -   2       : z size (sz)
;//! -   3       : xc Sun center in pix
;//! -   4       : yc Sun center in pix
;//! -   5       : zc Sun center in pix
;//!               -  note: (0,0,0) is the center of the first voxel
;//! -   6       : voxel size in rsun, same for the 3 directions of space
;//! -   7       : data cube in lexicographical order (x,y,z)
;
;
; CVS:
;  $Id: loaddenscube.pro,v 1.1 2006/09/08 16:03:20 nathan Exp $
;-

function loaddenscube,file

print,'Loading density cube : '+file

hdr=headfits(file)

x=long(sxpar(hdr,'naxis1'))
y=long(sxpar(hdr,'naxis2'))
z=long(sxpar(hdr,'naxis3'))
xc=sxpar(hdr,'xcenter')
yc=sxpar(hdr,'ycenter')
zc=sxpar(hdr,'zcenter')
voxsize=sxpar(hdr,'voxsize')

; -- use that returning method for memory saving
return,[x,y,z,xc,yc,zc,voxsize,reform(readfits(file,/silent),x*y*z)]
end
