;+
;  $Id: rtcalccntr.pro,v 1.1 2006/09/08 16:03:21 nathan Exp $
;
; PURPOSE:
;  Calculate the center of the sun in the image according to FOV and obspos
; CATEGORY:
;  simulation, raytracing
; INPUTS:
;  fovpix : the field of view of one pixel in rad
;  obsang : [Rx,Ry,Rz] orientation angle of the observer in rad
;  insize : size of the image in pix
; OUTPUTS:
;  cntr : center of the Sun in the image
;-


function rtcalccntr,fovpix,obsang,imsize

cntr=fltarr(2)
cntr[0]=obsang[0]/fovpix+float(imsize[0]-1)/2
cntr[1]=-obsang[1]/fovpix+float(imsize[1]-1)/2


return,cntr
end
