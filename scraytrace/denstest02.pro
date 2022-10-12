;+
; $Id: denstest02.pro,v 1.1 2006/09/08 16:03:19 nathan Exp $
;
; PURPOSE: generate a dummy density cube for testing
; CATEGORY: raytracing, 3d
; INPUTS:
;  spix: size of the cube in pix
;  modul: modulation factor of the randomization of the voxel values
;  max: max value for a voxel
;  csize: cube side size in Rsun 
; OUTPUTS:
;  sdc : returned structure
;     dc: the created density cube
;     cntr: center of the density cube in pix units
;     ssize: voxel side size in Rsun
;-

function denstest02,spix,modul,max,csize

if spix eq 0 then spix=32
dc=(randomu(seed,spix,spix,spix)-0.5)*modul+max


cntr=([spix,spix,spix]-1.)/2.
ssize=csize/float(spix) ; voxel side size [Rsun]

sdc={dc:dc,cntr:cntr,ssize:ssize}


return,sdc
end
