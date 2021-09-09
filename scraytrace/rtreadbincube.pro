;+
; PURPOSE:
;  Read a cube created by buildcloud
;
;-

pro rtreadbincube,fn,c,szs,orig,nbp,cntr,retcube=retcube

openr,lun,fn,/get_lun

outtype=0L ; density cube type
readu,lun,outtype

; -- process only if it's a density cube
if outtype ne 2 then begin
    print,'The output type is : ',outtype
    print,'The file is not a density cube. Cannot process !'
    return
endif

nbp=0UL ; number of pix
readu,lun,nbp
szs=0. ; size in Rsun
readu,lun,szs
orig=fltarr(3) ; position of the first voxel in Rsun
readu,lun,orig

; -- create and read the cube
c=fltarr(nbp,nbp,nbp)
readu,lun,c

free_lun,lun

; ---- format the cube for raytracewl or rtdenscube
; -- axis size

cntr=-orig/szs*float(nbp-1)

if ~keyword_set(retcube) then begin
	c=[nbp,nbp,nbp,cntr,szs/(nbp-1),reform(c,nbp*nbp*nbp)]
endif


return
end
