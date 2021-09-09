;+
;  $Id: load_gibsondens.pro,v 1.1 2006/09/08 16:03:20 nathan Exp $
;
; PURPOSE: 
;  Load and build modparam input parameter for the density 35
;
; CATEGORY:
;  raytracing, io
;
; DESCRIPTION:
;  Load and build modparam input parameter for the density 35. This
;  uses the data cube produced by gibsoncmewrapper.pro and the model
;  of S.Gibson and B.Low.
;
; INPUTS:
;  filedens: filename of the density cube
;  filetemp: filename of the temperature cube
; 
; OUTPUTS:
;  modparam: modparam formated array to use with density model 35 in raytrace
;
;-
pro load_gibsondens,filedens,filetemp,modparam

rtinitenv
progpath=getenv('RT_PATH')+'/'

flagnofile=0B
if n_elements(filedens) eq 0 then flagnofile=1B else if strlen(filedens) eq 0 then flagnofile=1B

if flagnofile  then begin
    print,'Using default density file'
    filedens=progpath+'testDens00.xdr'
    filetemp=progpath+'testTemp00.xdr'
endif

openr,lun,filedens,/xdr,/get_lun
lstreadstruct,sd,lun
free_lun,lun

openr,lun,filetemp,/xdr,/get_lun
lstreadstruct,st,lun
free_lun,lun

rco=[sd.sr.vstart,(sd.sr.vend-sd.sr.vstart)/sd.sr.nbp,sd.sr.nbp]
phico=[sd.sphi.vstart*!dtor,!dtor*(sd.sphi.vend-sd.sphi.vstart)/sd.sphi.nbp,sd.sphi.nbp]
thetaco=[sd.stheta.vstart*!dtor,!dtor*(sd.stheta.vend-sd.stheta.vstart)/sd.stheta.nbp,sd.stheta.nbp]

; ---- quick check that the density cube match the size of the
;      temperature cube
szt=size(st.temp,/dim)
szd=size(sd.dens,/dim)

if szt[0] ne szd[0] or szt[1] ne szd[1] or szt[2] ne szd[2] then begin
    print,'The density and temperature cube sizes do not match: abort !'
    return
endif

modparam=[rco,phico,thetaco,reform(sd.dens,n_elements(sd.dens)),reform(st.temp,n_elements(st.temp))]

return
end


