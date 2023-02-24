;+
;  $Id: rtgetinstrwcsparam.pro,v 1.5 2010/08/26 13:23:59 mcnutt Exp $
;
; PURPOSE:
;  Extract pointing parameters from wcs header to initialize raytrace.
;
; INPUTS:
;  instr : detector
;  imsize : size of the image to be simulated
;  scchead : image fits header
;  pcin : PC matrix
;  flagfovpix: set if fovpix is defined by user
;
; OUTPUTS:
;  the different parameters needed for the raytracing.
;
; CVSLOG:
;  $Log: rtgetinstrwcsparam.pro,v $
;  Revision 1.5  2010/08/26 13:23:59  mcnutt
;  added KEYWORD_NULL_VALUE to call to fitshead2wcs
;
;  Revision 1.4  2008/06/06 17:38:25  thernis
;  Modif to make it work with SOHO data.
;
;  Revision 1.3  2007/07/24 14:52:39  thernis
;  Update comments
;
;-

pro rtgetinstrwcsparam,instr,imsize,scchead,fovpix,crpix,obsang,pc,imszratio,projtypepreset=projtypepreset,pv2_1=pv2_1,rollang=rollang,crval=crval,pcin=pcin,flagfovpix=flagfovpix

if n_elements(instr) eq 0 then return
instr=strlowcase(instr)

; ---- Get image resolution
wcs=fitshead2wcs(scchead,KEYWORD_NULL_VALUE=0)
imszratio=(float(wcs.naxis[0])/float(imsize[0]))
if not keyword_set(flagfovpix) then fovpix=float((wcschangeunits(wcs.cunit[0],'rad',abs(wcs.cdelt[0]))*imszratio)[0])
    
; ---- get the angular coordinates of the center of the image: optical axis
crval=wcschangeunits(wcs.cunit,'rad',wcs.crval)
crpix=float(piximchangereso(wcs.crpix-[1,1],-alog(imszratio)/alog(2)))




; ---- compute the spacecraft attitude

rmat=rotmat(crval[1],2)##rotmat(-crval[0],1)##rotmat((n_elements(rollang) eq 0 ? 0 : -rollang),3)
obsang=float(rtrotmat2rxryrz(rmat))
if n_elements(pcin) ne 4 then pc=float(wcs.pc) else pc=float(pcin)

; ---- get projection type
projtypepreset=wcs.projection
if tag_exist(wcs,'proj_names') then begin
	m=where(wcs.proj_names eq 'PV2_1',cnt)
	if cnt gt 0 then pv2_1=wcs.proj_values[m[0]] else pv2_1=0.
endif else pv2_1=0.


return
end
