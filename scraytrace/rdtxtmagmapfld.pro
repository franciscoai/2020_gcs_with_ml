;+
; $Id: rdtxtmagmapfld.pro,v 1.1 2006/09/08 16:03:21 nathan Exp $
;
; PURPOSE:
;  Download source surface field map from VSO and format the map for the density 37 of the raytracewl
;
; CATEGORY:
;  Raytracing, data handling, io
;
; INPUTS:
;  crot: carrington rotation to download
;  crcalcmeth: type of map (see WSO web site)
;  latsin: use projected sin latitude map instead of unprojected
;  rebindex: rebin index to enlarge the map
;  nbcr: number of carrington rotation to stitch together after the
;        crot which is the first one.
;  crshift:
;  rebmap:
;
; OUTPUTS:
;  mapinterp:
;  modparam:
;
;-

pro rdtxtmagmapfld,crot,mapinterp,modparam,crcalcmeth=crcalcmeth,latsin=latsin,rebindex=rebindex,nbcr=nbcr,crshift=crshift,rebmap=rebmap

; ---- latitude in linear scale or sinus linear scale ?
latsin=keyword_set(latsin)

; nbcr only valid with crot
if n_elements(nbcr) eq 0 then nbcr=1

if n_elements(rebindex) eq 0 then rebindex=1L

if n_elements(crcalcmeth) eq 0 then crcalcmeth='250.rad' else $
  case crcalcmeth of
    1 : crcalcmeth='250.los'
    2 : crcalcmeth='250.rad'
    3 : crcalcmeth='325.rad'
    else : crcalcmeth='250.rad'
endcase

txtin=strarr(294,nbcr)

for i=0,nbcr-1 do begin

    print,'Fetching data at WSO web site...'
    sock_list,'quake.stanford.edu/~wso/Source.'+crcalcmeth+'/CR'+string(crot+i,form='(I4)'),txt
    nbline=n_elements(txt)
    if nbline ne 295 then begin
        message,'Transmission error or the downloaded file doesn''t have the proper size: abort !',/info
        return
    endif
    print,'Download done.'
                                ; -- remove first blanc line
    txt=txt[1:*]
    nbline=nbline-1
    latsin=1B
    
    txtin[*,i]=txt
endfor

nbrec=30 

map=fltarr(72*nbcr+1,nbrec)

k=0
for j=0,nbcr-1 do begin
for i=2,nbline-5,4 do begin
    map[k,0:5]=float((str_sep(strcompress(txtin[i,j]),' '))[1:6])
    map[k,6:13]=float((str_sep(strcompress(txtin[i+1,j]),' '))[1:8])
    map[k,14:21]=float((str_sep(strcompress(txtin[i+2,j]),' '))[1:8])
    if latsin then map[k,22:29]=float((str_sep(strcompress(txtin[i+3,j]),' '))[1:8]) else map[k,22:28]=float((str_sep(strcompress(txtin[i+3,j]),' '))[1:7])
    k=k+1
endfor
endfor
j=j-1
map[k,0:5]=float((str_sep(strcompress(txtin[i,j]),' '))[1:6])
map[k,6:13]=float((str_sep(strcompress(txtin[i+1,j]),' '))[1:8])
map[k,14:21]=float((str_sep(strcompress(txtin[i+2,j]),' '))[1:8])
if latsin then map[k,22:29]=float((str_sep(strcompress(txtin[i+3,j]),' '))[1:8]) else map[k,22:28]=float((str_sep(strcompress(txtin[i+3,j]),' '))[1:7])


map=reverse(map,1)
map=reverse(map,2)

szm=size(map,/dim)

; ---- over sampling if requested
if n_elements(rebmap) ne 1 then rebmap=1
mapr=congrid(map,szm[0]*rebmap,szm[1]*rebmap,/cubic,/center);,/center)

szmr=size(mapr,/dim)

; ---- find position of the neutral line
lon=findgen(szmr[0])/(szmr[0]-1)*360*nbcr

; -- take into accound resampling effect
; analytic correction: does not work !
latwidth=2.*14.5/15
latfullstep=(2.*14.5/15)/(szmr[1]-1)
lathpixshift=( (rebmap gt 1) ? (latfullstep/rebmap) : 0.)/2.
lat=asin(findgen(szmr[1])/(szmr[1]-1)*(latwidth+2*lathpixshift)-(14.5/15+lathpixshift))*!radeg 

; ---- interpolation 
lon=findgen(szmr[0])/(szmr[0]-1)*360*nbcr



; -- init map
slon=360L*rebindex*nbcr
;if n_elements(crshift) eq 0 then slon=360L*rebindex*nbcr else slon=360L*rebindex
slat=180L*rebindex+1L
nsheetmap=fltarr(slon,slat)


lonm=replicate(1.,slat)##(findgen(slon)/rebindex/5)

; see 4 nov 2004
aaa=-14.5/15
bbb=(14.5/15)*2./29

latm=(sin(!dtor*(findgen(slat)/(slat-1)*180.-90))/bbb-aaa/bbb)##replicate(1.,slon)


mapinterp=bilinear(map,lonm,latm)

modparam=[360.,181,reform(mapinterp,n_elements(mapinterp))]


return
end
