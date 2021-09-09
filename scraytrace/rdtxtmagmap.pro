;+
; $Id: rdtxtmagmap.pro,v 1.3 2014/10/07 18:37:43 thernis Exp $
;
; PURPOSE:
;  Read or download on the net a WSO formated magnetogram Carrington map
;
; CATEGORY:
;  Raytracing, data handling, io
;
; INPUTS:
;  file: filename of the map
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
;  nsheetmap: raw neutral sheet Carrington map
;  loninterp: 
;  neutinterp:
;  map:
;  mapr:
;-

pro rdtxtmagmap,nsheetmap,loninterp,neutinterp,file=file,crot=crot,crcalcmeth=crcalcmeth,latsin=latsin,rebindex=rebindex,nbcr=nbcr,crshift=crshift,rebmap=rebmap,map=map,mapr=mapr

; ---- latitude in linear scale or sinus linear scale ?
latsin=keyword_set(latsin)

; nbcr only valid with crot
if n_elements(nbcr) eq 0 then nbcr=1

if n_elements(rebindex) eq 0 then rebindex=1L

if n_elements(file) eq 0 then begin
    file='/home/thernis/work/stream012204/magneto/CR2012_250.txt'
    latsin=0B
endif
if n_elements(crcalcmeth) eq 0 then crcalcmeth='R250.' else $
  case crcalcmeth of
    1 : crcalcmeth='S.'
    2 : crcalcmeth='R250.'
    3 : crcalcmeth='R325.'
    else : crcalcmeth='R250.'
endcase

if n_elements(crot) eq 0 then begin
    readtxtfile,file,txt,nbline
    print,'Loading '+file
endif else begin
    txtin=strarr(294,nbcr)

    for i=0,nbcr-1 do begin

        print,'Fetching data at WSO web site...'
        ;sock_list,'quake.stanford.edu/~wso/Source.'+crcalcmeth+'/CR'+string(crot+i,form='(I4)'),txt
        ;sock_list,'wso.stanford.edu/Source.'+crcalcmeth+'/CR'+string(crot+i,form='(I4)'),txt

        ; ---- 20141007: new URL format: http://wso.stanford.edu/synoptic/WSO-S.2037.txt
        sock_list, 'http://wso.stanford.edu/synoptic/WSO-' + crcalcmeth + string(crot+i,form='(I4)') + '.txt', txt


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


endelse


if latsin then nbrec=30 else nbrec=29

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

; -- cut out the Carr map
;if n_elements(crshift) ne 0 then begin
;    d1=crshift/5
;    d2=d1+72
;    map=map[d1:d2,*]
;endif
szm=size(map,/dim)

if n_elements(rebmap) ne 1 then rebmap=1
mapr=congrid(map,szm[0]*rebmap,szm[1]*rebmap,/cubic,/center);,/center)
;mapr=rebin(map,szm[0]*rebmap,szm[1]*rebmap)

szmr=size(mapr,/dim)
;stop
; ---- find position of the neutral line
;lon=findgen(73)*5
lon=findgen(szmr[0])/(szmr[0]-1)*360*nbcr
;if n_elements(crshift) eq 0 then lon=findgen(szmr[0])/(szmr[0]-1)*360*nbcr else lon=findgen(szmr[0])/(szmr[0]-1)*360
if latsin then begin

    ; -- take into accound resampling effect

    ; analytic correction: does not work !
    latwidth=2.*14.5/15
    latfullstep=(2.*14.5/15)/(szmr[1]-1)
    lathpixshift=( (rebmap gt 1) ? (latfullstep/rebmap) : 0.)/2.
    lat=asin(findgen(szmr[1])/(szmr[1]-1)*(latwidth+2*lathpixshift)-(14.5/15+lathpixshift))*!radeg 

    ; bourrin correction !
    ;lathpixshift=lathpixshift*20
    ;lat=asin(findgen(szmr[1])/(szmr[1]-1)*(latwidth+2*lathpixshift)-(14.5/15+lathpixshift))*!radeg 

    ;latold=asin(findgen(szmr[1])/(szmr[1]-1)*(2.*14.5/15)-(14.5/15))*!radeg

endif else begin
    lat=findgen(szmr[1])/(szmr[1]-1)*140-70
endelse

neutline=fltarr(szmr[0])

for i=0,szmr[0]-1 do begin
    p=reform(mapr[i,*])

    mneg=where(p le 0,cntneg)
    if cntneg eq 0 then begin
        neutline[i]=100
        break
    endif
    maxn=max(p[mneg])
    mfn=(where(p eq maxn))[0]

    mpos=where(p gt 0,cntpos)
    if cntpos eq 0 then begin
        neutline[i]=-100
        break
    endif
    minp=min(p[mpos])
    mfp=(where(p eq minp))[0]

    ; --- zero crossing nearest points should have contiguous subscripts
    if mfn ne mfp+1 and mfn ne mfp-1 then begin
        message,'I''ve got a problem'
    endif
    
    x0=(lat[mfn]*minp-lat[mfp]*maxn)/(minp-maxn)
    ;plot,lat,p,psym=-2

;stop
    neutline[i]=x0

endfor


; ---- format a map so that it works with the raytracing prog

; -- old code
;nsheetmap=fltarr(360,181)
;loninterp=findgen(360*16)/16
;neutinterp=spline(lon,neutline,loninterp)
;m=long(loninterp)+long(neutinterp+90)*360L
;nsheetmap[m]=1


; -- init map
slon=360L*rebindex*nbcr
;if n_elements(crshift) eq 0 then slon=360L*rebindex*nbcr else slon=360L*rebindex
slat=180L*rebindex+1L
nsheetmap=fltarr(slon,slat)

; -- over sampling
loninterp=findgen(slon)/float(slon-1)*(360*nbcr-1)
;if n_elements(crshift) eq 0 then loninterp=findgen(slon)/float(slon-1)*(360*nbcr-1) else loninterp=findgen(slon)/float(slon-1)*(360-1)
neutinterp=spline(lon,neutline,loninterp)

xx=round(loninterp[0]*float(rebindex))
yy=round((neutinterp[0]+90)*float(rebindex))

nsheetmap[xx,yy]=1

for i=1,slon-1 do begin
    xnn=round(loninterp[i]*float(rebindex))
    ynn=round((neutinterp[i]+90)*float(rebindex))

    if abs(yy-ynn) gt 1 then begin
        if yy lt ynn then for j=yy,ynn do begin
            b=1./float(ynn-yy)
            nsheetmap[xnn+(j-yy)*b,j]=1
        endfor else for j=ynn,yy do begin
            b=1./float(ynn-yy)
            nsheetmap[xnn+(j-yy)*b,j]=1
        endfor
    endif else nsheetmap[xnn,ynn]=1

    xx=xnn
    yy=ynn

endfor

; -- cut out if requested
if n_elements(crshift) ne 0 then begin
    d1=crshift*rebindex
    d2=(crshift+360)*rebindex-1
    nsheetmap=nsheetmap[d1:d2,*]
    loninterp=loninterp[d1:d2]
    neutinterp=neutinterp[d1:d2]
endif

return
end


; $Log: rdtxtmagmap.pro,v $
; Revision 1.3  2014/10/07 18:37:43  thernis
; Update URL location of synoptic carrington maps in Stanford web site.
;