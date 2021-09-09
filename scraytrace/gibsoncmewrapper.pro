;+
; PURPOSE:
;  Generates a density and temperature cube with the S.Gibson and B.Low MHD CME model
;
; CATEGORY:
;  raytracing, simulation, modeling
;
; DESCRIPTION:
;  Generates a density and temperature cube with the S.Gibson and
;  B.Low MHD CME model
;
; RESTRICTIONS:
;  Requires the MHD CME model program, not part of the raytracing
;  software package
;
; INPUTS:
; time : time array
; sr : radial resolution {vstart: starting value,vend: end value, nbp:
;                         number of points}
; stheta : theta resolution (same struct as for sr)
; sphi : phi resolution (same struct as for sr)
; fname : filename to save the density and temperature cubes
; 
; OUTPUT:
;  file written on the disk
;
; $Id: gibsoncmewrapper.pro,v 1.1 2006/09/08 16:03:20 nathan Exp $
;
;-

pro gibsoncmewrapper,time,sr,stheta,sphi,accel,fname

; ---- how many dates requested
nbt=n_elements(time)

sr={vstart:1.,vend:6.,nbp:150}
stheta={vstart:0.,vend:180.,nbp:181}
sphi={vstart:0.,vend:359.,nbp:180}

; ---- init density and temperature cubes
dens=fltarr(sr.nbp,stheta.nbp,sphi.nbp)
temp=fltarr(sr.nbp,stheta.nbp,sphi.nbp)

; ---- init slices coordinates
r=replicate(1.,stheta.nbp)##(findgen(sr.nbp)/float(sr.nbp-1)*(sr.vend-sr.vstart)+sr.vstart)

theta=(findgen(stheta.nbp)/float(stheta.nbp-1)*(stheta.vend-stheta.vstart)+stheta.vstart)##replicate(1.,sr.nbp)

phi=fltarr(stheta.nbp,sr.nbp)
philine=findgen(sphi.nbp)/float(sphi.nbp-1)*(sphi.vend-sphi.vstart)+sphi.vstart


; ---- loop for each time
for t=0,nbt-1 do begin

    ; ---- loop over each longitude slices
    for i=0,sphi.nbp-1 do begin
        print,'time : '+strtrim(time[t],2)+' : '+strtrim(t,2)+' / '+strtrim(nbt-1,2)+', slice : '+strtrim(i,2)+' / '+strtrim(sphi.nbp-1,2)
        phi[*]=philine[i]

        ; -- call S.Gibson prog
        drivecme,time[t],r,theta,phi,accel,densslice,tempslice
        
        dens[*,*,i]=densslice
        temp[*,*,i]=tempslice

    endfor

    ; ---- save the result on disk
    densfname=fname+'Dens'+string(t,form='(I2.2)')+'.xdr'
    tempfname=fname+'Temp'+string(t,form='(I2.2)')+'.xdr'

    openw,lun,densfname,/xdr,/get_lun
    lstsavestruct,{dens:dens,sr:sr,stheta:stheta,sphi:sphi,time:time[t],fname:fname},lun
    free_lun,lun

    openw,lun,tempfname,/xdr,/get_lun
    lstsavestruct,{temp:temp,sr:sr,stheta:stheta,sphi:sphi,time:time[t],fname:fname},lun
    free_lun,lun

endfor


return
end
