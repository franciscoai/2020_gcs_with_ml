;+
;*************************************************************************
; Francisco A. Iglesias - UTN-FRM/GEAA - franciscoaiglesias@frm.utn.edu.ar
;
; Reads all the .sav files containing the parameters of the fitted CGS model
; ,following Thernisien et. al. 2009, and plots them vs time.
; 
; History:
;	20180411 - First version
;*************************************************************************
;-
pro  tevo_FMwLASCO20110317
@gcs_config

DIR=GCSD_PATH+'/GCS_20110317'

f=file_search(DIR+'/','*.sav')
nf=n_elements(f)
; stores relevant data from all .sav files in d struct
restore, f[0]
awl=2.*(sgui.HAN + asin(sgui.RAT)) *!RADEG
awd=2.*asin(sgui.RAT)  *!RADEG

d={awl:awl, awd:awd, hgt:sgui.HGT, rat:sgui.RAT, han:sgui.HAN *!RADEG, rot:sgui.ROT*!RADEG, lon:sgui.LON*!RADEG , lat:sgui.LAT*!RADEG,date:sgui.ERUPTIONDATE}
for i=1, nf-1 do begin
  restore, f[i]
  awl=2.*(sgui.HAN + asin(sgui.RAT)) *!RADEG
  awd=2.*asin(sgui.RAT) *!RADEG
  d=[d, {awl:awl, awd:awd, hgt:sgui.HGT, rat:sgui.RAT, han:sgui.HAN *!RADEG, rot:sgui.ROT*!RADEG, lon:sgui.LON*!RADEG , lat:sgui.LAT*!RADEG,date:sgui.ERUPTIONDATE}]
  print, sgui.ERUPTIONDATE
endfor
d=d[sort(d.date)] ; sorts by date

;delte last point 
;   stop
; d=d[0:nf-4]
;plots settings
mydevice = !D.NAME
SET_PLOT, 'PS'
DEVICE, FILENAME=DIR + '/tevo.ps', /LANDSCAPE;,xoffset=0.5, yoffset=0.5;, XSIZE=11, YSIZE=4,/inches
!P.Multi = [0,5,2,0,1]
FONT=1.
MARGIN=[3,4]
XMARGIN=[6,2]
YLOG=0
!P.Color = '000000'xL
!P.Background = 'FFFFFF'xL

;Plots
time=str2utc(d.date)
time=float(time.time)/1000./3600.
;window, 2, xsize=1000, ysize=800
plot, time,  d.hgt, ylog=YLOG, psym=-2, ytitle='HEIGHT [Rs]', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
      xtitle='time [h]', yrange=[min(d.hgt)*0.95,max(d.hgt)*1.05]
plot, d.hgt, d.han, ylog=YLOG, psym=-2, ytitle='HALF ANGLE [Deg]', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
      xtitle='Height [Rs]', yrange=[min(d.han)*0.95,max(d.han)*1.05]
plot, d.hgt, d.lon, ylog=YLOG, psym=-2, ytitle='Lon[Deg]', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
      xtitle='Height [Rs]', yrange=[min(d.lon)*0.95,max(d.lon)*1.05]
plot, d.hgt, d.lat, psym=-2, ytitle='LAT [Deg]', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
      xtitle='Height [Rs]', yrange=[min(d.lat)*1.05,max(d.lat)*0.95]
plot, d.hgt, d.rat, ylog=YLOG, psym=-2, ytitle='RATIO', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
      xtitle='Height [Rs]', yrange=[min(d.rat)*0.95,max(d.rat)*1.05]
plot, d.hgt, d.rot, ylog=YLOG, psym=-2, ytitle='TILT ANGLE [DEG]', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
      xtitle='Height [Rs]', yrange=[min(d.rot)*0.80, max(d.rot)*1.20]
plot, d.hgt, d.awl, ylog=YLOG, psym=-2, ytitle='AWL [Deg]', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
      xtitle='Height [Rs]', yrange=[min(d.awl)*0.95,max(d.awl)*1.05]
plot, d.hgt, d.awd, psym=-2, ytitle='AWD [Deg]', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
      xtitle='Height [Rs]', yrange=[min(d.awd)*0.95,max(d.awd)*1.05]
dawl=deriv(time, d.awl)
plot, d.hgt, dawl, psym=-4, linestyle=0, $
      xtitle='Height [Rs]', ytitle='Derivative [Deg/h]', yrange=[-max(dawl)*0.05,max(dawl)*1.05], charsize=FONT,$
      YMARGIN=MARGIN,xmargin=XMARGIN
dawd=deriv(time,d.awd)
oplot, d.hgt, dawd, psym=-2
legend,[' AWL  ',' AWD  '],psym=[4,2], /left,charsize=0.7
plot, d.hgt, dawl-dawd, psym=-2, linestyle=0, $
      xtitle='Height [Rs]', ytitle='Diff. [Deg/h]', yrange=[min(dawl-dawd)*0.95, max(dawl-dawd,/NaN)*1.05],$
     charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN

; Close the PostScript file:
DEVICE, /CLOSE
SET_PLOT, mydevice

end
; ;delte last point 
; ;d=d[0:nf-2]
; ;plots settings
; mydevice = !D.NAME
; SET_PLOT, 'PS'
; DEVICE, FILENAME=DIR + '/tevo.ps', /LANDSCAPE
; !P.Multi = [0,5,2,0,1]
; FONT=4.5
; MARGIN=[5,5]
; YLOG=0
; !P.Color = '000000'xL
; !P.Background = 'FFFFFF'xL

; ;Plots
; time=str2utc(d.date)
; time=float(time.time)/1000./3600.
; ;window, 2, xsize=1000, ysize=800
; plot, time,  d.hgt, ylog=YLOG, psym=-2, ytitle='HEIGHT [Rs]', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
;       xtitle='time [h]', yrange=[min(d.hgt)*0.95,max(d.hgt)*1.05]
; plot, d.hgt, d.han, ylog=YLOG, psym=-2, ytitle='HALF ANGLE [Deg]', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
;       xtitle='Height [Rs]', yrange=[min(d.han)*0.95,max(d.han)*1.05]
; plot, d.hgt, d.lon, ylog=YLOG, psym=-2, ytitle='Lon[Deg]', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
;       xtitle='Height [Rs]', yrange=[min(d.lon)*0.95,max(d.lon)*1.05]
; plot, d.hgt, d.lat, psym=-2, ytitle='LAT [Deg]', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
;       xtitle='Height [Rs]', yrange=[min(d.lat)*0.95,max(d.lat)*1.05]
; plot, d.hgt, d.rat, ylog=YLOG, psym=-2, ytitle='RATIO', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
;       xtitle='Height [Rs]', yrange=[min(d.rat)*0.95,max(d.rat)*1.05]
; plot, d.hgt, d.rot, ylog=YLOG, psym=-2, ytitle='TILT ANGLE [DEG]', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
;       xtitle='Height [Rs]', yrange=((abs(min(d.rot)) gt abs(max(d.rot))) * (-1)) * [min(abs(d.rot))*0.80, max(abs(d.rot))*1.20]
; plot, d.hgt, d.awl, ylog=YLOG, psym=-2, ytitle='AWL [Deg]', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
;       xtitle='Height [Rs]', yrange=[min(d.awl)*0.95,max(d.awl)*1.05]
; plot, d.hgt, d.awd, psym=-2, ytitle='AWD [Deg]', charsize=FONT, YMARGIN=MARGIN,xmargin=XMARGIN, $
;       xtitle='Height [Rs]', yrange=[min(d.awd)*0.95,max(d.awd)*1.05]
; dawl=deriv(time, d.awl)
; plot, d.hgt, dawl, psym=-4, linestyle=0,title= 'Derivatives of the Angular Widths', $
;       xtitle='Height [Rs]', ytitle='Derivative [Deg/h]', yrange=[-max(dawl)*0.05,max(dawl)*1.05], font=FONT, YMARGIN=MARGIN
; dawd=deriv(time,d.awd)
; oplot, d.hgt, dawd, psym=-2
; legend,[' AWL  ',' AWD  '],psym=[4,2], /left,charsize=0.8
; plot, time, dawl-dawd, psym=-2, linestyle=0,title= 'Difference of the Derivatives', $
;       xtitle='Height [Rs]', ytitle='Diff. [Deg/h]', yrange=[min(dawl-dawd)*0.95, max(dawl-dawd,/NaN)*1.05], font=FONT, YMARGIN=MARGIN

; ; Close the PostScript file:
; DEVICE, /CLOSE
; SET_PLOT, mydevice

; end
