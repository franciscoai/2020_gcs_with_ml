;+
; PURPOSE: Load a cloud file for the rt dragger front-end
; CATEGORY: raytracing
; $Id: cloud.pro,v 1.1 2006/09/08 16:03:19 nathan Exp $
;-

PRO cloud,modelid,neangx,neangy,neangz

;which,'cloud.pro',outfile=x
mypath = getenv('RT_DATAPATH')+path_sep()

ofile = mypath+ "output" + strtrim(string(modelid),2) + ".txt"
print,'Starting cloud'
print,ofile

openr,lun,ofile,ERROR=err, /GET_LUN

; ---- skip first line
modelidfile=0L
cubesidenbpix=0L
cubesidersun=0.

readf,lun,modelidfile,cubesidenbpix,cubesidersun

if (err ne 0) then begin
    base = WIDGET_BASE(TITLE="Error", xsize=300, ysize=100, /COLUMN, $
                      XOFFSET=300, YOFFSET=300)
    label_text = WIDGET_LABEL(base, value="Ne Density File does not exist", /ALIGN_CENTER)
    label_text = WIDGET_LABEL(base, value="Please see MANUAL for help", /ALIGN_CENTER)
    label_blank = WIDGET_LABEL(base, value='')
    exit_butt = WIDGET_BUTTON(base, value="OK", /ALIGN_CENTER, EVENT_PRO="EXIT_PRO")
    WIDGET_CONTROL, base, /REALIZE
    XMANAGER, 'Error', base
    print,'Ne density file does not exist, see MANUAL for help'
    return
end


i=double(0)

x=fltarr(252012)
y=fltarr(252012)
z=fltarr(252012)
b=fltarr(252012)

;print,'Starting parsing...'


i= double(0)
while NOT EOF(lun) DO BEGIN
    readf,lun,yt,xt,zt,bt
    if (bt gt 0) then begin
        x[i] = xt
        y[i] = yt
        z[i] = zt
        b[i] = bt
        i = i + 1D
    end
end

print,'Done parsing'

close,lun, /ALL

size = n_elements(where(b gt 0))
xn=fltarr(size)
yn=fltarr(size)
zn=fltarr(size)

xn=x[where(b gt 0)]
yn=y[where(b gt 0)]
zn=z[where(b gt 0)]

;dragger,xn,yn,zn,neangx=neangx,neangy=neangy,neangz=neangz

dragger,xn,yn,zn,/SOLID,neangx=neangx,neangy=neangy,neangz=neangz

END
