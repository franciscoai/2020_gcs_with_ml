;+
; $Id: vol.pro,v 1.1 2006/09/08 16:03:22 nathan Exp $
;
; PURPOSE:
;  Read a cloud density file for the raytrace front-end dragger
; CATEGORY:
;  raytracing, data handling, visualization
; INPUTS:
;  modelid : the model you want to load the cloud file
; OUTPUTS:
;  cube : the density cube
;
;-

PRO vol, modelid, cube

;we need a 3-d array (cube) of Ne dens. values
;no bigger than about 256,256,256

;which,'cloud.pro',outfile=x
mypath = getenv('RT_DATAPATH')+path_sep()

ofile = mypath + "output" + strtrim(string(modelid),2) + ".txt"
print,'Starting vol'
print,ofile

openr,lun,ofile,ERROR=err, /GET_LUN

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

print,'Starting parsing...'
modelidfile=0L
cubesidenbpix=0L
cubesidersun=0.

readf,lun,modelidfile,cubesidenbpix,cubesidersun

cube=fltarr(cubesidenbpix,cubesidenbpix,cubesidenbpix)
reso=cubesidenbpix/cubesidersun
start=cubesidenbpix/2.

while NOT EOF(lun) DO BEGIN
    readf,lun,xt,yt,zt,bt
    xxx=fix(xt*reso+start)
    yyy=fix(yt*reso+start)
    zzz=fix(zt*reso+start)
    
    cube[xxx,yyy,zzz] = $
      cube[xxx,yyy,zzz] + bt
end

free_lun,lun

xvolume,cube*100

END
