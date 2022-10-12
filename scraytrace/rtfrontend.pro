;+
; $Id: rtfrontend.pro,v 1.1 2006/09/08 16:03:21 nathan Exp $
;
; PROJECT:
;  SOHO-LASCO-C2/3 / STEREO-SECCHI-COR1/2
;
; NAME: rtfrontend.pro
;
; PURPOSE:
;  Widget Front-End to White-light Raytracing program
;
; CATEGORY: gui, raytracing
;
; DESCRIPTION: 
;
; CALLING SEQUENCE: rtfrontend
;
; INPUTS: nil
;
; OUTPUTS: -
;
;
; INPUT KEYWORDS: nil
;
; OUTPUT KEYWORDS: nil
;
; HISTORY: Coded by Nishant Patel, NRL 2004
;
;-

;----------------
PRO BROWSE_FILE, ev
common share1, WIDS, COMMONMP
x=pickfile(TITLE='Save As:')
WIDGET_CONTROL, WIDS.save, SET_VALUE=x
END
;----------------
PRO SELECTOR_EV, ev
common share1
;combines and shows all images numbered in xnums, using the cutoff

WIDGET_CONTROL, WIDS.slider, GET_VALUE=cutoff
type = WIDGET_INFO(long(WIDS.type_menu), /DROPLIST_SELECT)
WIDGET_CONTROL, WIDS.rebin, GET_VALUE=rebinfactor
rebinfactor = fix(rebinfactor)
xnums=WIDGET_INFO(WIDS.list, /LIST_SELECT)

if (xnums[0] eq -1) then return

xnums = string(fix(xnums))
RESTORE,WIDS.usersavepath+'/image.sav'+strtrim(string(xnums[0]),2)

temp = "BTOT"
CASE type OF
    0: im = image.sbtot.im
    1: begin
        im = image.sbpol.im
        temp = "BPOL"
    end
    2: begin
        im = image.snetot.im
        temp = "NETOT"
    end
ENDCASE

WIDGET_CONTROL, WIDS.slider_label, SET_VALUE="ALOG10("+temp+" > 1E-"+$
  strtrim(string(cutoff),2)+")"

cutoff = (10.)^(-cutoff)
    if WIDS.AV eq 1 then begin
        WIDGET_CONTROL, WIDS.xs, GET_VALUE=xsize
        WIDGET_CONTROL, WIDS.ys, GET_VALUE=ysize
        xsize = fix(xsize)
        ysize = fix(ysize)
        xcen = 517.8*(xsize/1024.)
        ycen = 531.*(ysize/1024.)
        
        rd = distarr(xsize[0],ysize[0],xcen[0],ycen[0])
                                ;(258.9,265.5) ; FOR 512x512 C3 IMAGE
        rd = ((rd/8.5)>4)<20
        rd=rd-min(rd)
        rd=rd/max(rd)
        wnd,29,rd
        
        im = im * rd
    end

runtotal = alog(im>cutoff)


;xnums is an array of string numbers
for i=1,N_ELEMENTS(xnums)-1 do BEGIN
    RESTORE,WIDS.usersavepath+'/image.sav'+strtrim(string(xnums[i]),2)
    CASE type OF
        0: im = image.sbtot.im
        1: im = image.sbpol.im
        2: im = image.snetot.im
    ENDCASE
    
    if WIDS.AV eq 1 then begin
        WIDGET_CONTROL, WIDS.xs, GET_VALUE=xsize
        WIDGET_CONTROL, WIDS.ys, GET_VALUE=ysize
        xsize = fix(xsize)
        ysize = fix(ysize)
        xcen = 517.8*(xsize/1024.)
        ycen = 531.*(ysize/1024.)
        rd = distarr(xsize[0],ysize[0],xcen[0],ycen[0])
        ;(258.9,265.5) ; FOR 512x512 C3 IMAGE
        rd = ((rd/8.5)>4)<20
        rd=rd-min(rd)
        rd=rd/max(rd)

        im = im * rd
    end
    runtotal = runtotal + alog(im>cutoff)
END

wnd,30,runtotal

END
;----------------------------
PRO SAVIMAGE, ev
common share1

save=pickfile(TITLE='Save As:')
if (save eq '') then return
print,'Saving the results...'

WIDGET_CONTROL, WIDS.slider, GET_VALUE=cutoff
cutoff = (10.)^(-cutoff)
xnums=WIDGET_INFO(WIDS.list, /LIST_SELECT)

if (n_elements(xnums eq 0)) then begin
    print,'You must select 1 or more raytrace images to save.'
    return
end
if (n_elements(xnums eq 0)) then return

xnums = string(fix(xnums))

RESTORE,WIDS.usersavepath+'/image.sav'+strtrim(string(xnums[0]),2)

sbtotim = image.sbtot.im
sbpolim = image.sbpol.im
snetotim = image.snetot.im

sbtotruntotal = alog(sbtotim>cutoff)
sbpolruntotal = alog(sbpolim>cutoff)
snetotruntotal = alog(snetotim>cutoff)

for i=1,N_ELEMENTS(xnums)-1 do BEGIN
    RESTORE,WIDS.usersavepath+'/image.sav'+strtrim(string(xnums[i]),2)
    sbtotruntotal = sbtotruntotal + alog(image.sbtot.im>cutoff)
    sbpolruntotal = sbpolruntotal + alog(image.sbpol.im>cutoff)
    snetotruntotal = snetotruntotal + alog(image.snetot.im>cutoff)
END

image.sbtot.im = sbtotruntotal
image.sbpol.im = sbpolruntotal
image.snetot.im = snetotruntotal

SAVE,image,FILENAME=save
        
;openw,lun,save+'btot.xdr',/xdr,/get_lun
;lstsavestruct,image.sbtot,lun
;free_lun,lun
;
;openw,lun,save+'bpol.xdr',/xdr,/get_lun
;lstsavestruct,image.sbpol,lun
;free_lun,lun
;
;openw,lun,save+'netot.xdr',/xdr,/get_lun
;lstsavestruct,image.snetot,lun
;free_lun,lun

print,'Done.'

END

;----------------------
;This is a comment 
PRO savmovie, s, e, cutoff
common share1

;WIDGET_INFO, WIDS.type_menu, DROPLIST_SELECT=type
type = WIDGET_INFO(long(WIDS.type_menu), /DROPLIST_SELECT)

if (n_elements(cutoff) eq 0) then cutoff = 1e-13
arr = fltarr(512,512,e-s+1)

for i=s,e do begin
    
    RESTORE,WIDS.usersavepath+'/image.sav'+strtrim(string(i),2)

    help,arr[*,*,i-s],i

    CASE type OF
        0: im = image.sbtot.im
        1: im = image.sbpol.im
        2: im = image.snetot.im
    ENDCASE
 
    arr[*,*,i-s] = rebin(alog(im > cutoff),512,512)        
end

;arr = bytscl(arr,MIN=-30,MAX=-20,TOP=200)
arr = bytscl(arr)
generic_movie,arr ;>50

END

;-----------------------
PRO MOVIE_PRO, ev
common share1
;build a movie of windows WIDS.from through WIDS.to
;using specified cutoff

 WIDGET_CONTROL, WIDS.from, GET_VALUE=from
 WIDGET_CONTROL, WIDS.to, GET_VALUE=to
 WIDGET_CONTROL, WIDS.slider, GET_VALUE=cutoff
 
 from = (fix(from))[0]
 to = (fix(to))[0]
 cutoff = (10.)^(-cutoff)

 if to-from eq 0 then begin
     print,'Sorry, you must have at least 2 frames in a movie.' 
     base = WIDGET_BASE(TITLE="Error", xsize=300, ysize=100, /COLUMN, $
                        XOFFSET=300, YOFFSET=300)
     label_text = WIDGET_LABEL(base, value="Sorry, you must have at least 2 frames in a movie.", /ALIGN_CENTER)
     label_text = WIDGET_LABEL(base, value="Please see MANUAL for help", /ALIGN_CENTER)
     label_blank = WIDGET_LABEL(base, value='')
     exit_butt = WIDGET_BUTTON(base, value="OK", /ALIGN_CENTER, EVENT_PRO="EXIT_PRO")
     WIDGET_CONTROL, base, /REALIZE
     XMANAGER, 'Error', base

     ENDIF else $
   savmovie,from,to,cutoff

END
;---------------------
PRO AXIS_PRO, ev
common share1

base = WIDGET_BASE(TITLE="Axis help", xsize=300, ysize=300, /COLUMN)
label_text = WIDGET_LABEL(base, value="ORDER OF AXIS ROTATION:", /ALIGN_CENTER)
label_text = WIDGET_LABEL(base, value="Z then Y then X", /ALIGN_CENTER)
label_text = WIDGET_LABEL(base, value="-------------------", /ALIGN_CENTER)
label_text = WIDGET_LABEL(base, value="Z: The Viewer (Optical) axis", /ALIGN_CENTER)
label_text = WIDGET_LABEL(base, value="Y: The Horizontal axis", /ALIGN_CENTER)
label_text = WIDGET_LABEL(base, value="X: The Solar Rotation axis", /ALIGN_CENTER)
label_text = WIDGET_LABEL(base, value="-------------------", /ALIGN_CENTER)
exit_butt = WIDGET_BUTTON(base, value="OK", /ALIGN_CENTER, EVENT_PRO="EXIT_PRO")
WIDGET_CONTROL, base, /REALIZE
XMANAGER, 'Axishelp', base
END
;---------------------
PRO ABOUT_PRO, ev
base = WIDGET_BASE(TITLE="About", xsize=600, ysize=150, /COLUMN)
label_text = WIDGET_LABEL(base, value="Frontend by Nishant Patel", /ALIGN_CENTER)
label_text = WIDGET_LABEL(base, $
                          value="Dragger by Nishant Patel, Dr. Russell Howard, and Arnaud Thernisien",$
                          /ALIGN_CENTER)
label_text = WIDGET_LABEL(base, value="Raytrace by Arnaud Thernisien", /ALIGN_CENTER)
label_text = WIDGET_LABEL(base, $
                          value="Models by Arnaud Thernisien, Dr. Russell Howard, Angelos Vourlidas et. al.",$
                          /ALIGN_CENTER)
label_text = WIDGET_LABEL(base, $
                          value="Movie Maker by Scott Paswaters", $
                          /ALIGN_CENTER)
exit_butt = WIDGET_BUTTON(base, value="OK", /ALIGN_CENTER, EVENT_PRO="EXIT_PRO")
WIDGET_CONTROL, base, /REALIZE
XMANAGER, 'About', base
END
;----------------------
PRO CLEAR_WINDOWS, ev
common share1
  ;for i=0,30 do begin
      ;window,i
      ;wdelete,i
  ;end
while !D.WINDOW ne -1 do begin
    wdelete
end
END
;----------------------
PRO WIDGET2_EVENT, ev
common share1

WIDS.AV = ev.SELECT
SELECTOR_EV, 1

if ev.SELECT eq 1 then print,'Note: Anti-Vignetting function aligned for C3 Images'

END
;----------------------
PRO SELECTOR_PRO, ev
common share1

;destroy last Selector
if WIDS.base2 ne 0 then begin
    WIDGET_CONTROL, WIDS.base2, /DESTROY
    WIDS.base2 = 0
end

if WIDS.whichnum gt 0 then begin
    ;WIDGET_CONTROL, WIDS.base, SENSITIVE=0
    base2 = WIDGET_BASE(RESOURCE_NAME = 'base2', COLUMN=1, $
                        TITLE="Selector", XOFFSET=200, YOFFSET=200, $
                        GROUP_LEADER=WIDS.base, MBAR=barBase, $
                        TLB_FRAME_ATTR=8)

    WIDS.base2 = base2

    wFileButton = WIDGET_BUTTON(barBase, VALUE= 'File', /MENU)

    wSaveButton = WIDGET_BUTTON(wFileButton, EVENT_PRO='SAVIMAGE', $
                                VALUE='Save', UVALUE='SAVE')

    wImportButton = WIDGET_BUTTON(wFileButton, SENSITIVE=0, $
                                  VALUE='Import', UVALUE='IMPORT')

    wExportButton = WIDGET_BUTTON(wFileButton, SENSITIVE=0, $
                                  VALUE='Export', UVALUE='EXPORT')

    wPrintButton = WIDGET_BUTTON(wFileButton, SENSITIVE=0, $
                                 VALUE='Print', UVALUE='PRINT')
    
    wCloseButton = WIDGET_BUTTON(wFileButton, EVENT_PRO='EXIT_SELECTOR_PRO', $
                                VALUE='Close', UVALUE='CLOSE')
    
    wViewButton = WIDGET_BUTTON(barBase, VALUE= 'View', /MENU)

    wClearButton = WIDGET_BUTTON(wViewButton, VALUE='Clear Windows', $
                                 EVENT_PRO='CLEAR_WINDOWS')

    wRefreshButton = WIDGET_BUTTON(wViewButton, VALUE='Refresh Raytrace List', $
                                 EVENT_PRO='SELECTOR_PRO')

    base_hor = WIDGET_BASE(base2, column=5,  /base_align_top)
    label_blank = WIDGET_LABEL(base_hor, $
                               value='                    ')
    
    label_select = WIDGET_LABEL(base_hor, /ALIGN_LEFT, VALUE="Select 1 or more models:")
    label_blank = WIDGET_LABEL(base_hor, $
                               value='                    ')
    WIDS.type_menu = WIDGET_DROPLIST(base_hor, /ALIGN_CENTER, $
                                value=['Total Brightness','Polarized Brightness','Total Ne Density'], $
                                title='Show: ', EVENT_PRO="SELECTOR_EV")
    WIDS.AV = 0
    WIDS.anti_vignetting = CW_BGROUP(base_hor, ['Anti-Vignetting'], /NONEXCLUSIVE)

    label_blank = WIDGET_LABEL(base2, VALUE=" ")
    label_select = WIDGET_LABEL(base2, VALUE="Window/ModelID          "+$
                                "obsposx*y*z              "+$
                                "   angx*y*z                     neposx*y*z"+$
                                "                     angx*y*z"+$
                                "             range", /ALIGN_LEFT)
    values = indgen(WIDS.whichnum)
    ;print,values
    values = strtrim(string(values),2)+"/"
    ;print,WIDS.whichnum
    ;help,values
    for i=0,WIDS.whichnum-1 do BEGIN
        values[i] = values[i] + " " + WIDS.info[i]
    END
    ;print,values
    
    
    WIDS.list = WIDGET_LIST(base2,VALUE=values, YSIZE=20, XSIZE=145, $
                            /MULTIPLE, EVENT_PRO="SELECTOR_EV")
    
    base_slider = WIDGET_BASE(base2, /ROW)
    WIDS.slider = WIDGET_SLIDER(base_slider, MAXIMUM=19, MINIMUM=11, VALUE=13,  $
                                EVENT_PRO="SELECTOR_EV", /DRAG, XSIZE=95)
    WIDS.slider_label = WIDGET_LABEL(base_slider, VALUE="ALOG10(BTOT > 1E-13)", XSIZE=195)

    label_movie1 = WIDGET_LABEL(base_slider, VALUE="Show a movie of windows")
    WIDS.from = WIDGET_TEXT(base_slider, /EDITABLE, VALUE='0', XSIZE=3)
    label_movie2 = WIDGET_LABEL(base_slider, VALUE="through")
    WIDS.to = WIDGET_TEXT(base_slider, /EDITABLE, $
                          VALUE=strtrim(string(WIDS.whichnum-1),2), XSIZE=3)

    label_movie4 = WIDGET_LABEL(base_slider, VALUE="  ")
    button_go = WIDGET_BUTTON(base_slider, value=' Go! ', EVENT_PRO="MOVIE_PRO")
    label_rebin = WIDGET_LABEL(base_slider, VALUE="   Rebin: x", sensitive=0)
    WIDS.rebin = WIDGET_TEXT(base_slider, VALUE='1', XSIZE=2, sensitive=0)
    label_blank = WIDGET_LABEL(base_slider, VALUE="  ")
    button_close = WIDGET_BUTTON(base_slider, value='Close', UVALUE='CLOSE', EVENT_PRO="EXIT_SELECTOR_PRO", xsize=120, ysize=15)
    WIDGET_CONTROL, base2, /REALIZE
    XMANAGER, 'Widget2', base2
ENDIF ELSE print,'Please calculate a raytrace first.'

END

;------------------------------------------
PRO PRINT_PRO, ev
common share1

;destroy last Selector
if WIDS.base2 ne 0 then begin
    WIDGET_CONTROL, WIDS.base2, /DESTROY
    WIDS.base2 = 0
end

WIDGET_CONTROL, WIDS.xs, GET_VALUE=xs
WIDGET_CONTROL, WIDS.ys, GET_VALUE=ys
WIDGET_CONTROL, WIDS.fovpix, GET_VALUE=fovpix
fovpix = '1.*'+fovpix
;print,fovpix

WIDGET_CONTROL, WIDS.obsposx, GET_VALUE=obsposx
WIDGET_CONTROL, WIDS.obsposy, GET_VALUE=obsposy
WIDGET_CONTROL, WIDS.obsposz, GET_VALUE=obsposz

WIDGET_CONTROL, WIDS.obsangx, GET_VALUE=obsangx
WIDGET_CONTROL, WIDS.obsangy, GET_VALUE=obsangy
WIDGET_CONTROL, WIDS.obsangz, GET_VALUE=obsangz

WIDGET_CONTROL, WIDS.neposx, GET_VALUE=neposx
WIDGET_CONTROL, WIDS.neposy, GET_VALUE=neposy
WIDGET_CONTROL, WIDS.neposz, GET_VALUE=neposz

WIDGET_CONTROL, WIDS.neangx, GET_VALUE=neangx
WIDGET_CONTROL, WIDS.neangy, GET_VALUE=neangy
WIDGET_CONTROL, WIDS.neangz, GET_VALUE=neangz

WIDGET_CONTROL, WIDS.losnbp, GET_VALUE=losnbp
WIDGET_CONTROL, WIDS.losunits, GET_VALUE=losunits

WIDGET_CONTROL, WIDS.lstart, GET_VALUE=lstart
WIDGET_CONTROL, WIDS.lend, GET_VALUE=lend

WIDGET_CONTROL, WIDS.modelid, GET_VALUE=modelid

WIDGET_CONTROL, WIDS.modparam, GET_VALUE=modparam

fakelasco = string(WIDS.FL)

CASE WIDS.C OF
    0: imagetype = 'c2image=0, c3image=0'
    1: imagetype = '/c2image, c3image=0'
    2: imagetype = 'c2image=0, /c3image'
    ELSE: imagetype = 'c2image=0, c3image=0'
ENDCASE

inhibit = WIDS.IW

WIDGET_CONTROL, WIDS.fovpixunits, GET_VALUE=fovpixunits
WIDGET_CONTROL, WIDS.obsangunits, GET_VALUE=obsangunits
WIDGET_CONTROL, WIDS.neangunits, GET_VALUE=neangunits
WIDGET_CONTROL, WIDS.obsposunits, GET_VALUE=obsposunits
WIDGET_CONTROL, WIDS.neposunits, GET_VALUE=neposunits

WIDGET_CONTROL, WIDS.numsteps, GET_VALUE=numsteps
WIDGET_CONTROL, WIDS.increment, GET_VALUE=increment

;-----------------------------------------------------
;Conversion time

Txs = long(xs)
Tys = long(ys)
abc = execute('Tfovpix=float('+fovpix[0]+')')

;convert to radians
CASE fovpixunits OF
    'ArcSec/Pixel': Tfovpix = Tfovpix /3600 *!dtor
    'ArcMin/Pixel': Tfovpix = Tfovpix /60 *!dtor
    'ArcDeg/Pixel': Tfovpix = Tfovpix *!dtor
    'Rad/Pixel': Tfovpix = Tfovpix
ENDCASE

Tobsposx = float(obsposx)
Tobsposy = float(obsposy)
Tobsposz = float(obsposz)

CASE obsposunits OF
    'Rsun': BEGIN
    END
    'AU': BEGIN
        Tobsposx = Tobsposx * 215
        Tobsposy = Tobsposy * 215
        Tobsposz = Tobsposz * 215
    END
ENDCASE

Tobsangx = float(obsangx)
Tobsangy = float(obsangy)
Tobsangz = float(obsangz)
;convert to radians
CASE obsangunits OF
    'Degrees': BEGIN
        Tobsangx = Tobsangx *!dtor
        Tobsangy = Tobsangy *!dtor
        Tobsangz = Tobsangz *!dtor
    END
    'Minutes': BEGIN
        Tobsangx = Tobsangx /60 *!dtor
        Tobsangy = Tobsangy /60 *!dtor
        Tobsangz = Tobsangz /60 *!dtor
    END
    'Seconds': BEGIN
        Tobsangx = Tobsangx /3600 *!dtor
        Tobsangy = Tobsangy /3600 *!dtor
        Tobsangz = Tobsangz /3600 *!dtor
    END
    'Radians': BEGIN 
    END
ENDCASE

Tneposx = float(neposx)
Tneposy = float(neposy)
Tneposz = float(neposz)

CASE neposunits OF
    'Rsun': BEGIN
    END
    'AU': BEGIN
        Tneposx = Tneposx * 215.
        Tneposy = Tneposy * 215.
        Tneposz = Tneposz * 215.
    END
ENDCASE

Tneangx = float(neangx)
Tneangy = float(neangy)
Tneangz = float(neangz)

CASE neangunits OF
    'Degrees': BEGIN
        Tneangx = Tneangx *!dtor
        Tneangy = Tneangy *!dtor
        Tneangz = Tneangz *!dtor
    END
    'Minutes': BEGIN
        Tneangx = Tneangx /60 *!dtor
        Tneangy = Tneangy /60 *!dtor
        Tneangz = Tneangz /60 *!dtor
    END
    'Seconds': BEGIN
        Tneangx = Tneangx /3600 *!dtor
        Tneangy = Tneangy /3600 *!dtor
        Tneangz = Tneangz /3600 *!dtor
    END
    'Radians': BEGIN 
    END
ENDCASE

Tlosnbp = (long(losnbp))[0]
Tlstart = float(lstart)
Tlend = float(lend)

CASE losunits OF 
    'Rsun': BEGIN
    END
    'AU': BEGIN
        Tlstart = Tlstart * 215.
        Tlend = Tlend * 215.
    END
ENDCASE

Tmodelid = (long(modelid))[0]
Tmodparam = modparam
Tnumsteps = (fix(numsteps))[0]
Tincrement = (float(increment))[0]
Tfakelasco= fix(fakelasco)

Tc2image=0
Tc3image=0

CASE WIDS.C OF
    1: Tc2image=1
    2: Tc3image=1
    ELSE:
ENDCASE


 print,"-------------------"
 print,"xs:      |" + xs
 print,"ys:      |" + ys
 print,"fovpix:  |" + fovpix + " " + fovpixunits

 print,"obsposx: |" + obsposx + " " + obsposunits
 print,"obsposy: |" + obsposy + " " + obsposunits
 print,"obsposz: |" + obsposz + " " + obsposunits

 print,"obsangx: |" + obsangx + " " + obsangunits
 print,"obsangy: |" + obsangy + " " + obsangunits
 print,"obsangz: |" + obsangz + " " + obsangunits

 print,"neposx:  |" + neposx + " " + neposunits
 print,"neposy:  |" + neposy + " " + neposunits
 print,"neposz:  |" + neposz + " " + neposunits

 print,"neangx:  |" + neangx + " " + neangunits
 print,"neangy:  |" + neangy + " " + neangunits
 print,"neangz:  |" + neangz + " " + neangunits

 print,"losnbp:  |" + losnbp
 print,"lstart:  |" + lstart + " " + losunits
 print,"lend:    |" + lend + " " + losunits

 print,"modelid: |" + modelid
 print,"modparam:|" + modparam
 print,"fakelsco:|" + fakelasco
 print,"img type:|" + imagetype

 print,"-------------------"
 print,"COMMAND:"

fovpix = strtrim(string(Tfovpix),2)
obsangx = strtrim(string(Tobsangx),2)
obsangy = strtrim(string(Tobsangy),2)
obsangz = strtrim(string(Tobsangz),2)
obsposx = strtrim(string(Tobsposx),2)
obsposy = strtrim(string(Tobsposy),2)
obsposz = strtrim(string(Tobsposz),2)
neangx = strtrim(string(Tneangx),2)
neangy = strtrim(string(Tneangy),2)
neangz = strtrim(string(Tneangz),2)
neposx = strtrim(string(Tneposx),2)
neposy = strtrim(string(Tneposy),2)
neposz = strtrim(string(Tneposz),2)
lstart = strtrim(string(Tlstart),2)
lend = strtrim(string(Tlend),2)

temps = strtrim(modelid,2)
if strlen(temps) eq 1 then temps = ' ' + temps

TneangxOrig = Tneangx
TneangyOrig = Tneangy
TneangzOrig = Tneangz
TlstartOrig = Tlstart[0]
TlendOrig = Tlend[0]

if (modparam eq '') OR (modparam eq '0') OR (modparam eq '0.') $
  then TmodparamOrig = 0 else TmodparamOrig=COMMONMP

help,TmodparamOrig

;START MOVIE LOOP HERE
base = WIDGET_BASE(GROUP_LEADER=WIDS.base, title='Calculating', /ALIGN_CENTER, $
                   xoffset=350, yoffset=350,xsize=155,ysize=50, /ROW)
label = WIDGET_LABEL(base, /ALIGN_CENTER, value='    Working...0 of 0  ')
WIDGET_CONTROL, base, /REALIZE
XMANAGER, 'Working', base, /JUST_REG

;NEANGX goes up numsteps in increment deg. increments
for mx=0,Tnumsteps do BEGIN
    WIDGET_CONTROL, label, SET_VALUE='    Working...' + $
      strtrim(string(mx),2)+' of '+strtrim(string(Tnumsteps),2)

    if (modparam eq '') OR (modparam eq '0') OR (modparam eq '0.') $
      then Tmodparam = 0 else Tmodparam=COMMONMP

    CASE WIDS.AX OF
        0: Tneangx=TneangxOrig + (Tincrement*mx*!dtor)
        1: Tneangy=TneangyOrig + (Tincrement*mx*!dtor)
        2: Tneangz=TneangzOrig + (Tincrement*mx*!dtor)
        3: begin
            Tlstart = TlstartOrig + (Tincrement*mx)
            Tlend = TlendOrig + (Tincrement*mx)
        end
        4: if (modparam eq '') OR (modparam eq '0') OR (modparam eq '0.') $
          then Tmodparam = 0 else begin
            
            Tmodparam=COMMONMP
            help,WIDS.paramnum
            Tmodparam[WIDS.paramnum] = Tmodparam[WIDS.paramnum]+(Tincrement*mx)
            
        end            
    ENDCASE
    
    neangx = strtrim(string(Tneangx),2)
    neangy = strtrim(string(Tneangy),2)
    neangz = strtrim(string(Tneangz),2)
    lstart = strtrim(string(Tlstart[0]),2)
    lend   = strtrim(string(Tlend[0]),2)
    

    WIDS.info[WIDS.whichnum] = temps+"      | "+obsposx+" * "+obsposy+" * "+obsposz+$
      " | "+obsangx+" * "+obsangy+" * "+obsangz+" | "+neposx+" * "+neposy+" * "+neposz+$
      " | "+neangx+" * "+neangy+" * "+neangz+" |["+lstart+","+lend+"]"
    
    ;tempex = "raytracewl,sbtot,sbpol,snetot,imsize=["+xs+","+ys+"],fovpix="+$
    ;  fovpix+",obspos=["+obsposx+","+obsposy+","+obsposz+"],obsang=["+$
    ;  obsangx+","+obsangy+","+obsangz+"],nepos=["+$
    ;  neposx+","+neposy+","+neposz+"],neang=["+$
    ;  neangx+","+neangy+","+neangz+"],losnbp="+losnbp+",losrange=["+ $
    ;  lstart+","+lend+"],modelid="+modelid+",fakelasco="+fakelasco+","+imagetype

    ;if (modparam eq '') OR (modparam eq '0') OR (modparam eq '0.') $
    ;  then modparam = '0' else tempex = tempex + ",modparam="+WIDS.mp
    
    
;implement: file
;---------------------
    ;print,tempex
;---------------------
    ;tempy = execute(tempex(0))

    raytracewl,sbtot,sbpol,snetot,imsize=[Txs,Tys],fovpix=Tfovpix,$
      obspos=[Tobsposx,Tobsposy,Tobsposz],obsang=[Tobsangx,Tobsangy,Tobsangz],$
      nepos=[Tneposx,Tneposy,Tneposz],neang=[Tneangx,Tneangy,Tneangz],$
      losnbp=Tlosnbp,losrange=[Tlstart,Tlend],modelid=Tmodelid,$
      fakelasco=Tfakelasco,c2image=Tc2image,c3image=Tc3image,$
      modparam=Tmodparam


    image = {Sbtot: sbtot, Sbpol: sbpol, Snetot: snetot}

    SAVE,image,FILENAME=WIDS.usersavepath+'/image.sav'+strtrim(string(WIDS.whichnum),2)

    if inhibit eq '0' then begin
        wnd,WIDS.whichnum mod 30,alog10(sbtot.im>1e-13)
        wnd,WIDS.whichnum mod 30+1,alog10(sbpol.im>1e-13)
        wnd,WIDS.whichnum mod 30+2,alog10(snetot.im>1e-13)
    end

    WIDS.whichnum = WIDS.whichnum + 1
    
;END MOVIE LOOP HERE
END

WIDGET_CONTROL,base,/DESTROY

;WIDGET_CONTROL, WIDS.base, SENSITIVE=0

END
;--------------------------------
PRO GO_PRO, ev
common share1

WIDGET_CONTROL, WIDS.button_print, /SENSITIVE
WIDGET_CONTROL, WIDS.numsteps, GET_VALUE=num

base = WIDGET_BASE(TITLE="Confirmation", xsize=100, ysize=100,$
                   XOFFSET=300, YOFFSET=300, /COLUMN)
label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
label_text = WIDGET_LABEL(base, value="Are you Sure?", /ALIGN_CENTER)
label_blank = WIDGET_LABEL(base, value='', YSIZE=20)
base2 = WIDGET_BASE(base, /ROW, /FRAME, /ALIGN_CENTER)
no = WIDGET_BUTTON(base2, value="Cancel", EVENT_PRO="EXIT_PRO")
yes = WIDGET_BUTTON(base2, value="Yes", EVENT_PRO="EXIT_PRO2")

if num gt 1 then begin
    WIDGET_CONTROL, base, /REALIZE
    XMANAGER, 'Confirmation', base
endif else PRINT_PRO,0

END

;-------------------------------
PRO EXIT_PRO2, ev
common share1

WIDGET_CONTROL, WIDS.base, SENSITIVE=1
WIDGET_CONTROL, ev.top, /DESTROY
PRINT_PRO,0

END

;--------------------------------
PRO EXIT_PRO, ev
common share1

WIDGET_CONTROL, WIDS.base, SENSITIVE=1
WIDGET_CONTROL, ev.top, /DESTROY

END
;--------------------------------
PRO EXIT_SELECTOR_PRO, ev
common share1

WIDS.base2 = 0
WIDGET_CONTROL, WIDS.base, SENSITIVE=1
WIDGET_CONTROL, ev.top, /DESTROY

END
;--------------------------------
pro model34_event1,ev
common share1
WIDGET_CONTROL,WIDS.text_file, $
  SET_VALUE=pickfile(TITLE='Select parameter data file')
return
end

pro model34_event2,ev
common share1
base = WIDGET_BASE(title='', GROUP_LEADER=WIDS.base, XOFFSET=350, YOFFSET=350)
;label = WIDGET_LABEL(base, value='Please Wait...')
;WIDGET_CONTROL, base, /REALIZE
;XMANAGER, 'Wait', base

WIDGET_CONTROL, WIDS.text_file, GET_VALUE=filepath

WIDGET_CONTROL, WIDS.TEMPA, GET_VALUE=x

if strlen(filepath) gt 0 then begin
    ; ---- read the fits file
    ns=readfits(filepath)
    COMMONMP=[reform(ns,n_elements(ns))] 

    if n_elements(COMMONMP) gt 100 then WIDGET_CONTROL,WIDS.modparam,SET_VALUE='HIDDEN' else WIDGET_CONTROL,WIDS.modparam,SET_VALUE=string(COMMONMP)

endif else begin
    place=[strsplit(x,','),strlen(x)]
    ncomma=n_elements(place)-1
    param=fltarr(ncomma)
    for i=0,ncomma-1 do begin
        param[i]=strmid(x,place[i],place[i]+place[i+1])
    endfor
    COMMONMP=param
    WIDGET_CONTROL,WIDS.modparam,SET_VALUE=x

endelse

;WIDGET_CONTROL, base, /DESTROY
WIDGET_CONTROL, ev.top, /DESTROY
return
end




PRO MODEL30_EVENT2, ev
common share1
WIDGET_CONTROL,WIDS.text_file, $
  SET_VALUE=pickfile(TITLE='Select parameter data file')
END
PRO MODEL30_EVENT3, ev
common share1
base = WIDGET_BASE(title='', GROUP_LEADER=WIDS.base, XOFFSET=350, YOFFSET=350)
label = WIDGET_LABEL(base, value='Please Wait...')
WIDGET_CONTROL, base, /REALIZE
XMANAGER, 'Wait', base

WIDGET_CONTROL, WIDS.text_file, GET_VALUE=filepath

WIDGET_CONTROL, WIDS.TEMPA, GET_VALUE=x
WIDGET_CONTROL, WIDS.TEMPB, GET_VALUE=y

x=float(fix(x))
y=fix(y)

ns=readfits(filepath)
ns=rebin(ns,x,y)
COMMONMP=[x,y,reform(ns,n_elements(ns))] 

WIDGET_CONTROL,WIDS.modparam,SET_VALUE='HIDDEN'

WIDGET_CONTROL, base, /DESTROY
WIDGET_CONTROL, ev.top, /DESTROY
END

;--------------------------------
PRO MODEL11_EVENT1, ev
common share1
base = WIDGET_BASE(title='', GROUP_LEADER=WIDS.base, XOFFSET=350, YOFFSET=350)
label = WIDGET_LABEL(base, value='Please Wait...')
WIDGET_CONTROL, base, /REALIZE
XMANAGER, 'Wait', base

WIDGET_CONTROL,WIDS.TEMPA, GET_VALUE=CRNUM
CRNUM=strtrim(CRNUM,2)
URL='http://quake.stanford.edu/~wso/Source.250.rad/CR'+CRNUM
;sock_copy,URL,/verb
;rdtxtmagmap,nsheetmap,file='CR'+CRNUM
rdtxtmagmap,nsheetmap,crot=long(CRNUM)
COMMONMP=reform(nsheetmap,n_elements(nsheetmap))

WIDGET_CONTROL,WIDS.modparam,SET_VALUE='HIDDEN'

WIDGET_CONTROL, base, /DESTROY
WIDGET_CONTROL, ev.top, /DESTROY
END
PRO MODEL11_EVENT2, ev
common share1
WIDGET_CONTROL,WIDS.text_file, $
  SET_VALUE=pickfile(TITLE='Select parameter data file')
END
PRO MODEL11_EVENT3, ev
common share1
base = WIDGET_BASE(title='', GROUP_LEADER=WIDS.base, XOFFSET=350, YOFFSET=350)
label = WIDGET_LABEL(base, value='Please Wait...')
WIDGET_CONTROL, base, /REALIZE
XMANAGER, 'Wait', base

WIDGET_CONTROL, WIDS.text_file, GET_VALUE=filepath

ns=readfits(filepath)
;ns=rebin(ns,360,181)
COMMONMP=reform(ns,n_elements(ns))

WIDGET_CONTROL,WIDS.modparam,SET_VALUE='HIDDEN'

WIDGET_CONTROL, base, /DESTROY
WIDGET_CONTROL, ev.top, /DESTROY
END

;--------------------------------
PRO MODEL15_EVENT2, ev
common share1
WIDGET_CONTROL,WIDS.text_file, $
  SET_VALUE=pickfile(TITLE='Select parameter data file')
END
PRO MODEL15_EVENT3, ev
common share1
base = WIDGET_BASE(title='', GROUP_LEADER=WIDS.base, XOFFSET=350, YOFFSET=350)
label = WIDGET_LABEL(base, value='Please Wait...')
WIDGET_CONTROL, base, /REALIZE
XMANAGER, 'Wait', base

WIDGET_CONTROL, WIDS.text_file, GET_VALUE=filepath

COMMONMP=readfits(filepath)

WIDGET_CONTROL,WIDS.modparam,SET_VALUE='HIDDEN'

WIDGET_CONTROL, base, /DESTROY
WIDGET_CONTROL, ev.top, /DESTROY
END
;--------------------------------
PRO DENSITYCUBE_EVENT, ev
common share1
if (ev.select eq 1) then WIDGET_CONTROL, $
  WIDS.modelid, SET_VALUE=strtrim(string(26),2) else $
  WIDGET_CONTROL, WIDS.modelid, SET_VALUE=strtrim(string(25),2)
end
PRO DENSITYCUBE_EVENT2, ev
common share1
WIDGET_CONTROL,WIDS.text_file, $
  SET_VALUE=pickfile(TITLE='Select Ne Density Cube data file')
END
PRO DENSITYCUBE_EVENT3, ev
common share1
base = WIDGET_BASE(title='', GROUP_LEADER=WIDS.base, XOFFSET=350, YOFFSET=350)
label = WIDGET_LABEL(base, value='Please Wait...')
WIDGET_CONTROL, base, /REALIZE
XMANAGER, 'Wait', base

WIDGET_CONTROL, WIDS.text_file, GET_VALUE=filepath

openr,unit,filepath,/GET_LUN,/stdio
readf,unit,xdim
readf,unit,ydim
readf,unit,zdim
mp=fltarr((xdim*ydim*zdim)+7)
mp[0]=xdim
mp[1]=ydim
mp[2]=zdim

for i=3L,((xdim*ydim*zdim)+7-1) do begin
    readf,unit,tempy2
    mp[i] = tempy2
end

COMMONMP = mp
WIDGET_CONTROL,WIDS.modparam,SET_VALUE='HIDDEN'

WIDGET_CONTROL, base, /DESTROY
WIDGET_CONTROL, ev.top, /DESTROY
END
;--------------------------------
PRO MODEL31_EVENT, ev
common share1

WIDGET_CONTROL, WIDS.TEMPARR[0], GET_VALUE=nemax
WIDGET_CONTROL, WIDS.TEMPARR[1], GET_VALUE=nemin
WIDGET_CONTROL, WIDS.TEMPARR[2], GET_VALUE=rh
WIDGET_CONTROL, WIDS.TEMPARR[3], GET_VALUE=dh
WIDGET_CONTROL, WIDS.TEMPARR[4], GET_VALUE=d0
WIDGET_CONTROL, WIDS.TEMPARR[5], GET_VALUE=rc
WIDGET_CONTROL, WIDS.TEMPARR[6], GET_VALUE=rb
WIDGET_CONTROL, WIDS.TEMPARR[7], GET_VALUE=db

COMMONMP = [nemax,nemin,rh,dh,d0,rc,rb,db]

COMMONMP = float(COMMONMP)

WIDGET_CONTROL,WIDS.modparam,SET_VALUE=strcompress(string(COMMONMP, /print))
;print,COMMONMP
WIDGET_CONTROL, ev.top, /DESTROY
END

PRO MODEL31_EVENT2, ev
common share1

WIDGET_CONTROL, WIDS.TEMPARR[0], SET_VALUE='1.5'
WIDGET_CONTROL, WIDS.TEMPARR[1], SET_VALUE='1.'
WIDGET_CONTROL, WIDS.TEMPARR[2], SET_VALUE='14.'
WIDGET_CONTROL, WIDS.TEMPARR[3], SET_VALUE='4.'
WIDGET_CONTROL, WIDS.TEMPARR[4], SET_VALUE='0.3'
WIDGET_CONTROL, WIDS.TEMPARR[5], SET_VALUE='3.'
WIDGET_CONTROL, WIDS.TEMPARR[6], SET_VALUE='2.'
WIDGET_CONTROL, WIDS.TEMPARR[7], SET_VALUE='0.1'

END
;--------------------------------
PRO MODEL29_EVENT, ev
common share1

WIDGET_CONTROL, WIDS.TEMPA, GET_VALUE=d
WIDGET_CONTROL, WIDS.TEMPB, GET_VALUE=s
WIDGET_CONTROL, WIDS.TEMPC, GET_VALUE=h

d = float(d)
s = float(s)
h = float(h)

COMMONMP = [d,s,h]

WIDGET_CONTROL,WIDS.modparam,SET_VALUE=strcompress(string([d,s,h], /print))
WIDGET_CONTROL, ev.top, /DESTROY
END

PRO MODEL29_EVENT2, ev
common share1
WIDGET_CONTROL, WIDS.TEMPA, SET_VALUE='1.'
WIDGET_CONTROL, WIDS.TEMPB, SET_VALUE='1.5'
WIDGET_CONTROL, WIDS.TEMPC, SET_VALUE='1.'
END
;--------------------------------
PRO MODEL27_EVENT, ev
common share1

WIDGET_CONTROL, WIDS.TEMPA, GET_VALUE=r0
WIDGET_CONTROL, WIDS.TEMPB, GET_VALUE=d
WIDGET_CONTROL, WIDS.TEMPC, GET_VALUE=d0

r0 = float(r0)
d = float(d)
d0 = float(d0)

COMMONMP = [r0,d,d0]

WIDGET_CONTROL,WIDS.modparam,SET_VALUE=strcompress(string([r0,d,d0], /print))
WIDGET_CONTROL, ev.top, /DESTROY
END

PRO MODEL27_EVENT2, ev
common share1
WIDGET_CONTROL, WIDS.TEMPA, SET_VALUE='8.'
WIDGET_CONTROL, WIDS.TEMPB, SET_VALUE='2.'
WIDGET_CONTROL, WIDS.TEMPC, SET_VALUE='.3'
END
;--------------------------------
PRO MODEL28_EVENT, ev
common share1

WIDGET_CONTROL, WIDS.TEMPA, GET_VALUE=r0
WIDGET_CONTROL, WIDS.TEMPB, GET_VALUE=d
WIDGET_CONTROL, WIDS.TEMPC, GET_VALUE=len
WIDGET_CONTROL, WIDS.TEMPD, GET_VALUE=d0

r0 = float(r0)
d = float(d)
len = float(len)
d0 = float(d0)

COMMONMP = [r0,d,len,d0]

tempstring = strcompress(string(string([r0,d,len,d0])+',',/print))
strput,tempstring,' ',strlen(tempstring)-1

WIDGET_CONTROL,WIDS.modparam,$
  SET_VALUE=tempstring
WIDGET_CONTROL, ev.top, /DESTROY
END

PRO MODEL28_EVENT2, ev
    common share1
    WIDGET_CONTROL, WIDS.TEMPA, SET_VALUE='8.'
    WIDGET_CONTROL, WIDS.TEMPB, SET_VALUE='2.'
    WIDGET_CONTROL, WIDS.TEMPC, SET_VALUE='2.'
    WIDGET_CONTROL, WIDS.TEMPD, SET_VALUE='.3'
END

;--------------------------------
PRO MODEL33_EVENT, ev
common share1

WIDGET_CONTROL, WIDS.TEMPA, GET_VALUE=d0
WIDGET_CONTROL, WIDS.TEMPB, GET_VALUE=rb
WIDGET_CONTROL, WIDS.TEMPC, GET_VALUE=alpha
WIDGET_CONTROL, WIDS.TEMPD, GET_VALUE=rf
WIDGET_CONTROL, WIDS.TEMPE, GET_VALUE=ratio

d0 = (float(d0))[0]
rb = (float(rb))[0]
alpha = (float(alpha))[0]
rf = (float(rf))[0]
ratio = (float(ratio))[0]

;help,d0,rb,alpha,rf,ratio

COMMONMP = [d0,rb,alpha,rf,ratio]

;stop

tempstring = strcompress(string(string([d0,rb,alpha,rf,ratio])+',',/print))

;help,tempstring

strput,tempstring,' ',strlen(tempstring)-1

WIDGET_CONTROL,WIDS.modparam,$
  SET_VALUE=tempstring
WIDGET_CONTROL, ev.top, /DESTROY
END

PRO MODEL33_EVENT2, ev
    common share1
    WIDGET_CONTROL, WIDS.TEMPA, SET_VALUE='0.7'
    WIDGET_CONTROL, WIDS.TEMPB, SET_VALUE='2.55'
    WIDGET_CONTROL, WIDS.TEMPC, SET_VALUE='30.'
    WIDGET_CONTROL, WIDS.TEMPD, SET_VALUE='10.'
    WIDGET_CONTROL, WIDS.TEMPE, SET_VALUE='0.2'
END

;---------------------------------

PRO MODEL17_EVENT, ev 
    common share1
    WIDGET_CONTROL, WIDS.TEMPA, GET_VALUE=a
    WIDGET_CONTROL, WIDS.TEMPB, GET_VALUE=b
    WIDGET_CONTROL, WIDS.TEMPC, GET_VALUE=c
    WIDGET_CONTROL, WIDS.TEMPD, GET_VALUE=d
    
    a = float(a)
    b = float(b)
    c = float(c)
    d = float(d)
    
    COMMONMP = [a,b,c,d]
    tempstring = strcompress(string(string([a,b,c,d])+',',/print))
    strput,tempstring,' ',strlen(tempstring)-1
    
    WIDGET_CONTROL,WIDS.modparam,$
      SET_VALUE=tempstring
    WIDGET_CONTROL, ev.top, /DESTROY
END

;--------------------------------
PRO SET_MODPARAM,val
common share1

switch1 = 0
if val eq 26 then begin
    val=25
    switch1 = 1
end

CASE val OF


    34 : BEGIN
        
        base = WIDGET_BASE(TITLE="Parameter Widget", $
                           xsize=300, ysize=340,$
                           XOFFSET=300, YOFFSET=300, /COLUMN, $
                           GROUP_LEADER=WIDS.base)
        
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        WIDS.text_file = WIDGET_TEXT(base, value='', /ALIGN_CENTER)
        button_file = WIDGET_BUTTON(base, value="Browse", $
                                    EVENT_PRO='MODEL34_EVENT1')

        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        basex = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(basex, value='Param :')
        text_x = WIDGET_TEXT(basex, value='', /EDITABLE)

        WIDS.TEMPA=text_x
        
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        OK = WIDGET_BUTTON(base, value="OK", EVENT_PRO="MODEL34_EVENT2")
        label_blank = WIDGET_LABEL(base, value='', YSIZE=20)
        

        WIDGET_CONTROL, base, /REALIZE
        XMANAGER, 'model34', base
    END




    31: BEGIN
        base = WIDGET_BASE(TITLE="Model 31 Parameters", $
                           xsize=300, ysize=500,$
                           XOFFSET=300, YOFFSET=300, /COLUMN, $
                           GROUP_LEADER=WIDS.base)
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='Model by Russell Howard')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='-----------------------------')
        basenemax = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(basenemax, value='Ne at Max:')
        text_nemax = WIDGET_TEXT(basenemax, value='1.5', /EDITABLE)

        basenemin = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(basenemin, value='Ne at Min:')
        text_nemin = WIDGET_TEXT(basenemin, value='1.', /EDITABLE)

        baserh = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(baserh, value='Height of leading edge of cyl. at base:')
        text_rh = WIDGET_TEXT(baserh, value='14.', /EDITABLE)

        basedh = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(basedh, value='Diameter of cyl. at top:')
        text_dh = WIDGET_TEXT(basedh, value='4.', /EDITABLE)

        based0 = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(based0, value='1/2 thickness of shell:')
        text_d0 = WIDGET_TEXT(based0, value='0.3', /EDITABLE)

        baserc = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(baserc, value='Radius of curvature of cyl.:')
        text_rc = WIDGET_TEXT(baserc, value='3.', /EDITABLE)

        baserb = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(baserb, value='Height of base of cyl.:')
        text_rb = WIDGET_TEXT(baserb, value='2.', /EDITABLE)

        basedb = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(basedb, value='Diameter of cyl. at base:')
        text_db = WIDGET_TEXT(basedb, value='.1', /EDITABLE)

        WIDS.TEMPARR=[text_nemax,text_nemin,text_rh,text_dh,text_d0,text_rc,text_rb,text_db]

        button_DEFAULTS = WIDGET_BUTTON(base, value="Defaults", $
                                        EVENT_PRO="MODEL31_EVENT2")
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        button_OK = WIDGET_BUTTON(base, value="OK", $
                                 EVENT_PRO="MODEL31_EVENT")

        if (n_elements(COMMONMP) eq n_elements(WIDS.TEMPARR)) then for i=0,7 do begin
            WIDGET_CONTROL,WIDS.TEMPARR[i],SET_VALUE=strtrim(string(COMMONMP[i]),2)
        end

        WIDGET_CONTROL, base, /REALIZE
        XMANAGER, 'MODEL31', base



    END
    30: BEGIN
        base = WIDGET_BASE(TITLE="Model 30 Parameters", $
                           xsize=300, ysize=330,$
                           XOFFSET=300, YOFFSET=300, /COLUMN, $
                           GROUP_LEADER=WIDS.base)
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='2-D File Format:')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='-----------------------')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, value='FITS image file: User-defined')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, value='Must be multiple of XxY')
;        label_text = WIDGET_LABEL(base, /ALIGN_LEFT, $
;                                  value='Or text file, one number per line')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='-----------------------')
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        WIDS.text_file = WIDGET_TEXT(base, value='', /ALIGN_CENTER)
        button_file = WIDGET_BUTTON(base, value="Browse", $
                                    EVENT_PRO='MODEL30_EVENT2')
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        basex = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(basex, value='X:')
        text_x = WIDGET_TEXT(basex, value='360', /EDITABLE)
        basey = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(basey, value='Y:')
        text_y = WIDGET_TEXT(basey, value='181', /EDITABLE)

        OK = WIDGET_BUTTON(base, value="OK", EVENT_PRO="MODEL30_EVENT3")
        label_blank = WIDGET_LABEL(base, value='To use actual Field Map, see Model 11', YSIZE=20)
        
        WIDS.TEMPA=text_x
        WIDS.TEMPB=text_y

        WIDGET_CONTROL, base, /REALIZE
        XMANAGER, 'MODEL30', base
    END
    11: BEGIN
        base = WIDGET_BASE(TITLE="Model 11 Parameters", $
                           xsize=300, ysize=280,$
                           XOFFSET=300, YOFFSET=300, /COLUMN, $
                           GROUP_LEADER=WIDS.base)
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='2-D File Format:')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='-----------------------')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, value='FITS image file: 360x181')
;        label_text = WIDGET_LABEL(base, /ALIGN_LEFT, $
;                                  value='Or text file, one number per line')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='-----------------------')
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        WIDS.text_file = WIDGET_TEXT(base, value='', /ALIGN_CENTER)
        button_file = WIDGET_BUTTON(base, value="Browse", $
                                    EVENT_PRO='MODEL11_EVENT2')
        OK = WIDGET_BUTTON(base, value="OK, Use FITS File", EVENT_PRO="MODEL11_EVENT3")
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='-----------------------')
        label_blank = WIDGET_LABEL(base, value='', YSIZE=20)

        base2 = WIDGET_BASE(base, /ROW, /ALIGN_CENTER)
        label_text = WIDGET_LABEL(base2, value='Or enter CR#:')
        WIDS.TEMPA = WIDGET_TEXT(base2, value='', /EDITABLE)
        button_CRNUM = WIDGET_BUTTON(base2, value='Use CR#', EVENT_PRO='MODEL11_EVENT1')

        WIDGET_CONTROL, base, /REALIZE
        XMANAGER, 'MODEL11', base
    END
    15: BEGIN
        base = WIDGET_BASE(TITLE="Model 15 Parameters", $
                           xsize=300, ysize=340,$
                           XOFFSET=300, YOFFSET=300, /COLUMN, $
                           GROUP_LEADER=WIDS.base)
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='120-Element 1-D Array in FITS file:')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='-----------------------')
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        WIDS.text_file = WIDGET_TEXT(base, value='', /ALIGN_CENTER)
        button_file = WIDGET_BUTTON(base, value="Browse", $
                                    EVENT_PRO='MODEL15_EVENT2')
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        OK = WIDGET_BUTTON(base, value="OK", EVENT_PRO="MODEL15_EVENT3")
        label_blank = WIDGET_LABEL(base, value='', YSIZE=20)
        
        WIDGET_CONTROL, base, /REALIZE
        XMANAGER, 'MODEL15', base
    END
    17: BEGIN
        base = WIDGET_BASE(TITLE="Model 17 Parameters", $
                           xsize=300, ysize=140,$
                           XOFFSET=300, YOFFSET=300, /COLUMN, $
                           GROUP_LEADER=WIDS.base)
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='4-Element Float Array:')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='-----------------------')
        base_params = WIDGET_BASE(base, /ROW,/FRAME, /ALIGN_CENTER)
        label_text = WIDGET_LABEL(base_params, value='[')
        WIDS.TEMPA = WIDGET_TEXT(base_params,/EDITABLE, XSIZE=6)
        WIDS.TEMPB = WIDGET_TEXT(base_params,/EDITABLE, XSIZE=6)
        WIDS.TEMPC = WIDGET_TEXT(base_params,/EDITABLE, XSIZE=6)
        WIDS.TEMPD = WIDGET_TEXT(base_params,/EDITABLE, XSIZE=6)
        label_text = WIDGET_LABEL(base_params, value=']')
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        OK = WIDGET_BUTTON(base, value="OK", EVENT_PRO="MODEL17_EVENT")
        label_blank = WIDGET_LABEL(base, value='', YSIZE=20)
        
        WIDGET_CONTROL, base, /REALIZE
        XMANAGER, 'MODEL17', base
    END
        
    25: BEGIN
        
        base = WIDGET_BASE(TITLE="Density Cube Parameters", $
                           xsize=300, ysize=340,$
                           XOFFSET=300, YOFFSET=300, /COLUMN, $
                           GROUP_LEADER=WIDS.base)
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='Datacube File Format:')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='-----------------------')
        label_text = WIDGET_LABEL(base, /ALIGN_LEFT, value='  First 7 lines:')
        label_text = WIDGET_LABEL(base, /ALIGN_LEFT, $
                                  value='    Dimensions of cube (x,y,z)')
        label_text = WIDGET_LABEL(base, /ALIGN_LEFT, $
                                  value='    Sun Center (x,y,z):')
        label_text = WIDGET_LABEL(base, /ALIGN_LEFT, $
                                  value='    Resolution of one ''Voxel'' in Rsun')
        label_text = WIDGET_LABEL(base, /ALIGN_LEFT, $
                                  value='  One line for each element of cube,')
        label_text = WIDGET_LABEL(base, /ALIGN_LEFT, $
                                  value='     in lexographical order (X,Y,Z).')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='-----------------------')
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        WIDS.text_file = WIDGET_TEXT(base, value='', /ALIGN_CENTER)
        button_file = WIDGET_BUTTON(base, value="Browse", $
                                    EVENT_PRO='DENSITYCUBE_EVENT2')
        if switch1 eq 0 then $
          TI = CW_BGROUP(base, $
                         ['Use Trilinear Interpolation?'], /NONEXCLUSIVE) $
        else TI = CW_BGROUP(base, $
                            ['Use Trilinear Interpolation?'], $
                            /NONEXCLUSIVE, SET_VALUE=[1])
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        OK = WIDGET_BUTTON(base, value="OK", EVENT_PRO="DENSITYCUBE_EVENT3")
        label_blank = WIDGET_LABEL(base, value='', YSIZE=20)
        
        WIDGET_CONTROL, base, /REALIZE
        XMANAGER, 'DensityCube', base
    END
    29: BEGIN
        base = WIDGET_BASE(TITLE="Bow Shock Parameters", $
                           xsize=300, ysize=300,$
                           XOFFSET=300, YOFFSET=300, /COLUMN, $
                           GROUP_LEADER=WIDS.base)
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='Model by Angelos Vourlidas')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='-----------------------------')
        based = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(based, value='d:')
        text_d = WIDGET_TEXT(based, value='1.', /EDITABLE)
        bases = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(bases, value='s:')
        text_s = WIDGET_TEXT(bases, value='1.5', /EDITABLE)
        baseh = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(baseh, value='h:')
        text_h = WIDGET_TEXT(baseh, value='1.', /EDITABLE)

        WIDS.TEMPA=text_d
        WIDS.TEMPB=text_s
        WIDS.TEMPC=text_h
        button_DEFAULTS = WIDGET_BUTTON(base, value="Defaults", $
                                        EVENT_PRO="MODEL29_EVENT2")
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        button_OK = WIDGET_BUTTON(base, value="OK", $
                                 EVENT_PRO="MODEL29_EVENT")
        WIDGET_CONTROL, base, /REALIZE
        XMANAGER, 'MODEL29', base
    END
        27: BEGIN
        base = WIDGET_BASE(TITLE="Spherical Shell Parameters", $
                           xsize=400, ysize=250,$
                           XOFFSET=300, YOFFSET=300, /COLUMN, $
                           GROUP_LEADER=WIDS.base)
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='Model by Russell Howard')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='-----------------------------')
        based = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(based, value='Height of center sphere (r0):')
        text_r0 = WIDGET_TEXT(based, value='8.', /EDITABLE)
        bases = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(bases, value='Radius of sphere (d):')
        text_d = WIDGET_TEXT(bases, value='2.', /EDITABLE)
        baseh = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(baseh, value='1/2 Thickness of shell (d0):')
        text_d0 = WIDGET_TEXT(baseh, value='.3', /EDITABLE)

        WIDS.TEMPA=text_r0
        WIDS.TEMPB=text_d
        WIDS.TEMPC=text_d0
        button_DEFAULTS = WIDGET_BUTTON(base, value="Defaults", $
                                        EVENT_PRO="MODEL27_EVENT2")
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        button_OK = WIDGET_BUTTON(base, value="OK", $
                                 EVENT_PRO="MODEL27_EVENT")
        WIDGET_CONTROL, base, /REALIZE
        XMANAGER, 'MODEL27', base
    END
        28: BEGIN
        base = WIDGET_BASE(TITLE="Cylindrical Shell Parameters", $
                           xsize=400, ysize=300,$
                           XOFFSET=300, YOFFSET=300, /COLUMN, $
                           GROUP_LEADER=WIDS.base)
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='Model by Russell Howard')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='-----------------------------')
        baser0 = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(baser0, value='Height of center sphere (r0):')
        text_r0 = WIDGET_TEXT(baser0, value='8.', /EDITABLE)

        based = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(based, value='Radius of cylinder (d):')
        text_d = WIDGET_TEXT(based, value='2.', /EDITABLE)

        baselen = WIDGET_BASE(base, /ROW, /FRAME)        
        label_text = WIDGET_LABEL(baselen, $
                                  value='Length of cylinder in each dir. (len):')
        text_len = WIDGET_TEXT(baselen, value='2.', /EDITABLE)

        based0 = WIDGET_BASE(base, /ROW, /FRAME)        
        label_text = WIDGET_LABEL(based0, value='1/2 Thickness of shell (d0):')
        text_d0 = WIDGET_TEXT(based0, value='.3', /EDITABLE)

        WIDS.TEMPA=text_r0
        WIDS.TEMPB=text_d
        WIDS.TEMPC=text_len
        WIDS.TEMPD=text_d0

        button_DEFAULTS = WIDGET_BUTTON(base, value="Defaults", $
                                        EVENT_PRO="MODEL28_EVENT2")
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        button_OK = WIDGET_BUTTON(base, value="OK", $
                                 EVENT_PRO="MODEL28_EVENT")
        WIDGET_CONTROL, base, /REALIZE
        XMANAGER, 'MODEL28', base
    END
        33: BEGIN
        base = WIDGET_BASE(TITLE="Tube Shell Parameters", $
                           xsize=400, ysize=340,$
                           XOFFSET=300, YOFFSET=300, /COLUMN, $
                           GROUP_LEADER=WIDS.base)
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='Model by Arnaud Thernisien & Nishant Patel')
        label_text = WIDGET_LABEL(base, /ALIGN_CENTER, $
                                  value='-------------------------------------')
        based0 = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(based0, value='Full thickness of shell (d0):')
        text_d0 = WIDGET_TEXT(based0, value='0.7', /EDITABLE)

        baserb = WIDGET_BASE(base, /ROW, /FRAME)
        label_text = WIDGET_LABEL(baserb, value='Height of bottom of structure (rb):')
        text_rb = WIDGET_TEXT(baserb, value='2.55', /EDITABLE)

        basealpha = WIDGET_BASE(base, /ROW, /FRAME)        
        label_text = WIDGET_LABEL(basealpha, $
                                  value='Angle (deg.) between axis and foot (alpha):')
        text_alpha = WIDGET_TEXT(basealpha, value='30.', /EDITABLE)

        baserf = WIDGET_BASE(base, /ROW, /FRAME)        
        label_text = WIDGET_LABEL(baserf, value='Distance to line-circle junction (rf):')
        text_rf = WIDGET_TEXT(baserf, value='10.', /EDITABLE)

        baseratio = WIDGET_BASE(base, /ROW, /FRAME)        
        label_text = WIDGET_LABEL(baseratio, value='Ratio of tube radius to height (ratio):')
        text_ratio = WIDGET_TEXT(baseratio, value='.2', /EDITABLE)

        WIDS.TEMPA=text_d0
        WIDS.TEMPB=text_rb
        WIDS.TEMPC=text_alpha
        WIDS.TEMPD=text_rf
        WIDS.TEMPE=text_ratio

        button_DEFAULTS = WIDGET_BUTTON(base, value="Defaults", $
                                        EVENT_PRO="MODEL33_EVENT2")
        label_blank = WIDGET_LABEL(base, value='', YSIZE=12)
        button_OK = WIDGET_BUTTON(base, value="OK", $
                                 EVENT_PRO="MODEL33_EVENT")
        WIDGET_CONTROL, base, /REALIZE
        XMANAGER, 'MODEL33', base
    END
ELSE: WIDGET_CONTROL,WIDS.modparam,SET_VALUE='0.'
ENDCASE

END
;---------------------------
PRO PARAM_EVENT, ev
common share1

WIDGET_CONTROL, WIDS.paramnum, GET_VALUE=temp
WIDGET_CONTROL, WIDS.label_increment, SET_VALUE="Param["+temp[0]+"]/Increment:"

WIDS.paramnum = fix(temp[0])

EXIT_PRO,ev
END

;-------------------------------
PRO WIDGET_EVENT, ev
common share1
;print,ev
;help,ev, /STRUCTURES

if ev.ID eq WIDS.axisMenu then begin
    WIDS.AX = ev.index 
    if (ev.index eq 3) then $
      WIDGET_CONTROL, WIDS.label_increment, SET_VALUE="Rsun/Increment:" else $
      if (ev.index eq 4) then begin
        base = WIDGET_BASE(TITLE="Increment Parameter", $
                           xsize=260, ysize=30,$
                           XOFFSET=300, YOFFSET=300, /ROW, $
                           GROUP_LEADER=WIDS.base)
        label_text = WIDGET_LABEL(base, value='Param index to iter:')
        WIDS.paramnum = WIDGET_TEXT(base, /EDITABLE, xsize=15)
        button_OK = WIDGET_BUTTON(base, value="OK", $
                                  EVENT_PRO="PARAM_EVENT")

        WIDGET_CONTROL, base, /REALIZE
        XMANAGER, 'Increment Param', base
    endif else $
        
      WIDGET_CONTROL, WIDS.label_increment, SET_VALUE="Degrees/Increment:"
endif else $
if ev.ID eq WIDS.button_change_modparam then begin
    WIDGET_CONTROL, WIDS.modelid, GET_VALUE=modelid
    modelid = fix(modelid)
    SET_MODPARAM, modelid
endif else $
if ev.ID eq WIDS.inhibit then WIDS.IW = ev.select else $
;if ev.ID eq WIDS.fakelasco then WIDS.FL = ev.SELECT else $
if ev.ID eq WIDS.editObs then begin
    mapped = WIDGET_INFO(long(WIDS.base_frame2), /MAP)
    if (mapped eq 0) then begin
        WIDGET_CONTROL, WIDS.editObs, SET_BUTTON=1
        WIDGET_CONTROL, WIDS.base_frame2, /MAP
        WIDGET_CONTROL, WIDS.base_frame2, XSIZE=-1,YSIZE=-1
    endif else begin
        WIDGET_CONTROL, WIDS.editObs, SET_BUTTON=0
        WIDGET_CONTROL, WIDS.base_frame2, MAP=0
        WIDGET_CONTROL, WIDS.base_frame2, XSIZE=1,YSIZE=1
    end
endif else $
if ev.ID eq WIDS.editNe then begin
    mapped = WIDGET_INFO(long(WIDS.base_frame3), /MAP)
    if (mapped eq 0) then begin
        WIDGET_CONTROL, WIDS.editNe, SET_BUTTON=1
        WIDGET_CONTROL, WIDS.base_frame3, /MAP
        WIDGET_CONTROL, WIDS.base_frame3, XSIZE=-1,YSIZE=-1
    endif else begin
        WIDGET_CONTROL, WIDS.editNe, SET_BUTTON=0
        WIDGET_CONTROL, WIDS.base_frame3, MAP=0
        WIDGET_CONTROL, WIDS.base_frame3, XSIZE=1,YSIZE=1
    end
endif else $
if ev.ID eq WIDS.editMovie then begin
    mapped = WIDGET_INFO(long(WIDS.base_movie), /MAP)
    if (mapped eq 0) then begin
        WIDGET_CONTROL, WIDS.editMovie, SET_BUTTON=1
        WIDGET_CONTROL, WIDS.base_movie, /MAP
        WIDGET_CONTROL, WIDS.base_movie, XSIZE=-1,YSIZE=-1
    endif else begin
        WIDGET_CONTROL, WIDS.editMovie, SET_BUTTON=0
        WIDGET_CONTROL, WIDS.base_movie, MAP=0
        WIDGET_CONTROL, WIDS.base_movie, XSIZE=1,YSIZE=1
    end
endif else $
if ev.ID eq WIDS.cimage then begin
    WIDS.C = ev.INDEX
    if (ev.INDEX eq 1) OR (ev.INDEX eq 2) then WIDS.FL=1 else WIDS.FL=0
endif else $
if ev.ID eq WIDS.volume then begin
    WIDGET_CONTROL, WIDS.modelid, GET_VALUE=modelid
    modelid = strtrim(modelid,2)
    modelid = fix(modelid)
    
    vol, modelid, cube
    
endif else $
if ev.ID eq WIDS.cloud then begin
    WIDGET_CONTROL, WIDS.neangx, GET_VALUE=neangx
    WIDGET_CONTROL, WIDS.neangy, GET_VALUE=neangy
    WIDGET_CONTROL, WIDS.neangz, GET_VALUE=neangz
    WIDGET_CONTROL, WIDS.neangunits, GET_VALUE=neangunits
    Tneangx = float(neangx)
    Tneangy = float(neangy)
    Tneangz = float(neangz)
    
    CASE neangunits OF
        'Degrees': BEGIN
            Tneangx = Tneangx *!dtor
            Tneangy = Tneangy *!dtor
            Tneangz = Tneangz *!dtor
        END
        'Minutes': BEGIN
            Tneangx = Tneangx /60 *!dtor
            Tneangy = Tneangy /60 *!dtor
            Tneangz = Tneangz /60 *!dtor
        END
        'Seconds': BEGIN
            Tneangx = Tneangx /3600 *!dtor
            Tneangy = Tneangy /3600 *!dtor
            Tneangz = Tneangz /3600 *!dtor
        END
        'Radians': BEGIN 
        END
    ENDCASE
    neangx = strtrim(string(Tneangx * !radeg) ,2)
    neangy = strtrim(string(Tneangy * !radeg) ,2)
    neangz = strtrim(string(Tneangz * !radeg) ,2)

    WIDGET_CONTROL, WIDS.modelid, GET_VALUE=modelid
    modelid = strtrim(modelid,2)
    tempy = "cloud," + modelid + "," + neangx + "," + neangy + "," + neangz
    print,tempy
    tempy = execute(tempy[0])
endif else $
if ev.ID eq WIDS.wireframe then begin
    WIDGET_CONTROL, WIDS.neangx, GET_VALUE=neangx
    WIDGET_CONTROL, WIDS.neangy, GET_VALUE=neangy
    WIDGET_CONTROL, WIDS.neangz, GET_VALUE=neangz
    WIDGET_CONTROL, WIDS.neangunits, GET_VALUE=neangunits
    Tneangx = float(neangx)
    Tneangy = float(neangy)
    Tneangz = float(neangz)
    
    CASE neangunits OF
        'Degrees': BEGIN
            Tneangx = Tneangx *!dtor
            Tneangy = Tneangy *!dtor
            Tneangz = Tneangz *!dtor
        END
        'Minutes': BEGIN
            Tneangx = Tneangx /60 *!dtor
            Tneangy = Tneangy /60 *!dtor
            Tneangz = Tneangz /60 *!dtor
        END
        'Seconds': BEGIN
            Tneangx = Tneangx /3600 *!dtor
            Tneangy = Tneangy /3600 *!dtor
            Tneangz = Tneangz /3600 *!dtor
        END
        'Radians': BEGIN 
        END
    ENDCASE
    neangx = strtrim(string(Tneangx * !radeg) ,2)
    neangy = strtrim(string(Tneangy * !radeg) ,2)
    neangz = strtrim(string(Tneangz * !radeg) ,2)

    WIDGET_CONTROL, WIDS.modelid, GET_VALUE=modelid
    WIDGET_CONTROL, WIDS.modparam, GET_VALUE=modparam
    modelid = strtrim(modelid,2)
    tempy = "makewire," + modelid + ",neangx=" + neangx + $
      ",neangy=" + neangy + ",neangz=" + neangz + ",modparam=" + modparam
    print,tempy
    ;tempy = execute(tempy[0])
    if (modparam eq '0') OR (modparam eq '0.') OR (modparam eq '') then $
      makewire,(long(modelid))[0],neangx=(float(neangx))[0],neangy=(float(neangy))[0],$
      neangz=(float(neangz))[0] else $
      makewire,(long(modelid))[0],neangx=(float(neangx))[0],neangy=(float(neangy))[0],$
      neangz=(float(neangz))[0],modparam=COMMONMP
endif else $
if ev.ID eq WIDS.modelidmenu then begin ;click on modelid menu
    val = ev.value - 1
    if val gt 22 then val = val-1
    WIDGET_CONTROL, WIDS.modelid, $
      SET_VALUE=strtrim(string(val),2)
    SET_MODPARAM,val ;figure out parameters for this model
endif else $
if ev.ID eq WIDS.obsposmenu then begin
    CASE ev.value OF
        1: WIDGET_CONTROL, WIDS.obsposunits, SET_VALUE='Rsun'
        2: WIDGET_CONTROL, WIDS.obsposunits, SET_VALUE='AU'
    ENDCASE
endif else $
if ev.ID eq WIDS.neposmenu then begin
    CASE ev.value OF
        1: WIDGET_CONTROL, WIDS.neposunits, SET_VALUE='Rsun'
        2: WIDGET_CONTROL, WIDS.neposunits, SET_VALUE='AU'
    ENDCASE
endif else $
if ev.ID eq WIDS.losmenu then begin
    CASE ev.value OF 
        1: WIDGET_CONTROL, WIDS.losunits, SET_VALUE='Rsun'
        2: WIDGET_CONTROL, WIDS.losunits, SET_VALUE='AU'
    ENDCASE
endif else $
if ev.ID eq WIDS.fovpixmenu then begin
    CASE ev.value OF
        1: WIDGET_CONTROL, WIDS.fovpixunits, SET_VALUE='ArcSec/Pixel'
        2: WIDGET_CONTROL, WIDS.fovpixunits, SET_VALUE='ArcMin/Pixel'
        3: WIDGET_CONTROL, WIDS.fovpixunits, SET_VALUE='ArcDeg/Pixel'
        4: WIDGET_CONTROL, WIDS.fovpixunits, SET_VALUE='Rad/Pixel'
    ENDCASE
endif else $
if ev.ID eq WIDS.quickmenu then begin
    CASE ev.value OF
        1: WIDGET_CONTROL, WIDS.xs, SET_VALUE='64'     
        2: WIDGET_CONTROL, WIDS.xs, SET_VALUE='128'        
        3: WIDGET_CONTROL, WIDS.xs, SET_VALUE='256'        
        4: WIDGET_CONTROL, WIDS.xs, SET_VALUE='512'        
        5: WIDGET_CONTROL, WIDS.xs, SET_VALUE='1024'        
    ENDCASE
    CASE ev.value OF
        1: WIDGET_CONTROL, WIDS.ys, SET_VALUE='64'
        2: WIDGET_CONTROL, WIDS.ys, SET_VALUE='128'
        3: WIDGET_CONTROL, WIDS.ys, SET_VALUE='256'
        4: WIDGET_CONTROL, WIDS.ys, SET_VALUE='512'
        5: WIDGET_CONTROL, WIDS.ys, SET_VALUE='1024'
    ENDCASE
endif else $
if ev.ID eq WIDS.obsangmenu then begin
    CASE ev.value OF
        1 : WIDGET_CONTROL, WIDS.obsangunits, SET_VALUE='Degrees'
        2 : WIDGET_CONTROL, WIDS.obsangunits, SET_VALUE='Minutes'
        3 : WIDGET_CONTROL, WIDS.obsangunits, SET_VALUE='Seconds'
        4 : WIDGET_CONTROL, WIDS.obsangunits, SET_VALUE='Radians'
    ENDCASE
endif else $
if ev.ID eq WIDS.neangmenu then begin
    CASE ev.value OF
        1 : WIDGET_CONTROL, WIDS.neangunits, SET_VALUE='Degrees'
        2 : WIDGET_CONTROL, WIDS.neangunits, SET_VALUE='Minutes'
        3 : WIDGET_CONTROL, WIDS.neangunits, SET_VALUE='Seconds'
        4 : WIDGET_CONTROL, WIDS.neangunits, SET_VALUE='Radians'
    ENDCASE
ENDIF else GO_PRO, ev

END
;------------------
PRO OP, ev
common share1

help,ev, /STRUCTURES

END
;================================
PRO RTFRONTEND
common share1

WIDS = {WIDSNAME, $
        base: 0, base2: 0, xs: 0, ys: 0, $
        base_frame2: 0, base_frame3: 0, $
        base_movie: 0, $
        editObs:0, editNe:0, editMovie: 0, $
        fovpix: 0, $
        obsposx: 0, obsposy: 0, obsposz: 0, $
        obsangx: 0, obsangy: 0, obsangz: 0, $
        neposx: 0, neposy: 0, neposz: 0, $
        neangx: 0, neangy: 0, neangz: 0, $
        losnbp: 0, lstart: 0, lend: 0, $
        modelid: 0, modparam: 0, $
        save: 0, fakelasco: 0, $
        whichnum: 0, button_change_modparam: 0, $
        obsangmenu: 0L, obsangunits: 0, $
        neangmenu: 0L, neangunits: 0, $
        losmenu: 0L, losunits: 0, $
        temp: 0, OP: 0, quickmenu: 0, $
        fovpixmenu: 0, fovpixunits: 0, $
        FL: 0, modelidmenu: 0, $
        obsposmenu: 0, obsposunits: 0, $
        neposmenu: 0, neposunits: 0, $
        info: strarr(1000), $
        slider: 0, list: 0L, slider_label: 0, $
        wireframe: 0, label_increment: 0, $
        AX: 0, axisMenu: 0, numsteps: 0, increment:0, $
        inhibit:1, IW:0, from: 0, to: 0, rotations: 0, $
        C: 0, cimage: 0, $
        rebin: 0, cloud: 0, volume: 0, $
        mypath: '', button_print: 0, type_menu: 0, $
        text_file: 0, TEMPA: 0, TEMPB: 0, TEMPC: 0, $
        TEMPD: 0, TEMPE: 0, TEMPARR: intarr(8), $
        Anti_Vignetting: 0, AV:0, paramnum: 0,usersavepath:'' }

;COMMONMP=0
if 0 eq 1 then cloud ;these are for RESOLVE_ALL,/CONTINUE_ON_ERROR
if 0 eq 1 then dragger
if 0 eq 1 then y = crossp()

base = WIDGET_BASE(RESOURCE_NAME = 'base', COLUMN=1,  $
                   TITLE="Frontend - Raytrace WhiteLight",$
                  /FRAME, XPAD=20, YPAD=0, XOFFSET=100, YOFFSET=20, $
                   TLB_FRAME_ATTR=1, MBAR=barBase)
WIDS.base = base

wFileButton = WIDGET_BUTTON(barBase, VALUE= 'File', /MENU)

wSaveButton = WIDGET_BUTTON(wFileButton, SENSITIVE=0, $
                            VALUE='Save', UVALUE='SAVE')

wRestoreButton = WIDGET_BUTTON(wFileButton, SENSITIVE=0, $
                              VALUE='Restore', UVALUE='RESTORE')

wExitButton = WIDGET_BUTTON(wFileButton, EVENT_PRO='EXIT_PRO', $
                             VALUE='Exit', UVALUE='EXIT')

wEditButton = WIDGET_BUTTON(barBase, VALUE='Edit', /MENU)

;MODELID SELECT MENU
desc_modelid = replicate({ flags:0, name:'' }, 43)
desc_modelid.flags = intarr(43)
desc_modelid.name = indgen(43)

desc_modelid[0].flags = 1
desc_modelid[0].name = 'Select Model'
desc_modelid[1].flags = 1
desc_modelid[1].name = 'Models 1-20'
desc_modelid[2].name = ' 1. M.Guhathakurta Model'
desc_modelid[3].name = ' 2. Simple cylindrical structure'
desc_modelid[4].name = ' 3.'
desc_modelid[5].name = ' 4. Density Cube from Y.M. Interpolation'
desc_modelid[6].name = ' 5. Same as 4. without Trilinear Interpolation'
desc_modelid[7].name = ' 6. Fake Model of Streamer Leaf'
desc_modelid[8].name = ' 7. Slab Model of Stream Leaf'
desc_modelid[9].name = ' 8. Spherical symmetry of the Ne using Hayes Model'
desc_modelid[10].name = ' 9. Guhathakurta Radial and Vibert shape'
desc_modelid[11].name = '10. Slab Model: Haze Density Modeling'
desc_modelid[12].name = '11. Same as 10 but w/ User-Defined Current Sheet *'
desc_modelid[13].name = '12. Same as 10 but w/ simple neutral sheet'
desc_modelid[14].name = '13. Slab, Haze model, orthoradial...'
desc_modelid[15].name = '14. Slab, Orthoradial composite model...'
desc_modelid[16].name = '15. Same as 14 but...'
desc_modelid[17].name = '16. Cylinder model, Haze radial...'
desc_modelid[18].name = '17. Same as 16 but...'
desc_modelid[19].name = '18. Slab, Same as 14 but...'
desc_modelid[20].name = '19. Simple spherical symmetry with Satio model of density'
desc_modelid[21].name = '20. Spherical Symmetry, Satio density model, Polar density' 
desc_modelid[21].flags = 2
desc_modelid[22].name = 'Models 21-40'
desc_modelid[22].flags = 1
desc_modelid[23].name = '21. Same as 20 but with Equitorial density' 
desc_modelid[24].name = '22. Same as 14 but with different angular sector'
desc_modelid[25].name = '23. Same as 18 but with different angular sector'
desc_modelid[26].name = '24. Cylinder Model'
desc_modelid[27].name = '25. User-defined Data Cube of Ne density'
desc_modelid[28].name = '26. Same as 25 but with trilinear interpolation'
desc_modelid[29].name = '27. Spherical Shell Model'
desc_modelid[30].name = '28. Cylindrical Shell Model'
desc_modelid[31].name = '29. Bow Shock Model'
desc_modelid[32].name = '30. Streamer belt simulation w/ source surface field map'
desc_modelid[33].name = '31. Graduated Curved Cylindrical Shell Model'
desc_modelid[34].name = '32. Based on 30 with source surface field map'
desc_modelid[35].name = '33. Tube shell model'
desc_modelid[36].name = '34. Tube shell (frozen param.)'
desc_modelid[37].name = '35. Gibson and Low model'
desc_modelid[38].name = '36. NULL'
desc_modelid[39].name = '37. NULL'
desc_modelid[40].name = '38. NULL'
desc_modelid[41].name = '39. NULL'
desc_modelid[42].name = '40. NULL'


;WIDS.modelidmenu = CW_PDMENU(base_model,desc_modelid)
WIDS.modelidmenu = CW_PDMENU(wEditButton, desc_modelid, /MBAR)

wResetButton = WIDGET_BUTTON(wEditButton, SENSITIVE=0, $
                             value='Reset Parameters')


;cimage = WIDGET_DROPLIST(wEditButton, value=['Normal', 'C2 Image', 'C3 Image'], $
;                                title='mhg')

;wC2Button = WIDGET_BUTTON(wEditButton, VALUE='C2 Image', /CHECKED_MENU)

wViewButton = WIDGET_BUTTON(barBase, VALUE= 'View', /MENU)

WIDS.editObs = WIDGET_BUTTON(wViewButton, value='Observer Parameters', $
                             /CHECKED_MENU)
WIDS.editNe = WIDGET_BUTTON(wViewButton, value='Electron Dens. Parameters', $
                            /CHECKED_MENU)

WIDGET_CONTROL, WIDS.editNe, /SET_BUTTON

WIDS.editMovie = WIDGET_BUTTON(wViewButton, value='Make Movie Parameters', $
                            /CHECKED_MENU)

wClearButton = WIDGET_BUTTON(wViewButton, VALUE='Clear Windows', $
                             EVENT_PRO='CLEAR_WINDOWS')

w3dButton = WIDGET_BUTTON(barBase, VALUE= '3-D Modeling', /MENU)

WIDS.wireframe = WIDGET_BUTTON(w3dButton, value="Wireframe")

WIDS.cloud = WIDGET_BUTTON(w3dButton, value="Ne Cloud")

WIDS.volume = WIDGET_BUTTON(w3dButton, value="XVolume")


wHelpButton = WIDGET_BUTTON(barBase, /HELP, $
                            VALUE='Help', /MENU)

wManualButton = WIDGET_BUTTON(wHelpButton, SENSITIVE=0, $
                              VALUE='Manual', UVALUE='MANUAL')

wAxisButton = WIDGET_BUTTON(wHelpButton, EVENT_PRO='AXIS_PRO', $
                            VALUE='Axis Definition', UVALUE='AXIS')

wAboutButton = WIDGET_BUTTON(wHelpButton, EVENT_PRO='ABOUT_PRO', $
                             VALUE='About Dragger/RaytraceWL', UVALUE='ABOUT')

label_blank = WIDGET_LABEL(base, VALUE="")

; ---- set user's directory to save data
rtgetuserdir,usersavepath
WIDS.usersavepath=usersavepath
;WIDS.usersavepath=getenv('HOME')
;print,'User''s home directory : '+WIDS.usersavepath
; ---- check if saveraytrace directory exist. If not, create it
;if file_exist(WIDS.usersavepath+'/saveraytrace') eq 0 then begin
;    spawn,'mkdir '+WIDS.usersavepath+'/saveraytrace'
;    print,'Creating :'+WIDS.usersavepath+'/saveraytrace'
;endif
;WIDS.usersavepath=WIDS.usersavepath+'/saveraytrace'

which,'rtfrontend.pro',outfile=x
WIDS.mypath = STRMID(x,0,strlen(x)-12)
title_img = WIDGET_BUTTON(base, $
                          value=filepath(WIDS.mypath[0]+"title4.bmp", root_dir='/'),$
                          /bitmap, EVENT_PRO="ABOUT_PRO")

label_blank = WIDGET_LABEL(base, VALUE="", ysize=20)
base_frame1 = WIDGET_BASE(base, ROW=2, /FRAME)
base_imsize = WIDGET_BASE(base_frame1, /ROW)
label_imsize = WIDGET_LABEL(base_imsize, VALUE="Image Size:", /ALIGN_RIGHT)
label_xs = WIDGET_LABEL(base_imsize, VALUE="  X:", /ALIGN_RIGHT)
WIDS.xs = WIDGET_TEXT(base_imsize, /EDITABLE, VALUE='256')
label_ys = WIDGET_LABEL(base_imsize, VALUE="  Y:", /ALIGN_RIGHT)
WIDS.ys = WIDGET_TEXT(base_imsize, /EDITABLE, VALUE='256')

;QUICK SELECT MENU
desc_quick = replicate({ flags:0, name:'' }, 6)
desc_quick.flags = [1,0,0,0,0,0]
desc_quick.name = ['Quick Select Size', '64x64', '128x128', '256x256', '512x512', '1024x1024']
WIDS.quickmenu = CW_PDMENU(base_imsize,desc_quick)

;FOVPIX
base_fovpix = WIDGET_BASE(base_frame1, /ROW)
label_fovpix = WIDGET_LABEL(base_fovpix, VALUE="Pixel angle:    ", /ALIGN_RIGHT)
WIDS.fovpix = WIDGET_TEXT(base_fovpix, /EDITABLE, VALUE='1/32')

;FOVPIX UNITS
desc_fovpix = replicate({ flags:0, name:'' }, 5)
desc_fovpix.flags = [1,0,0,0,0]
desc_fovpix.name = ['In:', 'ArcSec/Pixel', 'ArcMin/Pixel', 'ArcDeg/Pixel', 'Rad/Pixel']
WIDS.fovpixmenu = CW_PDMENU(base_fovpix,desc_fovpix)
WIDS.fovpixunits = WIDGET_LABEL(base_fovpix, VALUE='ArcDeg/Pixel', XSIZE=75, /ALIGN_LEFT)

label_blank = WIDGET_LABEL(base, VALUE="", ysize=10)

WIDS.base_frame2 = WIDGET_BASE(base, ROW=2, /FRAME, MAP=0,xsize=1,ysize=1)
base_obspos = WIDGET_BASE(WIDS.base_frame2, /ROW)
label_obspos = WIDGET_LABEL(base_obspos, VALUE="Observer Position:  ", /ALIGN_RIGHT)
label_obsposx = WIDGET_LABEL(base_obspos, VALUE="  X:", /ALIGN_RIGHT)
WIDS.obsposx = WIDGET_TEXT(base_obspos, /EDITABLE, VALUE='0', XSIZE=10)
label_obsposy = WIDGET_LABEL(base_obspos, VALUE="  Y:", /ALIGN_RIGHT)
WIDS.obsposy = WIDGET_TEXT(base_obspos, /EDITABLE, VALUE='0', XSIZE=10)
label_obsposz = WIDGET_LABEL(base_obspos, VALUE="  Z:", /ALIGN_RIGHT)
WIDS.obsposz = WIDGET_TEXT(base_obspos, /EDITABLE, VALUE='-214', XSIZE=10)

;OBSPOS UNITS
desc_obspos = replicate({ flags:0, name:'' }, 3)
desc_obspos.flags = [1,0,0]
desc_obspos.name = ['In:', 'Rsun', 'AU']
WIDS.obsposmenu = CW_PDMENU(base_obspos,desc_obspos)
WIDS.obsposunits = WIDGET_LABEL(base_obspos, VALUE='Rsun', XSIZE=35, /ALIGN_LEFT)

base_obsang = WIDGET_BASE(WIDS.base_frame2, /ROW)
label_obsang = WIDGET_LABEL(base_obsang, VALUE="Obsrvr. Orientation:", /ALIGN_RIGHT)

desc_obs = replicate({ flags:0, name:'' }, 5)
desc_obs.flags = [1,0,0,0,0]
desc_obs.name = ['In:', 'Degrees', 'Minutes', 'Seconds', 'Radians']

label_obsangx = WIDGET_LABEL(base_obsang, VALUE="  X:", /ALIGN_RIGHT)
WIDS.obsangx = WIDGET_TEXT(base_obsang, /EDITABLE, VALUE='0', XSIZE=10)
label_obsangy = WIDGET_LABEL(base_obsang, VALUE="  Y:", /ALIGN_RIGHT)
WIDS.obsangy = WIDGET_TEXT(base_obsang, /EDITABLE, VALUE='0', XSIZE=10)
label_obsangz = WIDGET_LABEL(base_obsang, VALUE="  Z:", /ALIGN_RIGHT)
WIDS.obsangz = WIDGET_TEXT(base_obsang, /EDITABLE, VALUE='0', XSIZE=10)
WIDS.obsangmenu = CW_PDMENU(base_obsang,desc_obs)

WIDS.obsangunits = WIDGET_LABEL(base_obsang, VALUE='Degrees', XSIZE=45, /ALIGN_LEFT)

label_blank = WIDGET_LABEL(base, VALUE="", ysize=10)

WIDS.base_frame3 = WIDGET_BASE(base, ROW=2, /FRAME)
base_nepos = WIDGET_BASE(WIDS.base_frame3, /ROW)
label_nepos = WIDGET_LABEL(base_nepos, VALUE="Ne Position:   ", /ALIGN_RIGHT)
label_neposx = WIDGET_LABEL(base_nepos, VALUE="  X:", /ALIGN_RIGHT)
WIDS.neposx = WIDGET_TEXT(base_nepos, /EDITABLE, VALUE='0', XSIZE=10)
label_neposy = WIDGET_LABEL(base_nepos, VALUE="  Y:", /ALIGN_RIGHT)
WIDS.neposy = WIDGET_TEXT(base_nepos, /EDITABLE, VALUE='0', XSIZE=10)
label_neposz = WIDGET_LABEL(base_nepos, VALUE="  Z:", /ALIGN_RIGHT)
WIDS.neposz = WIDGET_TEXT(base_nepos, /EDITABLE, VALUE='0', XSIZE=10)

;NEPOS UNITS
desc_nepos = replicate({ flags:0, name:'' }, 3)
desc_nepos.flags = [1,0,0]
desc_nepos.name = ['In:', 'Rsun', 'AU']
WIDS.neposmenu = CW_PDMENU(base_nepos,desc_nepos)
WIDS.neposunits = WIDGET_LABEL(base_nepos, VALUE='Rsun', XSIZE=35, /ALIGN_LEFT)

base_neang = WIDGET_BASE(WIDS.base_frame3, /ROW)
label_neang = WIDGET_LABEL(base_neang, VALUE="Ne Orientation:", /ALIGN_RIGHT)
desc_ne = replicate({ flags:0, name:'' }, 5)
desc_ne.flags = [1,0,0,0,0]
desc_ne.name = ['In:', 'Degrees', 'Minutes', 'Seconds', 'Radians']

label_neangx = WIDGET_LABEL(base_neang, VALUE="  X:", /ALIGN_RIGHT)
WIDS.neangx = WIDGET_TEXT(base_neang, /EDITABLE, VALUE='0', XSIZE=10)
label_neangy = WIDGET_LABEL(base_neang, VALUE="  Y:", /ALIGN_RIGHT)
WIDS.neangy = WIDGET_TEXT(base_neang, /EDITABLE, VALUE='0', XSIZE=10)
label_neangz = WIDGET_LABEL(base_neang, VALUE="  Z:", /ALIGN_RIGHT)
WIDS.neangz = WIDGET_TEXT(base_neang, /EDITABLE, VALUE='0', XSIZE=10)
WIDS.neangmenu = CW_PDMENU(base_neang,desc_ne)
WIDS.neangunits = WIDGET_LABEL(base_neang, VALUE='Degrees', XSIZE=45, /ALIGN_LEFT)

WIDS.base_movie = WIDGET_BASE(base, /ROW, /FRAME, MAP=0, XSIZE=1, YSIZE=1)
WIDS.axisMenu = WIDGET_DROPLIST(WIDS.base_movie, $
                                value=['X Axis', 'Y Axis', 'Z Axis', 'LOS', 'Param'], $
                                title='Animate: ')
label_numsteps = WIDGET_LABEL(WIDS.base_movie, VALUE="  # Of Increments:  ")
WIDS.numsteps = WIDGET_TEXT(WIDS.base_movie, /EDITABLE, VALUE='0', XSIZE=10)
WIDS.label_increment = WIDGET_LABEL(WIDS.base_movie, VALUE="   Degrees/Increment:  ")
WIDS.increment = WIDGET_TEXT(WIDS.base_movie, /EDITABLE, VALUE='5', XSIZE=10)

;label_blank = WIDGET_LABEL(WIDS.base_movie, VALUE="", ysize=10)

base_los = WIDGET_BASE(base, /ROW, /FRAME)
label_losrange = WIDGET_LABEL(base_los, VALUE="LOS Range:", /ALIGN_RIGHT)
label_lstart = WIDGET_LABEL(base_los, VALUE=" Start:", /ALIGN_RIGHT)
WIDS.lstart = WIDGET_TEXT(base_los, /EDITABLE, VALUE='-20', XSIZE=10)
label_lend = WIDGET_LABEL(base_los, VALUE=" End:", /ALIGN_RIGHT)
WIDS.lend = WIDGET_TEXT(base_los, /EDITABLE, VALUE='20', XSIZE=10)

;LOS UNITS
desc_los = replicate({ flags:0, name:'' }, 3)
desc_los.flags = [1,0,0]
desc_los.name = ['In:', 'Rsun', 'AU']
WIDS.losmenu = CW_PDMENU(base_los,desc_los)
WIDS.losunits = WIDGET_LABEL(base_los, VALUE='Rsun', XSIZE=35, /ALIGN_LEFT)

label_losnbp = WIDGET_LABEL(base_los, VALUE=" Integration steps:", /ALIGN_RIGHT)
WIDS.losnbp = WIDGET_TEXT(base_los, /EDITABLE, VALUE='64', XSIZE=10)

label_blank = WIDGET_LABEL(base, VALUE="", ysize=10)

base_frame4top = WIDGET_BASE(base, COLUMN=2, FRAME=5, xpad=0, ypad=10)

base_frame4 = WIDGET_BASE(base_frame4top, ROW=3);, FRAME=5)
base_model = WIDGET_BASE(base_frame4, /ROW)

label_blank = WIDGET_LABEL(base, VALUE="")
;MODELID SELECT MENU

base_modparam = WIDGET_BASE(base_frame4, /ROW)
label_modparam = WIDGET_LABEL(base_modparam, VALUE="Parameters: ", /ALIGN_RIGHT)
WIDS.modparam = WIDGET_TEXT(base_modparam, VALUE='0.', XSIZE=50)
WIDS.button_change_modparam = WIDGET_BUTTON(base_modparam, value="Change")
label_blank = WIDGET_LABEL(base, VALUE="")

WIDS.FL = 0
;WIDS.fakelasco = CW_BGROUP(base_misc, ['Use Fake LASCO Header?'], /NONEXCLUSIVE)

label_blank = WIDGET_LABEL(base_frame4top, value='', YSIZE=1)
label_blank = WIDGET_LABEL(base_frame4top, value='Image Type', YSIZE=20)

base_imagetype = WIDGET_BASE(base_frame4top, /FRAME, /COLUMN)

WIDS.C = 0
;WIDS.cimage = CW_BGROUP(base_imagetype, ['Normal', 'C2 Image', 'C3 Image'],$
;                        /COLUMN, /EXCLUSIVE, /NO_RELEASE, SET_VALUE=0)

WIDS.cimage = WIDGET_DROPLIST(base_imagetype, value=['Normal', 'C2 Image', 'C3 Image'], $
                                title='')
WIDS.OP = 0

label_blank = WIDGET_LABEL(base, VALUE="")

base_buttmain = WIDGET_BASE(base, COLUMN=3)

axis_img = WIDGET_BUTTON(base_buttmain, $
                         value=filepath(WIDS.mypath[0]+"axis2.bmp", root_dir='/'), $
                         /bitmap, EVENT_PRO="AXIS_PRO")

label_blank = WIDGET_LABEL(base_buttmain, $
                           VALUE="                                    ", YSIZE=30)

base_butt = WIDGET_BASE(base_buttmain, XPAD=40, YPAD=10, FRAME=5, COLUMN=1, /ALIGN_RIGHT)
WIDS.IW = 1
WIDS.inhibit = CW_BGROUP(base_butt, ['Inhibit Windows?'], /NONEXCLUSIVE,set_value=1)

WIDS.button_print = WIDGET_BUTTON(base_butt, value="Selector", UVALUE='SelectorBut', $
                             EVENT_PRO="SELECTOR_PRO", YSIZE=30, sensitive=0)

base_modelid = WIDGET_BASE(base_butt, /ROW, /ALIGN_CENTER)

label_modelid = WIDGET_LABEL(base_modelid, VALUE="Model ID#:", /ALIGN_CENTER)
WIDS.modelid = WIDGET_LABEL(base_modelid, VALUE='14')

button_go = WIDGET_BUTTON(base_butt, value="   -- GO! --   ", UVALUE='go',$
                             EVENT_PRO="GO_PRO", YSIZE=40)

;button_exit = WIDGET_BUTTON(base_butt, value='Exit', UVALUE='EXIT', EVENT_PRO="EXIT_PRO")

WIDGET_CONTROL, base, /REALIZE
XMANAGER, 'Widget', base, /NO_BLOCK

END
