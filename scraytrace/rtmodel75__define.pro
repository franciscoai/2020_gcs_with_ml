; $Id: rtmodel75__define.pro,v 1.2 2012/03/12 20:36:52 thernis Exp $


; ---- compute the cloud
function rtmodel75::calcCloud, r_out, r_in, swidth, nbvertshell, nbvertaxis, hgt

; ---- axis: X is the cylinder axis 
paxis = fltarr(3,nbvertaxis)
paxis[0,*] = lgen(nbvertaxis,-swidth,swidth)

; ---- outer and inner shell
poutershell = fltarr(3,nbvertaxis * nbvertshell)
; pinnershell = fltarr(3,nbvertaxis * nbvertshell)

; -- circle
angle = lgen(nbvertshell, 0., 360. -360. / nbvertshell)*!dtor
circouty = r_out * cos(angle)
circoutz = r_out * sin(angle)

; circiny = r_in * cos(angle)
; circinz = r_in * sin(angle)

for i=0l,nbvertshell-1 do begin
    poutershell[*,(i*nbvertaxis):((i+1)*nbvertaxis-1)] = transpose([    [reform(paxis[0,*])], $
                                                                        [replicate(circouty[i],nbvertaxis)], $
                                                                        [replicate(circoutz[i]+hgt,nbvertaxis)]])


;     pinnershell[*,(i*nbvertaxis):((i+1)*nbvertaxis-1)] = transpose([[reform(paxis[0,*])], [replicate(circiny[i],nbvertaxis)],  [replicate(circinz[i]+hgt,nbvertaxis)]])
endfor

; return,reform([ reform(poutershell,n_elements(poutershell)), reform(pinnershell,n_elements(pinnershell)) ], 3, 2 * nbvertaxis * nbvertshell)
return,poutershell
end


; ------ compute the cylinder inner radius
function rtmodel75::calcR_in,r_out
r_in = (r_out - 0.5) > 0.
return,r_in
end



; ------ update the cloud
pro rtmodel75::updateCloud

r_out = self->getWidgetValue(self.unamemodelid+'wsli.r_out')
; r_in = self->getWidgetValue(self.unamemodelid+'wsli.r_in')
swidth = self->getWidgetValue(self.unamemodelid+'wsli.swidth')
hgt = self->getWidgetValue(self.unamemodelid+'wsli.hgt')

nbvertshell = float((self->getWidgetValue(self.unamemodelid+'wtex.nbvertshell'))[0])
nbvertaxis = float((self->getWidgetValue(self.unamemodelid+'wtex.nbvertaxis'))[0])

r_in = self->calcR_in(r_out)

oc = self->calcCloud(r_out, r_in, swidth, nbvertshell, nbvertaxis, hgt)

; -- add last point at the surface of the sun, pointing in the direction of the model nose
szoc = size(oc,/dim)
oc = reform([reform(oc,n_elements(oc)),0,0,1.01],3,szoc[1]+1)

self.cloud=ptr_new(oc)

return
end

; ------ update neang
pro rtmodel75::updateNeang
lon=self->getLonSliderinCarrCoord()
lat=self->getWidgetValue(self.unamemodelid+'wsli.lat')
rot=self->getWidgetValue(self.unamemodelid+'wsli.rot')

self.neang=[lon,lat,rot]*!dtor
return
end


; ------ update nepos
pro rtmodel75::updateNepos

self->rtmodelbase::updateNepos

hgt = self->getWidgetValue(self.unamemodelid+'wsli.hgt')

self.nerotcntr = [0.,0,hgt]

return
end



function rtmodel75::getModParam

r_out = self->getWidgetValue(self.unamemodelid+'wsli.r_out')
; r_in = self->getWidgetValue(self.unamemodelid+'wsli.r_in')
swidth = self->getWidgetValue(self.unamemodelid+'wsli.swidth')
hgt = self->getWidgetValue(self.unamemodelid+'wsli.hgt')

r_in = self->calcR_in(r_out)

mp=[r_out,r_in,swidth,hgt,1000.]
; mp=[r_out,r_in,swidth,1000.]

self.pmodparam=ptr_new(mp)

return,self.pmodparam
end



; ------ event handler
function rtmodel75::eventHandler,ev,uname
if strmid(uname,0,4) ne self.unamemodelid then return,0b

foo=self->rtmodelbase::eventHandler(ev,uname)
if foo ne 0 then return,foo

widgettype=strmid(uname,4,4)
flagupdatecloud=0b
flagupdateneang=0b
; --flagout :  b1: refresh only , b0 : recalcViews
flagout='00'b
case widgettype of
    'wsli' : begin
        slidername=strmid(uname,9)
;         print,'slidername : ',slidername
        flagupdatecloud=(slidername eq 'r_out' || slidername eq 'swidth' || slidername eq 'hgt')
        flagupdateneang=1b
        flagout='01'b
    end
    'wtex' : begin
        flagupdatecloud=1b
        flagupdateneang=0b
        flagout='01'b
    end
    'wbut' : begin
        slidername=strmid(uname,9)
        ; -- wire on/off button
        if slidername eq 'wireonoff' then begin
            ; -- get value
            widget_control,ev.id,get_value=val
            ; -- switch value
            self.wireonoff=val eq 'Wire On'
            widget_control,ev.id,set_value=(self.wireonoff ? 'Wire Off' : 'Wire On')
            flagout='11'b
        endif

    end
endcase

; ---- recompute cloud in case of
if flagupdatecloud then self->updateCloud

; ---- recompute neang
if flagupdateneang then begin
    self->updateNeang
    self->rtmodel75::updateNepos
endif
return,flagout
end


; ------ build the model widget
function rtmodel75::buildWidget,wparent

self.wbase = widget_base(wparent,/column,$
    title=strtrim(self.instanceid,2)+':'+self.modelname,uname=self.unamemodelid+'base')
wtab = widget_tab(self.wbase,/align_left)

; ---- show current wireframe color of the model
self->rtmodelbase::buildCloudColorBox


; ---- build sliders common to all models
wtab_position = widget_base(wtab,/column,title='Position')
self->rtmodelbase::buildPositionGUI,wtab_position,self.unamemodelid

; ---- build model specific sliders
; -- Height slider
self->rtmodelbase::buildHeightGUI,wtab_position,self.unamemodelid,$
    heightinit = 10.

; -- outer radius
wslider_r_out = cw_fslider(wtab_position,minimum=0.01,maximum=10.,$
    value=2.,$ 
    title='Outer Radius',uname=self.unamemodelid+'wsli.r_out',/drag,/edit,xsize=360)
self.guiunamelist.add,'wsli.r_out'

; -- inner radius
; wslider_r_in = cw_fslider(wtab_position,minimum=0.,maximum=10.,$
;     value=1.5,$ 
;     title='Inner Radius',uname=self.unamemodelid+'wsli.r_in',/drag,/edit,xsize=360)
; self.guiunamelist.add,'wsli.r_in'

; -- semi width
wslider_swidth = cw_fslider(wtab_position,minimum=0.,maximum=10.,$
    value=3.,$ 
    title='Semi-Width',uname=self.unamemodelid+'wsli.swidth',/drag,/edit,xsize=360)
self.guiunamelist.add,'wsli.swidth'



; ---- model specific cloud parameters
wtab_cloud = widget_base(wtab,/column,title='Cloud')

wlabel_nbvertaxis = widget_label(wtab_cloud,value='# Points Axis',$
    uname=self.unamemodelid+'wlab.nbvertaxis')
wtext_nbvertaxis = widget_text(wtab_cloud,/editable,$
    value='50',$
    uname=self.unamemodelid+'wtex.nbvertaxis')
self.guiunamelist.add,'wtex.nbvertaxis'


wlabel_nbvertshell = widget_label(wtab_cloud,value='# Points Shell',$
    uname=self.unamemodelid+'wlab.nbvertshell')
wtext_nbvertshell = widget_text(wtab_cloud,/editable,$
    value='20',$
    uname=self.unamemodelid+'wtex.nbvertshell')
self.guiunamelist.add,'wtex.nbvertshell'


; wlabel_nbvertaxisp = widget_label(wtab_cloud,$
;     value='# Points Leg Axis',uname=self.unamemodelid+'wlab.nbvertaxisp')
; wtext_nbvertaxisp = widget_text(wtab_cloud,/editable,$
;     value='10',$ 
;     uname=self.unamemodelid+'wtex.nbvertaxisp')
; self.guiunamelist.add,'wtex.nbvertaxisp'


self->rtmodelbase::buildChangeCloudColor,wtab_cloud,self.unamemodelid


wlabel_msg1 = widget_label(wtab_cloud,$
    value='Press ENTER when done',uname=self.unamemodelid+'wlab.msg1')

wbutton_wireonoff = widget_button(wtab_cloud,value='Wire Off',$
    uname=self.unamemodelid+'wbut.wireonoff')
self.wireonoff=1b



; ---- build common Ne shift sliders
wtab_neshift = widget_base(wtab,/column,title='Ne Shift')
self->rtmodelbase::buildNeShift,wtab_neshift,self.unamemodelid

; ---- build delete button
self->rtmodelbase::buildDeleteButton


self->updateCloud
self->updateNeang

self.allowraytrace=1b

return,self.wbase
end

; ------ constructor
function rtmodel75::INIT,instanceid,_extra=extra

foo=self->rtmodelbase::init('Cylinder',75,instanceid,_extra=extra)

return,1
end


; ------ class structure definition
pro rtmodel75__define
    struct={rtmodel75, inherits rtmodelbase}
return
end

; $Log: rtmodel75__define.pro,v $
; Revision 1.2  2012/03/12 20:36:52  thernis
; Deal with rotation centered on the cylinder
;
; Revision 1.1  2012-03-12 13:55:20  thernis
; First commit of the Cylinder model
;
;