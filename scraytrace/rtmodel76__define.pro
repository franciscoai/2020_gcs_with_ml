; $Id: rtmodel76__define.pro,v 1.1 2013/01/29 19:55:55 thernis Exp $


; ---- compute the cloud
function rtmodel76::calcCloud, swidth_rad, Rma, r_out, r_in, hgt, nbvertshell, nbvertaxis

; ---- the torus axis is in the X, Z plane

; ---- outer shell circle
beta = lgen(nbvertshell, 0., 360. - 360. / nbvertshell) * !dtor
circle = [  [replicate(0., nbvertshell)],$
            [r_out * cos(beta)],$
            [r_out * sin(beta)]]

alpha = lgen(nbvertaxis, -swidth_rad, swidth_rad)

poutershell = fltarr(long(nbvertaxis) * long(nbvertshell),3)


for i=0l,nbvertaxis-1 do begin
    poutershell[(i*long(nbvertshell)):((i+1)*long(nbvertshell)-1),*] = $
        [   [(Rma + r_out * sin(beta)) * sin(alpha[i])],$
            [r_out * cos(beta)],$
            [(Rma + r_out * sin(beta)) * cos(alpha[i]) + hgt - Rma] ]
    
endfor


return,transpose(poutershell)
end


; ------ compute the cylinder inner radius
function rtmodel76::calcR_in,r_out
r_in = (r_out - 0.5) > 0.
return,r_in
end



; ------ update the cloud
pro rtmodel76::updateCloud

r_out = self->getWidgetValue(self.unamemodelid+'wsli.r_out')
swidth = self->getWidgetValue(self.unamemodelid+'wsli.swidth')
hgt = self->getWidgetValue(self.unamemodelid+'wsli.hgt')
rma = self->getWidgetValue(self.unamemodelid+'wsli.rma')    ; major radius

nbvertshell = float((self->getWidgetValue(self.unamemodelid+'wtex.nbvertshell'))[0])
nbvertaxis = float((self->getWidgetValue(self.unamemodelid+'wtex.nbvertaxis'))[0])

r_in = self->calcR_in(r_out)

oc = self->calcCloud(swidth * !dtor, rma, r_out, r_in, hgt, nbvertshell, nbvertaxis)

; -- add last point at the surface of the sun, pointing in the direction of the model nose
szoc = size(oc,/dim)
oc = reform([reform(oc,n_elements(oc)),0,0,1.01],3,szoc[1]+1)

self.cloud=ptr_new(oc)

return
end

; ------ update neang
pro rtmodel76::updateNeang
lon=self->getLonSliderinCarrCoord()
lat=self->getWidgetValue(self.unamemodelid+'wsli.lat')
rot=self->getWidgetValue(self.unamemodelid+'wsli.rot')

self.neang=[lon,lat,rot]*!dtor
return
end


; ------ update nepos (overload)
pro rtmodel76::updateNepos

self->rtmodelbase::updateNepos

hgt = self->getWidgetValue(self.unamemodelid+'wsli.hgt')

self.nerotcntr = [0.,0,hgt]

return
end


function rtmodel76::getModParam

r_out = self->getWidgetValue(self.unamemodelid+'wsli.r_out')
swidth = self->getWidgetValue(self.unamemodelid+'wsli.swidth')
hgt = self->getWidgetValue(self.unamemodelid+'wsli.hgt')
rma = self->getWidgetValue(self.unamemodelid+'wsli.rma')    ; major radius

r_in = self->calcR_in(r_out)

mp=[swidth * !dtor, rma, r_out, r_in, hgt, 1000.]
self.pmodparam=ptr_new(mp)

return,self.pmodparam
end



; ------ event handler
function rtmodel76::eventHandler,ev,uname
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
        flagupdatecloud=(slidername eq 'r_out' || slidername eq 'swidth' || slidername eq 'hgt' || slidername eq 'rma')
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
    self->updateNepos
endif
return,flagout
end


; ------ build the model widget
function rtmodel76::buildWidget,wparent

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
wslider_r_out = cw_fslider(wtab_position,minimum=0.001,maximum=10.,$
    value=2.,$ 
    title='Outer Radius',uname=self.unamemodelid+'wsli.r_out',/drag,/edit,xsize=360)
self.guiunamelist.add,'wsli.r_out'

; -- semi angular width
wslider_swidth = cw_fslider(wtab_position,minimum=0.,maximum=180.,$
    value=30.,$ 
    title='Semi Angular Width',uname=self.unamemodelid+'wsli.swidth',/drag,/edit,xsize=360)
self.guiunamelist.add,'wsli.swidth'

; -- major radius
wslider_R_maj = cw_fslider(wtab_position,minimum=0.,maximum=10.,$
    value=3.,$ 
    title='Major Radius',uname=self.unamemodelid+'wsli.rma',/drag,/edit,xsize=360)
self.guiunamelist.add,'wsli.rma'



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
function rtmodel76::INIT,instanceid,_extra=extra

foo=self->rtmodelbase::init('Torus',76,instanceid,_extra=extra)

return,1
end


; ------ class structure definition
pro rtmodel76__define
    struct={rtmodel76, inherits rtmodelbase}
return
end

; $Log: rtmodel76__define.pro,v $
; Revision 1.1  2013/01/29 19:55:55  thernis
; First commit.
;
;
