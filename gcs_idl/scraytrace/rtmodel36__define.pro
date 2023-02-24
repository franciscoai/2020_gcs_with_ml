; $Id: rtmodel36__define.pro,v 1.2 2013/08/12 21:43:30 avourlid Exp $

; ------ update the cloud
pro rtmodel36::updateCloud

e=self->getWidgetValue(self.unamemodelid+'wsli.e')
k=self->getWidgetValue(self.unamemodelid+'wsli.k')
h=self->getWidgetValue(self.unamemodelid+'wsli.hgt')
nr=float((self->getWidgetValue(self.unamemodelid+'wtex.nr'))[0])
nth=float((self->getWidgetValue(self.unamemodelid+'wtex.nth'))[0])

spheroidshock_cloud,e,k,h, xx,yy,zz, nr = nr, nth = nth

; -- reform and add point at the surface of the Sun, in the direction of the model nose
oc=transpose([[xx,0],[yy,0],[zz,1.01]])

self.cloud=ptr_new(oc)

return
end


; ------ update neang
pro rtmodel36::updateNeang
lon=self->getLonSliderinCarrCoord()
lat=self->getWidgetValue(self.unamemodelid+'wsli.lat')
rot=self->getWidgetValue(self.unamemodelid+'wsli.rot')

self.neang=[lon,lat,rot]*!dtor
return
end



; ---- compute the modparam when calling white light renderer
function rtmodel36::getModParam

e=self->getWidgetValue(self.unamemodelid+'wsli.e')
k=self->getWidgetValue(self.unamemodelid+'wsli.k')
h=self->getWidgetValue(self.unamemodelid+'wsli.hgt')

thickness=0.2
nel=1e5

mp=[e,k,h,thickness,nel]

self.pmodparam=ptr_new(mp)

return,self.pmodparam
end





; ------ model 29 event handler
function rtmodel36::eventHandler,ev,uname
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
		slidername=strmid(uname,9,3)
		flagupdatecloud=(slidername eq 'e' || slidername eq 'k' || slidername eq 'hgt')
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





function rtmodel36::buildWidget,wparent

self.wbase = widget_base(wparent,/column,$
    title=strtrim(self.instanceid,2)+':'+self.modelname,uname=self.unamemodelid+'base')

wtab = widget_tab(self.wbase,/align_left)

; ---- show current wireframe color of the model
self->rtmodelbase::buildCloudColorBox

wtab_position = widget_base(wtab,/column,title='Position')

; ---- build positioning sliders common to all model
self->rtmodelbase::buildPositionGUI,wtab_position,self.unamemodelid

; ---- build shape sliders specific to model 36
wslider_e = cw_fslider(wtab_position,minimum=-0.99,maximum=0.99,$
    value=0.0,$ 
    title='e',uname=self.unamemodelid+'wsli.e',/drag,/edit,scroll=1.,xsize=360)
self.guiunamelist.add,'wsli.e'


wslider_k = cw_fslider(wtab_position,minimum=0.01,maximum=2.,$
    value=1,$ 
    title='kappa',uname=self.unamemodelid+'wsli.k',/drag,/edit,scroll=1.,xsize=360)
self.guiunamelist.add,'wsli.k'

; -- Height slider
self->rtmodelbase::buildHeightGUI,wtab_position,self.unamemodelid,$
    heightinit = 2.


; ---- build cloud parameters specific to model 36
wtab_cloud = widget_base(wtab,/column,title='Cloud')

wlabel_nr = widget_label(wtab_cloud,$
    value='# Radial Elements',uname=self.unamemodelid+'wlab.nr')
wtext_nr = widget_text(wtab_cloud,/editable,$
    value='20',$
    uname=self.unamemodelid+'wtex.nr')
self.guiunamelist.add,'wtex.nr'

wlabel_nth = widget_label(wtab_cloud,value='# Azimuthal Elements',uname=self.unamemodelid+'wlab.nth')
wtext_nth = widget_text(wtab_cloud,/editable,$
    value='100',$
    uname=self.unamemodelid+'wtex.nth')
self.guiunamelist.add,'wtex.nth'

self->rtmodelbase::buildChangeCloudColor,wtab_cloud,self.unamemodelid

wlabel_msg1 = widget_label(wtab_cloud,$
    value='Press ENTER when done',uname=self.unamemodelid+'wlab.msg1')

wbutton_wireonoff = widget_button(wtab_cloud,$
    value='Wire Off',uname=self.unamemodelid+'wbut.wireonoff')

self.wireonoff = 1b

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


; ---- constructor
function rtmodel36::INIT,instanceid,_extra=extra;,reset=reset

foo=self->rtmodelbase::init('Spheroid Shock',36,instanceid,_extra=extra)

return,1
end

pro rtmodel36__define
	struct={rtmodel36,inherits rtmodelbase}
return
end

; $Log: rtmodel36__define.pro,v $
; Revision 1.2  2013/08/12 21:43:30  avourlid
; *** empty log message ***
;
; Revision 1.1  2013/06/18 16:47:22  avourlid
; added spheroid model (created by O. Olmedo) option to rtcloudwidget
; updated rtmodellist__define
;
; Revision 1.7  2012/03/12 13:57:53  thernis
; Add a cloud point at the surface of the sun, showing the direction of propagation.
;
; Revision 1.6  2011-08-10 17:31:39  thernis
; Insert CVS log
;
