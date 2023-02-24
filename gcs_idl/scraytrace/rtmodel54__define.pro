; $Id: rtmodel54__define.pro,v 1.7 2012/03/12 13:57:37 thernis Exp $

; ------ update the cloud
pro rtmodel54::updateCloud

han=!dtor*self->getWidgetValue(self.unamemodelid+'wsli.han')
hgt=self->getWidgetValue(self.unamemodelid+'wsli.hgt')
rat=self->getWidgetValue(self.unamemodelid+'wsli.rat')
nbvertaxisp=float((self->getWidgetValue(self.unamemodelid+'wtex.nbvertaxisp'))[0])
nbvertcirup=float((self->getWidgetValue(self.unamemodelid+'wtex.nbvertcirup'))[0])
nbvertshell=float((self->getWidgetValue(self.unamemodelid+'wtex.nbvertshell'))[0])

oc=cmecloud(han,hgt,nbvertaxisp,nbvertcirup,rat,nbvertshell,/distjuncisleadingedge)


; -- add last point at the surface of the sun, pointing in the direction of the model nose
szoc = size(oc,/dim)
oc = reform([reform(oc,n_elements(oc)),0,0,1.01],3,szoc[1]+1)

self.cloud=ptr_new(oc)

return
end

; ------ update neang
pro rtmodel54::updateNeang
lon=self->getLonSliderinCarrCoord()
lat=self->getWidgetValue(self.unamemodelid+'wsli.lat')
rot=self->getWidgetValue(self.unamemodelid+'wsli.rot')

self.neang=[lon,lat,rot]*!dtor
return
end


function rtmodel54::getModParam

han=!dtor*self->getWidgetValue(self.unamemodelid+'wsli.han')
rat=self->getWidgetValue(self.unamemodelid+'wsli.rat')
hgt=self->getWidgetValue(self.unamemodelid+'wsli.hgt')
hgtfeet=hgt*(1.-rat)*cos(han)/(1.+sin(han))

mp=[1.5,han,hgtfeet,rat,1e6,0.,0.,0.,0.1,0.1]
self.pmodparam=ptr_new(mp)

return,self.pmodparam
end



; ------ event handler
function rtmodel54::eventHandler,ev,uname
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
		flagupdatecloud=(slidername eq 'han' || slidername eq 'hgt' || slidername eq 'rat')
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
function rtmodel54::buildWidget,wparent

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

; -- ratio
wslider_rat = cw_fslider(wtab_position,minimum=0.001,maximum=1.,$
    value=0.4,$ 
    title='Ratio',uname=self.unamemodelid+'wsli.rat',/drag,/edit,xsize=360)
self.guiunamelist.add,'wsli.rat'

; -- half angle
wslider_han = cw_fslider(wtab_position,minimum=0.,maximum=90.,$
    value=20.,$ 
    title='Half Angle',uname=self.unamemodelid+'wsli.han',/drag,/edit,scroll=0.5,xsize=360)
self.guiunamelist.add,'wsli.han'

; ---- model specific cloud parameters
wtab_cloud = widget_base(wtab,/column,title='Cloud')

wlabel_nbvertshell = widget_label(wtab_cloud,value='# Points Axis',$
    uname=self.unamemodelid+'wlab.nbvertshell')

wtext_nbvertshell = widget_text(wtab_cloud,/editable,$
    value='50',$
    uname=self.unamemodelid+'wtex.nbvertshell')
self.guiunamelist.add,'wtex.nbvertshell'


wlabel_nbvertcirup = widget_label(wtab_cloud,value='# Points Shell',$
    uname=self.unamemodelid+'wlab.nbvertcirup')
wtext_nbvertcirup = widget_text(wtab_cloud,/editable,$
    value='20',$
    uname=self.unamemodelid+'wtex.nbvertcirup')
self.guiunamelist.add,'wtex.nbvertcirup'


wlabel_nbvertaxisp = widget_label(wtab_cloud,$
    value='# Points Leg Axis',uname=self.unamemodelid+'wlab.nbvertaxisp')
wtext_nbvertaxisp = widget_text(wtab_cloud,/editable,$
    value='10',$ 
    uname=self.unamemodelid+'wtex.nbvertaxisp')
self.guiunamelist.add,'wtex.nbvertaxisp'


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
function rtmodel54::INIT,instanceid,_extra=extra

foo=self->rtmodelbase::init('Flux Rope',54,instanceid,_extra=extra)

return,1
end


; ------ class structure definition
pro rtmodel54__define
	struct={rtmodel54, inherits rtmodelbase}
return
end

; $Log: rtmodel54__define.pro,v $
; Revision 1.7  2012/03/12 13:57:37  thernis
; Add a cloud point at the surface of the sun, showing the direction of propagation.
;
; Revision 1.6  2011-08-10 17:31:39  thernis
; Insert CVS log
;