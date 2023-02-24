function rtguimodelbase::getCloudColor
return,self.cloudcolor
end
function rtguimodelbase::getNepos
return,self.nepos
end
function rtguimodelbase::getNeshift
return,self.neshift
end
function rtguimodelbase::getModelid
return,self.modelid
end
function rtguimodelbase::getModelname
return,self.modelname
end
function rtguimodelbase::getNeang
return,self.neang
end
function rtguimodelbase::getCloud
return,self.cloud
end
function rtguimodelbase::getModParam
return,self.pmodparam
end
function rtguimodelbase::getAllowRaytrace
return,self.allowraytrace
end


pro rtguimodelbase::saveInitVal
return
end



; ------ update nepos
pro rtguimodelbase::updateNepos

zne=self->getWidgetValue(self.unamemodelid+'wsli.zne')
hne=self->getWidgetValue(self.unamemodelid+'wsli.hne')

self.nepos=[zne,0.,0.]
self.neshift=[0.,0,hne]

return
end






function rtguimodelbase::eventHandler,ev,uname
;print,'In rtguimodelbase::eventHandler'

flagout=0b



widgetname=strmid(uname,4)
if widgetname eq 'wtex.cloudcolor' then begin
	widget_control,ev.id,get_value=colortxt
	rvb=txtcolor2rvb(colortxt)
	self.cloudcolor=rvb

;	print,'Yeah !'

	if obj_valid(self.parentguiref) then begin
		;print,'Cool !'

		modelid=fix(strmid(uname,0,3))
		self.parentguiref->setCloudColorinAllViews,modelid
		;self.parentguiref->recalcViews,/refreshonly
	endif	
endif

return,flagout
end


function rtguimodelbase::getWidgetValue,widgetuname

wid=widget_info(self.wbase,find_by_uname=widgetuname)
widget_control,wid,get_value=val

return,val
end

pro rtguimodelbase::setCarrorStony,carrorstony

if carrorstony eq self.carrorstony then return

if self.wslider_lon ne 0 then lonsliderincarr=self->getLonSliderinCarrCoord()

self.carrorstony=carrorstony

if self.wslider_lon ne 0 then begin
	lonslider=self->calcSliderLon(lonsliderincarr)
	widget_control,self.wslider_lon,set_value=lonslider
endif

return
end

function rtguimodelbase::getCarrorStony
return,self.carrorstony
end


pro rtguimodelbase::setCarrStonyLonOffset,carrstonylonoffset

if self.wslider_lon ne 0 then lonsliderincarr=self->getLonSliderinCarrCoord()

self.carrstonylonoffset=carrstonylonoffset

if self.wslider_lon ne 0 then begin
	lonslider=self->calcSliderLon(lonsliderincarr)
	widget_control,self.wslider_lon,set_value=lonslider
endif


return
end


function rtguimodelbase::getLonSliderinCarrCoord
widget_control,self.wslider_lon,get_value=lonslider
lonsliderincarr=lonslider - (self.carrorstony ? self.carrstonylonoffset : 0.)
return,lonsliderincarr
end


function rtguimodelbase::getWireOnOff
return,self.wireonoff
end


function rtguimodelbase::getWbase
return,self.wbase
end


function rtguimodelbase::buildWidget,wparent
self.wbase=widget_base(wparent,/column)
wlabel_nbvertcirup=widget_label(self.wbase,value='This is rtguimodelbase::buildWidget',uname='wlab.rtguimodelbaseINIT')
return,self.wbase
end

function rtguimodelbase::calcSliderLon,loninit
lonslider=loninit + (self.carrorstony ? self.carrstonylonoffset : 0.)
makeit0_360,lonslider
return,lonslider
end

pro rtguimodelbase::buildPositionGUI,wparent,unamemodelid,loninit=loninit

if ~keyword_set(loninit) then loninit=0.

wslider_lon=cw_fslider(wparent,minimum=0.,maximum=360.,value=loninit,title='Longitude',uname=unamemodelid+'wsli.lon',/drag,/edit,scroll=1.,xsize=360)
self.wslider_lon=wslider_lon
wslider_lat=cw_fslider(wparent,minimum=-90,maximum=90,value=0.,title='Latitude',uname=unamemodelid+'wsli.lat',/drag,/edit,scroll=1.,xsize=360)
wslider_rot=cw_fslider(wparent,minimum=-90,maximum=90,value=0.,title='Tilt Angle',uname=unamemodelid+'wsli.rot',/drag,/edit,scroll=1.,xsize=360)

return
end

pro rtguimodelbase::buildChangeCloudColor,wparent,unamemodelid

colorinit=rvb2txtcolor(self.cloudcolor)

wlabel_cloudcolor=widget_label(wparent,value='Cloud Color : ',uname=self.unamemodelid+'wlab.cloudcolor')
wtext_cloudcolor=widget_text(wparent,/editable,value=colorinit,uname=self.unamemodelid+'wtex.cloudcolor')

return
end


pro rtguimodelbase::buildNeShift,wparent,unamemodelid

wslider_zne=cw_fslider(wparent,minimum=-2.,maximum=2.,value=0.,title='Z Position',uname=unamemodelid+'wsli.zne',/drag,/edit,scroll=0.1,xsize=360)

wslider_hne=cw_fslider(wparent,minimum=-2.,maximum=2.,value=0.,title='Height Position',uname=unamemodelid+'wsli.hne',/drag,/edit,scroll=0.1,xsize=360)


return
end



pro rtguimodelbase::setParentGuiRef,guiref
self.parentguiref=guiref
return
end

function rtguimodelbase::INIT,modelid,modelname,modelguiid=modelguiid,_extra=_extra

;print,'In rtguimodelbase::INIT'

self.modelid=modelid
self.modelname=modelname
;self.unamemodelid=string(self.modelid,form='(I3.3)')+'.'
if keyword_set(modelguiid) then self.modelguiid=modelguiid
self.unamemodelid=string(self.modelguiid,form='(I3.3)')+'.'


self.cloudcolor=[255,0,0]

return,1
end


; ---- destructor
pro rtguimodelbase::CLEANUP
;print,'In rtguimodelbase::CLEANUP'
ptr_free,self.cloud,self.pmodparam
return
end



; ------ defines the GUI common to all models
pro rtguimodelbase__define

struct={rtguimodelbase,modelid:0l,modelname:'',cloud:ptr_new(),unamemodelid:'',neang:[0.,0.,0.],modelguiid:0l,pmodparam:ptr_new(),allowraytrace:0b,wbase:0l,wireonoff:0b,wslider_lon:0l,parentguiref:obj_new(),carrorstony:0b,carrstonylonoffset:0.,cloudcolor:[0b,0b,0b],nepos:[0.,0.,0.],neshift:[0.,0.,0.]}

return
end





; ------ Class testing
function rtguimodelbase_test

; ---- init counters
nbtest=1l
nberror=0l

; ---- default constructor
print,'Test ',nbtest
p=obj_new('rtguimodelbase')
if ~obj_valid(p) then begin
	message,'Default constructor did not work.',/info
	nberror++
endif
nbtest++

; ---- constructor with parameter init
print,'Test ',nbtest
p=obj_new('rtguimodelbase',40,'titi tutu')
if ~obj_valid(p) then begin
	message,'Constructor with parameter init failed.',/info
	nberror++
endif
nbtest++

; ---- exit if constructor does not work
if ~obj_valid(p) then goto,summary

; ---- getModelid method
print,'Test ',nbtest
if p->getModelid() ne 40 then begin
	message,'getModelid method did not returned the right value.',/info
	nberror++
endif
nbtest++
; ---- getModelname method
print,'Test ',nbtest
if p->getModelname() ne 'titi tutu' then begin
	message,'getModelname method did not returned the right value.',/info
	nberror++
endif
nbtest++


; ---- destructor
print,'Test ',nbtest
obj_destroy,p
nbtest++


; ---- summary
summary:
print,'nbtest : ',nbtest-1
print,'nberror : ',nberror

return,(nberror eq 0)
end



