; $Id: rtmodelbase__define.pro,v 1.7 2011/08/10 17:31:39 thernis Exp $

function rtmodelbase::getSymbol
return,self.symbol
end
function rtmodelbase::getCloudColor
return,self.cloudcolor
end
function rtmodelbase::getNepos
return,self.nepos
end
pro rtmodelbase::setNepos,nepos
self.nepos=nepos
return
end
function rtmodelbase::getNeshift
return,self.neshift
end
function rtmodelbase::getInstanceid
return,self.instanceid
end
function rtmodelbase::getModelid
return,self.modelid
end
function rtmodelbase::getModelname
return,self.modelname
end
function rtmodelbase::getNeang
return,self.neang
end
function rtmodelbase::getNerotang
return,self.nerotang
end
function rtmodelbase::getNerotaxis
return,self.nerotaxis
end
function rtmodelbase::getNerotcntr
return,self.nerotcntr
end
function rtmodelbase::getCloud
return,self.cloud
end
function rtmodelbase::getModParam
return,self.pmodparam
end
function rtmodelbase::getAllowRaytrace
return,self.allowraytrace
end


pro rtmodelbase::saveInitVal,sgui
sinstanceval = {modelname:self.modelname,$
                instanceid:self.instanceid,$
                classname:obj_class(self),$
                modelid:self.modelid,$
                cloud:*(self.cloud),$
                cloudcolor:self.cloudcolor,$
                wireonoff:self.wireonoff,$
                carrorstony:self.carrorstony,$
                carrstonylonoffset:self.carrstonylonoffset,$
                symbol:self.symbol}

guihash = hash()
foreach guiuname,self.guiunamelist do begin
    ; ---- get longitude in carrington 
    if guiuname eq 'wsli.lon' then $
        val = self->getLonSliderinCarrCoord() else $
        val = self->getWidgetValue(self.unamemodelid+guiuname)
    guihash += hash(guiuname,val)
endforeach

sgui = {sinstanceval:sinstanceval,$
        guihash:guihash}

return
end

pro rtmodelbase::updateInitVal,sgui
foreach guiuname,self.guiunamelist do begin
    ; ---- get widget id from uname
    wid = widget_info(self.wbase,find_by_uname=self.unamemodelid+guiuname)
    ; ---- update value
    widget_control,wid,set_value = (sgui.guihash)[guiuname]
endforeach
self->updateCloud
self->updateNeang
self->updateNepos
return
end


pro rtmodelbase::destroygui
widget_control,self.wbase,/destroy
return
end

; ------ update nepos
pro rtmodelbase::updateNepos

nrt = self->getWidgetValue(self.unamemodelid+'wsli.nrt')
ntl = self->getWidgetValue(self.unamemodelid+'wsli.ntl')
zne = self->getWidgetValue(self.unamemodelid+'wsli.zne')
hne = self->getWidgetValue(self.unamemodelid+'wsli.hne')

self.nepos = [zne,0.,0.]

self.neshift = [0.,0,hne]

self.nerotang = [ntl,nrt,0.]*!dtor
self.nerotaxis = long([2,3,1])
self.nerotcntr = [0.,0,1.]

return
end
pro rtmodelbase::updateCloud
return
end
pro rtmodelbase::updateNeang
return
end



function rtmodelbase::eventHandler,ev,uname
flagout = 0b
widgetname = strmid(uname,4)
case widgetname of 
    'wtex.cloudcolor' : begin
        widget_control, ev.id, get_value=colortxt
        rvb = txtcolor2rvb(colortxt)
        self.cloudcolor = rvb
        self->updatecloudcolorbox
        self.symbol->setProperty, color=rvb
        flagout = 7b
    end
    'wtex.entermaxheight' : begin
        flagout = '0001'b
        widget_control,ev.id,get_value = maxh
        maxh = strcompress(maxh, /remove_all)
        if is_a_number(maxh) then begin
            unamemodelid = strmid(uname, 0, 3)
            hsliderid = widget_info(self.wbase,$
                find_by_uname=unamemodelid+'.wsli.hgt')
            widget_control, hsliderid, get_value=sliderval
            widget_control, hsliderid, set_value=[sliderval, 0., float(maxh)]
        endif
    end
    else :
endcase


return,flagout
end


function rtmodelbase::getWidgetValue,widgetuname

wid = widget_info(self.wbase, find_by_uname=widgetuname)
widget_control, wid,get_value=val

return,val
end

pro rtmodelbase::setCarrorStony,carrorstony

if carrorstony eq self.carrorstony then return

if self.wslider_lon ne 0 then lonsliderincarr=self->getLonSliderinCarrCoord()

self.carrorstony = carrorstony

if self.wslider_lon ne 0 then begin
	lonslider = self->calcSliderLon(lonsliderincarr)
	widget_control, self.wslider_lon, set_value=lonslider

endif


return
end

function rtmodelbase::getCarrorStony
return,self.carrorstony
end


pro rtmodelbase::setCarrStonyLonOffset, carrstonylonoffset

if self.wslider_lon ne 0 then lonsliderincarr = self->getLonSliderinCarrCoord()

self.carrstonylonoffset = carrstonylonoffset

if self.wslider_lon ne 0 then begin
	lonslider = self->calcSliderLon(lonsliderincarr)
	widget_control, self.wslider_lon, set_value=lonslider
endif


return
end


function rtmodelbase::getLonSliderinCarrCoord
widget_control, self.wslider_lon, get_value=lonslider
lonsliderincarr = lonslider - (self.carrorstony ? self.carrstonylonoffset : 0.)
return,lonsliderincarr
end


function rtmodelbase::getWireOnOff
return,self.wireonoff
end


function rtmodelbase::getWbase
return,self.wbase
end


function rtmodelbase::buildWidget,wparent
self.wbase = widget_base(wparent, /column)
wlabel_nbvertcirup = widget_label(self.wbase,$
                                    value='This is rtmodelbase::buildWidget',$
                                    uname='wlab.rtmodelbaseINIT')
return,self.wbase
end

function rtmodelbase::calcSliderLon,loninit
lonslider = loninit + (self.carrorstony ? self.carrstonylonoffset : 0.)
makeit0_360,lonslider
return,lonslider
end

pro rtmodelbase::buildPositionGUI,wparent,unamemodelid,loninit=loninit

if ~keyword_set(loninit) then loninit = 0.

wslider_lon = cw_fslider(wparent,$
                        minimum=0.,$
                        maximum=360.,$
                        value=loninit,$
                        title='Longitude',$
                        uname=unamemodelid + 'wsli.lon',$
                        /drag,$
                        /edit,$
                        scroll=1.,$
                        xsize=360)
self.guiunamelist.add,'wsli.lon'
self.wslider_lon = wslider_lon

wslider_lat = cw_fslider(wparent,$
                        minimum=-90,$
                        maximum=90,$
                        value=0.,$
                        title='Latitude',$
                        uname=unamemodelid+'wsli.lat',$
                        /drag,$
                        /edit,$
                        scroll=1.,$
                        xsize=360)
self.guiunamelist.add,'wsli.lat'

wslider_rot = cw_fslider(wparent,$
                        minimum=-90,$
                        maximum=90,$
                        value=0.,$
                        title='Tilt Angle',$
                        uname=unamemodelid + 'wsli.rot',$
                        /drag,$
                        /edit,$
                        scroll=1.,$
                        xsize=360)
self.guiunamelist.add,'wsli.rot'


return
end

pro rtmodelbase::buildHeightGUI,wparent,unamemodelid,heightinit=heightinit

if ~keyword_set(heightinit) then heightinit=10.

wslider_hgt = cw_fslider(wparent,minimum=0.,maximum=25.,$
        value=heightinit,$
        title='Height',uname=unamemodelid+'wsli.hgt',/drag,/edit,scroll=1.,xsize=360)
self.guiunamelist.add,'wsli.hgt'

wlabel_entermaxheight = widget_label(wparent,$
        value='Enter slider max height : ',uname=unamemodelid+'wlab.entermaxheight')

wtext_entermaxheight = widget_text(wparent,/editable,value='25',$
        uname=unamemodelid+'wtex.entermaxheight')
self.guiunamelist.add,'wtex.entermaxheight'

return
end


pro rtmodelbase::buildChangeCloudColor,wparent,unamemodelid

colorinit=rvb2txtcolor(self.cloudcolor)

wlabel_cloudcolor = widget_label(wparent,$
    value='Cloud Color : ',$
    uname=self.unamemodelid+'wlab.cloudcolor')
wtext_cloudcolor = widget_text(wparent,$
    /editable,$
    value=colorinit,$
    uname=self.unamemodelid+'wtex.cloudcolor')

return
end



pro rtmodelbase::buildNeShift,wparent,unamemodelid


wslider_nerot = cw_fslider(wparent,minimum=-180.,maximum=180.,value=0.,$
    title='Ne Rotation',uname=unamemodelid+'wsli.nrt',/drag,/edit,scroll=1.,xsize=360)
self.guiunamelist.add,'wsli.nrt'

wslider_netilt = cw_fslider(wparent,minimum=-100.,maximum=100.,value=0.,$
    title='Ne Tilt',uname=unamemodelid+'wsli.ntl',/drag,/edit,scroll=1.,xsize=360)
self.guiunamelist.add,'wsli.ntl'

wslider_zne=cw_fslider(wparent,minimum=-5.,maximum=5.,value=0.,$
    title='Z Position',uname=unamemodelid+'wsli.zne',/drag,/edit,scroll=0.1,xsize=360)
self.guiunamelist.add,'wsli.zne'

wslider_hne=cw_fslider(wparent,minimum=-5.,maximum=5.,value=0.,$
    title='Height Position',uname=unamemodelid+'wsli.hne',/drag,/edit,scroll=0.1,xsize=360)
self.guiunamelist.add,'wsli.hne'

self->updateNepos


return
end

pro rtmodelbase::buildDeleteButton
wbutton_deletemodel = widget_button(self.wbase,$
    value='Delete this Model',uname=self.unamemodelid+'wbut.deletemodel')
return
end


pro rtmodelbase::updateCloudColorBox
widget_control,self.wdraw_cloudcolorbox,get_value=owindow
self.oviewcloudcolor->setproperty,color=self.cloudcolor
owindow->draw,self.oviewcloudcolor
return
end

pro rtmodelbase::buildCloudColorBox
self.wdraw_cloudcolorbox = widget_draw(self.wbase,$
    graphics_level=2,$
    xsize=100,$
    ysize=20,$
    retain=2,$
    renderer=1,$
    uname=self.unamemodelid+'wdra.cloudcolorbox')
self.oviewcloudcolor = obj_new('IDLgrView')
return
end




pro rtmodelbase::setParentGuiRef,guiref
self.parentguiref=guiref
return
end

function rtmodelbase::INIT,modelname,modelid,instanceid,color=color,sinitval=sinitval,_extra=extra

self.instanceid=instanceid
self.modelname=modelname
self.modelid=modelid
self.unamemodelid=string(self.instanceid,form='(I3.3)')+'.'
self.guiunamelist = list()

if n_elements(color) ne 0 then self.cloudcolor = color else self.cloudcolor=[0,255,0]

symbol=obj_new('IDLgrSymbol',1,color=self.cloudcolor)
self.symbol=symbol

return,1
end


; ---- destructor
pro rtmodelbase::CLEANUP
ptr_free,self.cloud,self.pmodparam
obj_destroy,self.oviewcloudcolor
return
end



; ------ defines the GUI common to all models
pro rtmodelbase__define
struct={    rtmodelbase,$
            modelname:'',$
            modelid:0l,$
            cloud:ptr_new(),$
            unamemodelid:'',$
            neang:[0.,0.,0.],$
            pmodparam:ptr_new(),$
            allowraytrace:0b,$
            wbase:0l,$
            wireonoff:0b,$
            wslider_lon:0l,$
            parentguiref:obj_new(),$
            carrorstony:0b,$
            carrstonylonoffset:0.,$
            cloudcolor:[0b,0b,0b],$
            nepos:[0.,0.,0.],$
            neshift:[0.,0.,0.],$
            nerotang:[0.,0,0],$
            nerotcntr:[0.,0,0],$
            nerotaxis:long([0,0,0]),$
            instanceid:0l,$
            wdraw_cloudcolorbox:0l,$
            oviewcloudcolor:obj_new(),$
            symbol:obj_new(),$
            guiunamelist:list()}
return
end


; $Log: rtmodelbase__define.pro,v $
; Revision 1.7  2011/08/10 17:31:39  thernis
; Insert CVS log
;