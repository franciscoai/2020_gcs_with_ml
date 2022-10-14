; $Id: rtwcloudmaingui__define.pro,v 1.7 2011/08/10 17:31:39 thernis Exp $

function rtwcloudmaingui::getRtmodellist
return,self.rtmodellist
end

function rtwcloudmaingui::getWbase
return,self.wbase
end

function rtwcloudmaingui::getView
return,*self.pview
end

function rtwcloudmaingui::getCarrStonyLonOffset
return,self.carrstonylonoffset
end


pro rtwcloudmaingui::setCloudColorinAllViews,instanceid

nbviews=n_elements(*self.pview)
for i=0l,nbviews-1 do begin
	((*self.pview)[i])->setCloudColor,instanceid
endfor

return
end


pro rtwcloudmaingui::calcCarrStonyLonOffset

if self.refdateobs eq (((*self.pview)[0])->getHdr()).date_obs then $
    self.carrstonylonoffset =   (((*self.pview)[0])->getHdr()).hgln_obs - $
                                (((*self.pview)[0])->getHdr()).crln_obs $
    else begin
	foo = get_stereo_lonlat(self.refdateobs,'earth',system='CAR',/degrees)
	self.carrstonylonoffset = foo[1]
endelse

; ---- update each models
(self.rtmodellist)->setCarrStonyLonOffset,self.carrstonylonoffset
return
end


function rtwcloudmaingui::formatscpos,viewid

if self.carrorstony eq 0 then begin
	; -- display Carrington coords
	lonobs=(((*self.pview)[viewid])->getHdr()).wcs_struct.position.crln_obs
	latobs=(((*self.pview)[viewid])->getHdr()).wcs_struct.position.crlt_obs
endif else begin
	; -- display Stonyhurst coords
	lonobs=(((*self.pview)[viewid])->getHdr()).wcs_struct.position.crln_obs-(((*self.pview)[viewid])->getHdr()).wcs_struct.position.carr_earth
	latobs=(((*self.pview)[viewid])->getHdr()).wcs_struct.position.crlt_obs
endelse

return,' LON: '+string(lonobs,form='(F6.1)')+' , LAT: '+string(latobs,form='(F5.1)')
end



; ---- deal with change of coord system
function rtwcloudmaingui::carrstony,ev,uname


; -- switch the flag
oldcarrosstony=self.carrorstony
self.carrorstony=(ev.value eq 'stony' && ev.select eq 1)

; -- return if no change
if oldcarrosstony eq self.carrorstony then return,0b

; ---- update all the widget coordinate outputs
; -- update the coord for each view
nbview=n_elements(*self.pview)
for i=0l,nbview-1 do begin
	viewidtxt=string(i,form='(I3.3)')
	id=widget_info(self.wbase,find_by_uname='bas.wlab.viewcoord.'+viewidtxt)
	widget_control,id,set_value=self->formatscpos(i)
endfor

; -- tell the models to update the longitude slider
(self.rtmodellist)->setCarrorStony,self.carrorstony

return,0b
end



pro rtwcloudmaingui::recalcViews,refreshonly=refreshonly
nbview=n_elements(*self.pview)
for i=0l,nbview-1 do begin
	if ~keyword_set(refreshonly) then ((*self.pview)[i])->calcCloudProj
	((*self.pview)[i])->refreshView
endfor
return
end

pro rtwcloudmaingui::eventHandler,ev,uname

case strmid(uname,0,3) of
	'bas' : begin
		case strmid(uname,4) of
			'wlis.carrstony' : foo=self->carrstony(ev,uname)
			'wtex.referencedate' : begin
				widget_control,ev.id,get_value=date
				self.refdateobs=date
				self->calcCarrStonyLonOffset
			end
			'wbut.go' : begin
				; ---- raytrace each view
				nbview=n_elements(*self.pview)
				for i=0l,nbview-1 do begin
					widget_control,self.wtext_losrangemin,get_value=losrangemin
					widget_control,self.wtext_losrangemax,get_value=losrangemax
					widget_control,self.wtext_losnbp,get_value=losnbp
					((*self.pview)[i])->raytrace,float([losrangemin,losrangemax]),long(losnbp[0]);,self.rtmodellist

				endfor
			end
			'wmen.addmodel' : begin
				widget_control,ev.id,get_uvalue=modelname
				print,'Adding model ',modelname
				idi=self.rtmodellist->addmodel(modelname)
				self->calcCarrStonyLonOffset
                ; -- set carrorstony of the new model
                (self.rtmodellist->getmodelbyinstanceid(idi))->setCarrorStony,self.carrorstony
				; ---- sync the views with the new model
				for i=0l,n_elements(*self.pview)-1 do begin
					((*self.pview)[i])->syncwithmodellist
					((*self.pview)[i])->calcCloudProj
					((*self.pview)[i])->refreshView
                endfor
            end
            'modeltab' : ; do nothing
            else : message,'No case found for uname : '+uname,/info
        endcase
    end
    else : begin
        case 1 of 
            strmid(uname,4) eq 'wbut.deletemodel' : begin
                instanceid=long(strmid(uname,0,3))

                ((self.rtmodellist)->getmodelbyinstanceid(instanceid))->destroygui

                foo=(self.rtmodellist)->deletemodel(instanceid)
                ; ---- resync the views the new model list
                for i=0l,n_elements(*self.pview)-1 do begin
                    ((*self.pview)[i])->syncwithmodellist
                    ((*self.pview)[i])->calcCloudProj
                    ((*self.pview)[i])->refreshView
                endfor
            end
		else : begin
			instanceid=long(strmid(uname,0,3))
			flagupdate=((self.rtmodellist)->getmodelbyinstanceid(instanceid))->eventHandler(ev,uname)
			if (flagupdate and 4) ne 0 then begin
				self->setCloudColorinAllViews,instanceid
			endif
			if flagupdate and 1b then self->recalcViews,refreshonly=(flagupdate and 2b)

		end
		endcase
	end
endcase

return
end





; ---- constructor
function rtwcloudmaingui::INIT,view=view,reset=reset

; ---- init the model related objects
self.rtmodellist = obj_new('rtmodellist',reset=reset)
(self.rtmodellist)->setParentGuiRef,self

if keyword_set(view) then begin
	self.pview=ptr_new(view)
	nbviews=n_elements(view)
	for i=0l,nbviews-1 do begin
		((*self.pview)[i])->setmodellist,self.rtmodellist
	endfor
endif


; -- init gui parameters
self.losrange=[-10.,10]
self.losnbp=64l
self.carrorstony=0b
(self.rtmodellist)->setCarrorStony,self.carrorstony

; -- set reference date
self.refdateobs = (((*self.pview)[0])->getHdr()).date_obs
self->calcCarrStonyLonOffset

; ---- draw main GUI
wbase = widget_base(RESOURCE_NAME = 'base',title='STEREO Cloud',/column,/BASE_ALIGN_TOP,mbar=bar)
addmodel_menu = widget_button(bar,value='Add Model',/menu)
availablemodelsdesc = (self.rtmodellist)->getavailablemodelsdesc()
availablemodelsname = (self.rtmodellist)->getavailablemodelsname()

for i=0l,n_elements(availablemodelsdesc)-1 do begin
	addmodel_menu_foo = widget_button(addmodel_menu,$
	    value = availablemodelsdesc[i],$
	    uvalue = availablemodelsname[i],$
	    uname = 'bas.wmen.addmodel')
endfor

wtab = widget_tab(wbase,/align_left)

; ---- tab 1: simu parameters 
wbase_tab1 = widget_base(wtab,title='Simu',/column)
wlabel_losrangemin = widget_label(wbase_tab1,value='LOS Range Mini')
self.wtext_losrangemin = widget_text(wbase_tab1,$
    /editable,$
    value=strtrim(self.losrange[0],2),$
    uname='bas.wtext_losrangemin',$
    /kbrd_focus_events)
wlabel_losrangemax = widget_label(wbase_tab1,value='LOS Range Maxi')
self.wtext_losrangemax = widget_text(wbase_tab1,$
    /editable,$
    value=strtrim(self.losrange[1],2),$
    uname='bas.wtext_losrangemax',$
    /kbrd_focus_events)
wlabel_losnbp = widget_label(wbase_tab1,value='LOS Number of Points')
self.wtext_losnbp = widget_text(wbase_tab1,$
    /editable,$
    value=strtrim(self.losnbp,2),$
    uname='bas.wtext_losnbp',$
    /kbrd_focus_events)

wlabel_msg1 = widget_label(wbase_tab1,$
    value='['+strtrim(self.losrange[0],2)+','+strtrim(self.losrange[1],2)+'],'+strtrim(self.losnbp,2),$
    uname='bas.wlabel_msg3')

wlabel_msg2 = widget_label(wbase_tab1,value='Press ENTER when done',uname='bas.wlabel_msg2')

wbase_tab3 = widget_base(wtab,title='Models',/column)
wtab_modeltab = widget_tab(wbase_tab3,/align_left,uname='bas.modeltab')

; ---- Draw each model GUI
self.rtmodellist->buildWidget,wtab_modeltab


; ---- Draw rest of the GUI
wlist_carrstony=cw_bgroup(wbase,['Carrington','Stonyhurst'],button_uvalue=['carr','stony'],column=2,set_value=self.carrorstony,ids=[0,1],/exclusive,uname='bas.wlis.carrstony')

; -- the reference date (date of view id 0)
wlabel_refdate=widget_label(wbase,value='Reference Date',uname='wlabel_referencedate')
wtext_refdate=widget_text(wbase,/editable,value=self.refdateobs,uname='bas.wtex.referencedate')


; ---- position and date info for each views
; -- how many views
nbview=n_elements(*self.pview)
for i=0l,nbview-1 do begin

	viewidtxt=string(i,form='(I3.3)')

	junk=widget_label(wbase,value='View '+ (((*self.pview)[i])->getHdr()).obsrvtry +' '+  (((*self.pview)[i])->getHdr()).detector,uname='bas.wlab.viewname.'+viewidtxt)
	junk=widget_label(wbase,value=self->formatscpos(i),uname='bas.wlab.viewcoord.'+viewidtxt)
	junk=widget_label(wbase,value='Date Obs: '+(((*self.pview)[i])->getHdr()).date_obs,uname='bas.wlab.viewdate.'+viewidtxt)
endfor

; ---- Draw command buttons
wbutton_go=widget_button(wbase,value='Generate View',uname='bas.wbut.go')

wbutton_quit=widget_button(wbase,value='Quit',uname='quit')


self.wbase=wbase

if ~keyword_set(reset) then self.rtmodellist->updateInitValAllModels

self->recalcViews


return,1
end


pro rtwcloudmaingui::saveInitVal
(self.rtmodellist)->saveInitVal
return
end

function rtwcloudmaingui::getSgui
return,(self.rtmodellist)->getGUIVal()
end


; ---- destructor
pro rtwcloudmaingui::CLEANUP
ptr_free,self.pview,self.activatedmodels
obj_destroy,self.rtmodellist

return
end


pro rtwcloudmaingui__define

struct = {  rtwcloudmaingui,$
            rtmodellist:obj_new(),$
            wbase:0L,pview:ptr_new(),$
            refdateobs:'',$
            carrstonylonoffset:0.,$
            losrange:[0.,0],$
            losnbp:0l,$
            carrorstony:0b,$
            wtext_losrangemin:0L,$
            wtext_losrangemax:0L,$
            wtext_losnbp:0L,$
            activatedmodels:ptr_new()}

return
end

; $Log: rtwcloudmaingui__define.pro,v $
; Revision 1.7  2011/08/10 17:31:39  thernis
; Insert CVS log
;