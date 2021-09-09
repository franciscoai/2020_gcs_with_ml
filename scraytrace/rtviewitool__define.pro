; $Id: rtviewitool__define.pro,v 1.4 2011/08/12 15:07:45 thernis Exp $

function rtviewitool::calcidref,instanceid
idref=' '+strtrim(instanceid,2)+'_'+strtrim(self.viewid,2)
return,idref
end

pro rtviewitool::setCloudColor,instanceid

color=((self.rtmodellist)->getModel(instanceid))->getCloudColor()

; -- set color and point size
idref=self->calcidref(instanceid)
idvisu=self.otool->findidentifiers('*point'+idref,/visualization)
suc=self.otool->dosetproperty(idvisu,'sym_color',color)
suc=self.otool->dosetproperty(idvisu,'sym_size',1)
self.otool->commitactions


return
end


pro rtviewitool::refreshView

for i=0l,(self.rtmodellist)->getnbmodel()-1 do begin

	idref=self->calcidref(i)

	hide=~(((self.rtmodellist)->getModel(i))->getWireOnOff())

	idvisu=self.otool->findidentifiers('*point'+idref,/visualization)
	suc=self.otool->dosetproperty(idvisu,'hide',hide)
	self.otool->commitactions


endfor


return
end


function rtviewitool::getPswire
nbmodel = self.rtmodellist->getNbModel()
wire = list()
for i=0l,nbmodel-1 do begin
    cloud=((self.rtmodellist)->getModel(i))->getCloud()
    if ptr_valid(cloud) then begin

        idref=self->calcidref(i)

        idvisu=self.otool->findidentifiers('*point'+idref,/visualization)
        ovisu=self.otool->getbyidentifier(idvisu)
        odatavertices=ovisu->getparameter('vertices')
        suc=odatavertices->getdata(listofpoint)

        modelname = ((self.rtmodellist)->getModel(i))->getModelname()
        instanceid = ((self.rtmodellist)->getModel(i))->getInstanceid()

        wire.add,{modelname:modelname, instanceid:instanceid, projpoints:listofpoint}
 
    endif
endfor

return,wire
end




; ---- sync the current view with the modellist
pro rtviewitool::syncwithmodellist

; ---- clean up
for i=0l,self.nbdisplayedmodel-1 do begin
	idref=self->calcidref(i)
	idvisu=self.otool->findidentifiers('*point'+idref,/visualization)
	if idvisu ne '' then begin
		r=self.otool->removebyidentifier(idvisu)

		; -- remove a data
		iddata=self.otool->findidentifiers('*cloud'+idref,/data)
		oParamSet=self.otool->removebyidentifier(iddata)
	endif
endfor



nbmodel=self.rtmodellist->getNbModel()
self.nbdisplayedmodel=nbmodel

if nbmodel eq 0 then return

pcloudimage=ptr_new(ptrarr(nbmodel))
for i=0l,nbmodel-1 do begin

    ;	if obj_valid(parentguiref) then
    color=((self.rtmodellist)->getModel(i))->getCloudColor(); else color=[255,0,0]

	; -- define cloud itool data, with temporary cloud for now
	idref=self->calcidref(i)

	junkdata=fltarr(2,4)
	oParamSetplot=obj_new('IDLitParameterSet',name='Cloud'+idref,icon='plot',description='Cloud '+strtrim(i,2))
	odatavertex=obj_new('IDLitData',type='IDLVERTEX',junkdata)
	oParamSetplot->Add, odatavertex, PARAMETER_NAME = 'cloudpolyvertex'

	self.otool->add,oParamSetplot

	; -- define itool visualization
	oplot=obj_new('IDLitVisShapePoint',name='point'+idref)
	r=oplot->setparameterset(oParamSetplot)
	oplot->setproperty,sym_index=3
	oplot->setproperty,sym_size=1
	self.otool->add,oplot

	; -- set color and point size
	self->setCloudColor,i

    (*pcloudimage)[i]=ptr_new(junkdata)


endfor

self.pcloudimage=pcloudimage

return
end




pro rtviewitool::setParentGuiRef,guiref
self.parentguiref=guiref
return
end


; ---- compute the cloud projection for each of the models
pro rtviewitool::calcCloudProj


nbmodel=self.rtmodellist->getNbModel()
for i=0l,nbmodel-1 do begin

	cloud=((self.rtmodellist)->getModel(i))->getCloud()
	nerotang=((self.rtmodellist)->getModel(i))->getNerotang()
    nerotaxis=((self.rtmodellist)->getModel(i))->getNerotaxis()
        nerotcntr=((self.rtmodellist)->getModel(i))->getNerotcntr()
 
	if ptr_valid(cloud) then begin
        ; ---- compute the point cloud projection here
        ;      output is in the listofpoint array
		rtcloud,*cloud,sout,imsize=self.imdispsize,scchead=*self.phdr,neang=((self.rtmodellist)->getModel(i))->getNeang(),nepos=((self.rtmodellist)->getModel(i))->getNepos(),netranslation=((self.rtmodellist)->getModel(i))->getNeshift(),/fclip,/flistout,listout=listofpoint,/unload,nerotang=nerotang,nerotaxis=nerotaxis,nerotcntr=nerotcntr,/quiet

        idref=self->calcidref(i)

		; ---- only change data if no new cloud has been calculated
		szcloud=size(*cloud,/dim)
		szclouditool=size(*((*self.pcloudimage)[i]),/dim)
		if szcloud[0] eq szclouditool[0] and szcloud[1] eq szclouditool[1] then begin
			idvisu=self.otool->findidentifiers('*point'+idref,/visualization)
			ovisu=self.otool->getbyidentifier(idvisu)
			odatavertices=ovisu->getparameter('vertices')
			suc=odatavertices->setdata(listofpoint)
		endif else begin
		; ---- change data and plot if a new cloud has been calculated
			idvisu=self.otool->findidentifiers('*point'+idref,/visualization)
			r=self.otool->removebyidentifier(idvisu)

			; -- remove a data
			iddata=self.otool->findidentifiers('*cloud'+idref,/data)
			oParamSet=self.otool->removebyidentifier(iddata)

			; -- create new data and register it in the data manager browser
			odatavertex=obj_new('IDLitData',type='IDLVERTEX',listofpoint)
			oParamSetplot=obj_new('IDLitParameterSet',name='Cloud'+idref,icon='plot',description='Cloud'+idref)
			oParamSetplot->Add, odatavertex, PARAMETER_NAME = 'cloudpolyvertex'
			self.otool->add,oParamSetplot

			; -- set back the plot
			oplot=obj_new('IDLitVisShapePoint',name='point'+idref)
			oplot->setproperty,sym_index=3
			oplot->setproperty,sym_size=1
			r=oplot->setparameterset(oParamSetplot)
			self.otool->add,oplot

			; -- set color and point size
			self->setCloudColor,i

			; -- remember the cloud size present in the itool
			ptr_free,(*self.pcloudimage)[i]
			(*self.pcloudimage)[i]=ptr_new(*cloud)

		endelse

	endif
endfor
return
end


; ---- constructor
function rtviewitool::INIT,im,hdr,viewid,location=location

foo = self->rtviewbase::INIT(im,hdr,viewid,location=location)

; ---- Open the data and wireframe window: use iimage
iimage,im,identifier=idiimage,title=self.title
idtool=itgetcurrent(tool=otool)

; ---- store the iimage id here
self.otool=otool

return,1
end


pro rtviewitool__define
struct  = {rtviewitool,$
            inherits rtviewbase,$
            otool:obj_new()}
return
end


; $Log: rtviewitool__define.pro,v $
; Revision 1.4  2011/08/12 15:07:45  thernis
; Fix missing view name in windows title bar
;
; Revision 1.3  2011-08-12 13:49:05  thernis
; Implement rtviewbase class
;
; Revision 1.2  2011-08-10 17:31:39  thernis
; Insert CVS log
;
