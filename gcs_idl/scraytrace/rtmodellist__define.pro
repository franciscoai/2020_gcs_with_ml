; $Id: rtmodellist__define.pro,v 1.9 2013/06/18 16:47:22 avourlid Exp $

common com_rtmodellist,com_guival

function rtmodellist::getNbModel
return,self.nbmodel
end
function rtmodellist::getModel,id
return,(*(self.pmodel))[id]
end

function rtmodellist::getavailablemodelsdesc
return,*(self.pavailablemodelsdesc)
end
function rtmodellist::getavailablemodelsname
return,*(self.pavailablemodelsname)
end



function rtmodellist::getidfrominstanceid,instanceid
id=0l
while (instanceid ne ((*self.pmodel)[id]).instanceid) do id++
return,id
end

function rtmodellist::getmodelbyinstanceid,instanceid
id=self->getidfrominstanceid(instanceid)
return,(*self.pmodel)[id]
end


pro rtmodellist::setCarrStonyLonOffset,carrstonylonoffset
for i=0l,self.nbmodel-1 do begin
	((*(self.pmodel))[i])->setCarrStonyLonOffset,carrstonylonoffset
endfor
return
end

pro rtmodellist::setCarrorStony,carrorstony
for i=0l,self.nbmodel-1 do begin
	((*(self.pmodel))[i])->setCarrorStony,carrorstony
endfor
return
end


pro rtmodellist::setParentGuiRef,guiref

for i=0l,self.nbmodel-1 do begin
	((*(self.pmodel))[i])->setParentGuiRef,guiref
endfor
return
end




pro rtmodellist::updateAllCloudColorBox

; -- set color in colorbox
for i=0l,self.nbmodel-1 do begin
	((*(self.pmodel))[i])->updateCloudColorBox
endfor

return
end


function rtmodellist::getClouds

lcloud=list()
for i = 0l,self.nbmodel-1 do begin
    modelname = ((*(self.pmodel))[i])->getModelname()
    instanceid = ((*(self.pmodel))[i])->getInstanceid()
    cloud = *(((*(self.pmodel))[i])->getCloud())

    lcloud.add,{modelname:modelname, instanceid:instanceid, cloud:cloud}

endfor

return,lcloud
end



function rtmodellist::deletemodel,instanceid
id=self->getidfrominstanceid(instanceid)

if self.nbmodel eq 1 then begin
	obj_destroy,*self.pmodel[0]
	ptr_free,self.pmodel
	self.nbmodel=0L
endif else begin
	obj_destroy,(*(self.pmodel))[id]
	omodelarray=*(self.pmodel)
	if id eq 0 then begin
		self.pmodel=ptr_new(omodelarray[1:*])
	endif else if id eq self.nbmodel-1 then begin
		self.pmodel=ptr_new(omodelarray[0:id-1]) 
	endif else begin
		self.pmodel=ptr_new([omodelarray[0:id-1],omodelarray[id+1:*]])
	endelse
	self.nbmodel=self.nbmodel-1
endelse
return,1
end

function rtmodellist::addmodel,modelname
if self.nbmodel eq 0 then begin
	self.pmodel=ptr_new([obj_new(modelname,0)])
	self.nbmodel++
	newinstanceid=0l
endif else begin
	; ---- get a free instanceid
	; -- get all the instanceid
	instanceidlist=lonarr(self.nbmodel)
	for i=0l,self.nbmodel-1 do begin
		instanceidlist[i]=((*self.pmodel)[i]).instanceid
	endfor
	if max(instanceidlist) eq (self.nbmodel-1) then begin
		freeid=self.nbmodel
	endif else begin
		freeid=0l
		srt=sort(instanceidlist)
		while (instanceidlist[srt[freeid]] eq freeid) do freeid++
	endelse
	omodelarray=*(self.pmodel)
	self.pmodel=ptr_new([omodelarray,obj_new(modelname,freeid)])
	self.nbmodel++
	newinstanceid=freeid
endelse


wmodelbasejunk=(self->getmodelbyinstanceid(newinstanceid))->buildWidget(self.wtab_modeltab)
(self->getmodelbyinstanceid(newinstanceid))->updateCloudColorBox

return,newinstanceid
end



; ---- build position and parameter widget for each models
pro rtmodellist::buildWidget,wtab_modeltab
self.wtab_modeltab=wtab_modeltab
for i=0l,self->getNbModel()-1 do begin
	wmodelbasejunk=(self->getModel(i))->buildWidget(wtab_modeltab)
endfor
return
end


; ---- constructor
function rtmodellist::INIT,reset=reset
common com_rtmodellist,com_guival

self.pavailablemodelsdesc=ptr_new(['Flux Rope','Bow Shock','Cylinder', 'Spheroid Shock'])
self.pavailablemodelsname=ptr_new(['rtmodel54','rtmodel29','rtmodel75','rtmodel36'])

; ---- test if gui values were saved in a previous call
if n_elements(com_guival) ne 0 and ~keyword_set(reset) then begin
    ; ---- recall gui config
    ; -- how many models
    nbmodel = n_elements(com_guival)
    self.nbmodel = nbmodel
    self.pmodel = ptr_new(objarr(self.nbmodel))
    for i=0,nbmodel-1 do begin
        classname = com_guival[i].sinstanceval.classname
        color = com_guival[i].sinstanceval.cloudcolor
        (*(self.pmodel))[i]=obj_new(classname,i,color = color, sinitval = com_guival[i])
    endfor
endif else begin
    ; ---- initialize with default values
    self.nbmodel=2
    self.pmodel=ptr_new(objarr(self.nbmodel))
    (*(self.pmodel))[0]=obj_new('rtmodel54',0,color=[0,255,0])
    (*(self.pmodel))[1]=obj_new('rtmodel29',1,color=[0,0,255])
    (*(self.pmodel))[1]=obj_new('rtmodel75',1,color=[255,0,100])
    (*(self.pmodel))[1]=obj_new('rtmodel36',1,color=[255,0,100])

endelse

return,1
end


pro rtmodellist::saveInitVal
common com_rtmodellist,com_guival

com_guival = list()

for i=0l,self.nbmodel-1 do begin
    ((*(self.pmodel))[i])->saveInitVal,sguival
    com_guival.add, sguival
endfor
return
end

function rtmodellist::getGUIVal
common com_rtmodellist
return,com_guival
end

pro rtmodellist::updateInitValAllModels
common com_rtmodellist,com_guival
if n_elements(com_guival) eq 0 then return
for i=0l,self.nbmodel-1 do begin
    ((*(self.pmodel))[i])->updateInitVal,com_guival[i]
endfor
return
end

; ---- destructor
pro rtmodellist::CLEANUP

if ptr_valid(self.pmodel) then obj_destroy,*(self.pmodel)
ptr_free,self.pmodel,self.pinstanceid,self.pavailablemodelsdesc,self.pavailablemodelsname

return
end



pro rtmodellist__define

struct = {  rtmodellist,$
            pmodel:ptr_new(),$
            pinstanceid:ptr_new(),$
            nbmodel:0l,$
            pavailablemodelsdesc:ptr_new(),$
            pavailablemodelsname:ptr_new(),$
            wtab_modeltab:0L}

return
end



; $Log: rtmodellist__define.pro,v $
; Revision 1.9  2013/06/18 16:47:22  avourlid
; added spheroid model (created by O. Olmedo) option to rtcloudwidget
; updated rtmodellist__define
;
; Revision 1.8  2012/03/12 14:00:51  thernis
; Add cylinder model: model id 75
;
; Revision 1.7  2011-08-10 17:31:39  thernis
; Insert CVS log
;

