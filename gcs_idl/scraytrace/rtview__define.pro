; $Id: rtview__define.pro,v 1.12 2013/01/29 17:54:09 thernis Exp $


pro rtview::setCloudColor,instanceid
; -- set color and point size
self.win->draw,self.view
return
end


pro rtview::refreshView

; ---- hide model if requested
for i=0l,(self.rtmodellist)->getnbmodel()-1 do begin
    hide=~(((self.rtmodellist)->getModel(i))->getWireOnOff())
    (*self.grmodel)[i]->setProperty,hide=hide

endfor

self.win->draw,self.view

return
end



function rtview::getPswire
nbmodel = self.rtmodellist->getNbModel()
wire = list()
for i=0l,nbmodel-1 do begin
    cloud=((self.rtmodellist)->getModel(i))->getCloud()
    if ptr_valid(cloud) then begin
        poly = (*self.grmodel)[i]->get(position=0)
        poly->getProperty,data=listofpoint

        modelname = ((self.rtmodellist)->getModel(i))->getModelname()
        instanceid = ((self.rtmodellist)->getModel(i))->getInstanceid()

        wire.add,{modelname:modelname, instanceid:instanceid, projpoints:listofpoint}
 
    endif
endfor

return,wire
end




function rtview::getDisplayedView
oim = self.win->read()
oim->getProperty, data = disp
return,disp
end


; ---- sync the current view with the modellist
pro rtview::syncwithmodellist

; ---- clean up
for i=0l,self.nbdisplayedmodel-1 do begin
    self.view->remove,position=1
    obj_destroy,(*self.grmodel)[i]
endfor



nbmodel=self.rtmodellist->getNbModel()
self.nbdisplayedmodel=nbmodel

if nbmodel eq 0 then return

pcloudimage=ptr_new(ptrarr(nbmodel))
modelarray=objarr(nbmodel)
for i=0l,nbmodel-1 do begin

    color=((self.rtmodellist)->getModel(i))->getCloudColor()

    ; -- define cloud, with temporary dummy data for now
    junkdata=fltarr(2,4)

    polyline=obj_new('IDLgrPolyline',junkdata,symbol=((self.rtmodellist)->getModel(i))->getSymbol(),linestyle=6)
    modelarray[i]=obj_new('IDLgrModel')
    self.view->add,modelarray[i],position=i+1
    modelarray[i]->add,polyline

   (*pcloudimage)[i]=ptr_new(junkdata)
endfor

self.pcloudimage=pcloudimage
if ptr_valid(self.grmodel) then ptr_free,self.grmodel
self.grmodel=ptr_new(modelarray)
self.win->draw,self.view

return
end



; ---- link the model list with the view
pro rtview::setmodellist,rtmodellist
self.rtmodellist=rtmodellist
self->syncwithmodellist
return
end




pro rtview::setParentGuiRef,guiref
self.parentguiref=guiref
return
end


function rtview::getPswire
nbmodel = self.rtmodellist->getNbModel()
wire = list()
for i=0l,nbmodel-1 do begin
    cloud=((self.rtmodellist)->getModel(i))->getCloud()
    if ptr_valid(cloud) then begin
        poly = (*self.grmodel)[i]->get(position=0)
        poly->getProperty,data=listofpoint

        modelname = ((self.rtmodellist)->getModel(i))->getModelname()
        instanceid = ((self.rtmodellist)->getModel(i))->getInstanceid()

        wire.add,{modelname:modelname, instanceid:instanceid, projpoints:listofpoint}
 
    endif
endfor

return,wire
end


; ---- compute the cloud projection for each of the models
pro rtview::calcCloudProj

nbmodel=self.rtmodellist->getNbModel()
for i=0l,nbmodel-1 do begin

    cloud=((self.rtmodellist)->getModel(i))->getCloud()
    nerotang=((self.rtmodellist)->getModel(i))->getNerotang()
    nerotaxis=((self.rtmodellist)->getModel(i))->getNerotaxis()
    nerotcntr=((self.rtmodellist)->getModel(i))->getNerotcntr()
 
    if ptr_valid(cloud) then begin
        ; ---- compute the point cloud projection here
        ;      output is in the listofpoint array
        rtcloud,*cloud,sout,imsize=self.imdispsize,scchead=*self.phdr,neang=((self.rtmodellist)->getModel(i))->getNeang(),nepos=((self.rtmodellist)->getModel(i))->getNepos(),netranslation=((self.rtmodellist)->getModel(i))->getNeshift(),/fclip,/flistout,listout=listofpoint,nerotang=nerotang,nerotaxis=nerotaxis,nerotcntr=nerotcntr,/quiet;,/unload

        ; ---- update cloud visualization
        poly=(*self.grmodel)[i]->get(position=0)
        poly->setProperty,data=listofpoint
        
        ; ---- Save cloud if a new one has been calculated
        szcloud=size(*cloud,/dim)
        szclouditool=size(*((*self.pcloudimage)[i]),/dim)
        if ~(szcloud[0] eq szclouditool[0] and szcloud[1] eq szclouditool[1]) then begin

            ; -- remember the cloud size present in the itool
            ptr_free,(*self.pcloudimage)[i]
            (*self.pcloudimage)[i]=ptr_new(*cloud)
        endif

    endif
endfor

self.win->draw,self.view

return
end


; ---- constructor
function rtview::INIT,im,hdr,viewid,location=location

foo = self->rtviewbase::INIT(im,hdr,viewid,location=location)

; ---- init display
owin=obj_new('IDLgrWindow',dimension=self.imdispsize,title=self.title,retain=2,renderer=1,location=location)

oview=obj_new('IDLgrView',view=[0,0,self.imdispsize[0],self.imdispsize[1]],color=[0,0,0])

oidlgrmodel=obj_new('IDLgrModel')
oimage=obj_new('IDLgrImage',im)

; self.view->add,self.idlgrmodel,position=0
; self.idlgrmodel->add,self.image
; self.win->draw,self.view

oview->add,oidlgrmodel,position=0
oidlgrmodel->add,oimage
owin->draw,oview
;stop

self.win = owin
self.view = oview
self.idlgrmodel = oidlgrmodel
self.image = oimage


return,1
end

; ---- destructor
pro rtview::CLEANUP
obj_destroy,self.win
obj_destroy,self.view

self->rtviewbase::CLEANUP

return
end

pro rtview__define
struct={rtview,$
        inherits rtviewbase,$
        win:obj_new(),$
        view:obj_new()}

return
end

; $Log: rtview__define.pro,v $
; Revision 1.12  2013/01/29 17:54:09  thernis
; Change the way the display window are opened. This was needed for IDL 8.2, it was displaying an empty window.
;
; Revision 1.11  2011-08-12 15:07:45  thernis
; Fix missing view name in windows title bar
;
; Revision 1.10  2011-08-12 13:49:05  thernis
; Implement rtviewbase class
;
; Revision 1.9  2011-08-10 17:31:39  thernis
; Insert CVS log
;
