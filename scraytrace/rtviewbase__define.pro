; $Id: rtviewbase__define.pro,v 1.1 2011/08/12 13:49:05 thernis Exp $

function rtviewbase::calcidref,instanceid
idref=' '+strtrim(instanceid,2)+'_'+strtrim(self.viewid,2)
return,idref
end


pro rtviewbase::updateDataImage,im
*self.pdataimage=im
return
end


pro rtviewbase::setHdr,hdr
self.phdr=ptr_new(hdr)
return
end

function rtviewbase::getHdr
return,*self.phdr
end

function rtviewbase::getImage
return,self.image
end

function rtviewbase::getPssim
if ~ptr_valid(self.pssim) then return,0
return,*self.pssim
end

function rtviewbase::getTitle
return,self.title
end

function rtviewbase::getDisplayedView
return,0
end


pro rtviewbase::raytrace,losrange,losnbp;,modellist

; ---- for each model
im=fltarr(self.imdispsize[0],self.imdispsize[1])
nbmodels=self.rtmodellist->getNbModel()
lssim=list()
for i=0l,nbmodels-1 do begin

    if ~(((self.rtmodellist)->getModel(i))->getAllowRaytrace()) then continue

    neang=((self.rtmodellist)->getModel(i))->getNeang()
    nepos=((self.rtmodellist)->getModel(i))->getNepos()
    neshift=((self.rtmodellist)->getModel(i))->getNeshift()
    nerotang=((self.rtmodellist)->getModel(i))->getNerotang()
    nerotaxis=((self.rtmodellist)->getModel(i))->getNerotaxis()
    nerotcntr=((self.rtmodellist)->getModel(i))->getNerotcntr()
    pmp=((self.rtmodellist)->getModel(i))->getModParam()

    if ptr_valid(pmp) then mp=*pmp

    modelid=((self.rtmodellist)->getModel(i))->getModelid()

    rtraytracewcs,sbt,sbp,sne,modelid=modelid,imsize=self.imdispsize,losrange=losrange,modparam=mp,nepos=nepos,neang=neang,netranslation=neshift,scchead=*self.phdr,losnbp=losnbp,/progressonly,nerotang=nerotang,nerotcntr=nerotcntr,nerotaxis=nerotaxis

    im+=sbt.im
    lssim.add,{sbt:sbt,sbp:sbp,sne:sne}
endfor

self.rtimage->setProperty,data=bytscl(alog10(im > 1e-14))
self.rtwin->draw,self.rtview

; -- store in structure for user output
self.pssim=ptr_new({imbtot:im,nbmodels:nbmodels,lssim:lssim})

return
end




; ---- link the model list with the view
pro rtviewbase::setmodellist,rtmodellist
self.rtmodellist=rtmodellist
self->syncwithmodellist
return
end




pro rtviewbase::setParentGuiRef,guiref
self.parentguiref=guiref
return
end


function rtviewbase::getPswire
wire = list()
return,wire
end



; ---- constructor
function rtviewbase::INIT,im,hdr,viewid,location=location

self.viewid=viewid

self.imdispsize=size(im,/dim)

if ~tag_exist(hdr,'obsrvtry') then begin
    hdr=create_struct(hdr,'obsrvtry',hdr.telescop)
endif

if ~tag_exist(hdr,'wcs_struct') then begin
    wcs=fitshead2wcs(hdr)
    hdr=create_struct(hdr,'wcs_struct',wcs)
endif


self.phdr=ptr_new(hdr)
self.pdataimage=ptr_new(im)

title=hdr.obsrvtry+' '+hdr.detector
self.title=title

if keyword_set(location) then locationrt=location+[0,540]

; ---- Open the raytrace output window
self.rtwin=obj_new('IDLgrWindow',dimension=self.imdispsize,title=title+' Raytrace',retain=2,renderer=1,location=locationrt)
self.rtview=obj_new('IDLgrView',view=[0,0,self.imdispsize[0],self.imdispsize[1]],color=[0,0,0])
self.rtidlgrmodel=obj_new('IDLgrModel')
self.rtimage=obj_new('IDLgrImage')

self.rtview->add,self.rtidlgrmodel
self.rtidlgrmodel->add,self.rtimage
self.rtwin->draw,self.rtview



; -- number of models that are actually displayed in the view: 
;    useful to know before adding or removing a model
self.nbdisplayedmodel=0l


return,1
end

; ---- destructor
pro rtviewbase::CLEANUP
obj_destroy,self.rtwin
obj_destroy,self.rtview
ptr_free,self.phdr,self.pdataimage,self.pcloudimage,self.pssim,self.pswire

return
end

pro rtviewbase__define
struct={rtviewbase,$
        imdispsize:[0l,0],$
        pdataimage:ptr_new(),$
        phdr:ptr_new(),$
        pcloudimage:ptr_new(),$
        idlgrmodel:obj_new(),$
        image:obj_new(),$
        parentguiref:obj_new(),$
        rtwin:obj_new(),$
        rtview:obj_new(),$
        rtidlgrmodel:obj_new(),$
        rtimage:obj_new(),$
        viewid:0l,$
        rtmodellist:obj_new(),$
        nbdisplayedmodel:0l,$
        grmodel:ptr_new(),$
        pssim:ptr_new(),$
        pswire:ptr_new(),$
        title:''}

return
end

; $Log: rtviewbase__define.pro,v $
; Revision 1.1  2011/08/12 13:49:05  thernis
; Implement rtviewbase class
;