;+
; $Id: rtcloudwidget.pro,v 1.12 2012/03/12 15:45:32 thernis Exp $
;
; PURPOSE:
;  Graphic interface for doing forward modeling of structures of the solar corona
;
; INPUTS: 
;  pim : ptrarr (or list) of images, rescaled and bytescaled
;  phdr : ptrarr (or list) of hdr corresponding to the images. pim and phdr must be 
;         of the same type: either both ptrarr or both list
;
;  itool : set this keyword if you want to use the itool viewer instead of regular viewer.
;          The itool feature won't be maintained in future releases.
;  reset : reset all the parameters of the GUI
;  demo : run demo mode
;
; OUTPUTS:
;  ssim : structure containing the simulated thomson scattering views.
;         The user must press the generate view button in order for that structure to be fill out.
;           NBVIEW                  : number of views
;           LVIEW[nbview]           : list of structures for each view
;               IMBTOT              : simulated image in Bsun
;               NBMODELS            : number of models
;               LSSIM[nbmodels]     : List for each model
;                   SBT             : Total brightness for the given model
;                   SBP             : Polarized brightness for the given model
;                   SNE             : Total electron density for the given model
;  swire : list of structures for each view containing the projection of each model wireframe.
;           VIEWNAME                        : Name of the view
;           PROJECTEDWIREFRAMES[nbmodels]   : list of structures for each model
;               MODELNAME                   : name of the model
;               INSTANCEID                  : instance number of the model
;               PROJPOINTS                  : array of [2,nbpoints] x,y image coordinates 
;                                             of the projected wireframe for that model
;  ocout : list of structures for each model containing the 
;          array[3,nbpoints] x,y,z coordinates of the wireframe.
;           MODELNAME               : name of the model
;           INSTANCEID              : instance number of the model
;           CLOUD                   : array[3,nbpoints] x,y,z coordinates of the wireframe (not rotated).
;  ldisp : list of structures containing the data image and projeted wireframe for each view
;           VIEWNAME                : name of the view
;           IM                      : image in RGB channels
;
; CALL EXAMPLE:
;  sa0=sccreadfits('/net/cronus/opt/secchi/lz/L0/a/img/cor2/20081212/20081212_073754_d4c2A.fts')
;  sb0=sccreadfits('/net/cronus/opt/secchi/lz/L0/b/img/cor2/20081212/20081212_073754_d4c2B.fts')
;  sa=sccreadfits('/net/cronus/opt/secchi/lz/L0/a/img/cor2/20081212/20081212_140754_d4c2A.fts',hdrA)
;  sb=sccreadfits('/net/cronus/opt/secchi/lz/L0/b/img/cor2/20081212/20081212_140754_d4c2B.fts',hdrB)
;  ima=bytscl(alog10(rebin(sa-sa0,512,512) > 1e4 < 1e6))
;  imb=bytscl(alog10(rebin(sb-sb0,512,512) > 1e4 < 1e6))
;  pim=[ptr_new(ima),ptr_new(imb)]  ; or pim=list(ima,imb)
;  phdr=[ptr_new(hdrA),ptr_new(hdrB)]  ; or phdr=list(hdrA,hdrB)
;  rtcloudwidget,pim=pim,phdr=phdr
;
;  rtcloudwidget,pim=pim,phdr=phdr,/demo

; COMMENTS:
;  For additional information and tutorial, see:
;  http://secchi.nrl.navy.mil/wiki/ -> Data Processing and Analysis -> Solar Corona Ray-Tracing Software
;-

; ------ Unique event handler
pro rtcloudwidget_event_handler,ev
common common_de_merde,pmaingui

uname=widget_info(ev.id,/uname)

if uname eq 'quit' then begin
	; ---- tchao !
	; ---- broadcast save init values
	pmaingui->saveInitVal
	widget_control,ev.top,/destroy
endif else pmaingui->eventHandler,ev,uname

return
end


; ---- execute once, when the GUI is realized
pro rtcloudwidget_on_realize,id
common common_de_merde,pmaingui
(pmaingui->getrtmodellist())->updateAllCloudColorBox

return
end


pro rtcloudwidget,$
        pim=pim,$
        phdr=phdr,$
        itool=itool,$
        ssim=ssimout,$
        sgui=sguiout,$
        reset=reset,$
        swire=swire,$
        ocout=ocout,$
        ldisp=ldisp,$
        demo=demo

itreset,/no_prompt

; ---- demo mode
if keyword_set(demo) then begin
    rtinitenv
    sdemo = readstruct(getenv('RT_DATAPATH') + path_sep() + 'rtsccguiclouddemodata.xdr', /ignorestructnames)
    pim = list(sdemo.ima, sdemo.imb, bytscl(sdemo.imeuvia), bytscl(sdemo.imeuvib))
    phdr = list(sdemo.hdra, sdemo.hdrb, sdemo.hdreuvia, sdemo.hdreuvib)
endif

; ---- init view display
nbview = n_elements(pim)
view = objarr(nbview)

; -- use different display depending on the IDL version
if keyword_set(itool) then rtviewObjName='rtviewitool' else rtviewObjName='rtview'

inputtype = typename(pim)

for i = 0l,nbview-1 do begin
    if inputtype eq 'POINTER' then begin
    	view[i] = obj_new(rtviewObjName,$
    	                 *(pim[i]),$
    	                 *(phdr[i]),$
    	                 i,$
    	                 location=[(500*i) mod 1500l,500l * ((500*i) / 1500)])
    endif else begin
        ; -- assume that the input is a list of images and headers
        view[i] = obj_new(rtviewObjName,$
                            pim[i],$
                            phdr[i],$
                            i,$
                            location=[(500*i) mod 1500l,500l * ((500*i) / 1500)])
    endelse
endfor


; ---- Create main GUI
common common_de_merde,pmaingui
pmaingui = obj_new('rtwcloudmaingui', view=view, reset=reset)

; ---- run it !
widget_control, pmaingui->getWbase(), /realize, notify_realize='rtcloudwidget_on_realize'
xmanager, 'rtwcloudmaingui_gui', pmaingui->getWbase(), event_handler='rtcloudwidget_event_handler'

; ---- fill out output structures
; -- simulated images
lview = list()
for i = 0, nbview-1 do begin
    lview.add,view[i]->getPssim()
endfor
ssimout = {nbview:nbview,lview:lview}

; -- images with wireframes
ldisp = list()
for i = 0, nbview-1 do begin
    viewname = view[i]->getTitle()
    disp = view[i]->getDisplayedView()
    ldisp.add, {viewname:viewname, im:disp}
endfor

; -- GUI parameters
sguiout = pmaingui->getSgui()

; -- wireframe images
swire = list()
for i = 0, nbview-1 do begin
    viewname = view[i]->getTitle()
    swire.add, {viewname:viewname, projectedwireframes:view[i]->getPswire()}
endfor

; -- model 3D cloud of points

ocout = (pmaingui->getRtmodellist())->getClouds()

; ---- oy oy destroy !
obj_destroy, view
obj_destroy, pmaingui

return
end


; $Log: rtcloudwidget.pro,v $
; Revision 1.12  2012/03/12 15:45:32  thernis
; Fix bad merge during previous commit.
;
; Revision 1.11  2012/03/12 13:55:55  thernis
; Cleanup unuseful code
;
; Revision 1.10  2011-08-12 15:06:49  thernis
; - Add comment in header description
; - Implement demo mode
;
; Revision 1.9  2011-08-10 17:16:01  thernis
; - List of images is now also accepted as input
; - Export displayed views
; - Fix problem with simulated view export
; - Implement GUI reset keyword
;
; Revision 1.8  2011-08-09 21:52:30  thernis
; Implement output structures sgui, swire, ocout
;
; Revision 1.7  2011-07-20 18:01:54  thernis
; Implement regular object display instead of itools. Itools display is now optional.
;
; TODO:
;  - export images
;  - zoom
;  - change symbol
