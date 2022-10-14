;+
; $Id: rtsccguicloud.pro,v 1.33 2011/09/23 19:08:57 thernis Exp $
;
; PURPOSE:
;  Find the position of a flux rope CME using the GCS model and a GUI
;
; INPUTS:
;  ima,imb : secchi images SC A and B
;  hdra,hdrb : secchi header images A and B
;  imdispsize : size of the windows to display the images [sx,sy]:
;               default 512 x 512
;  maxheight : set the maximum value for the height slider (default is 20 Rsun). Useful when dealing with HI images.
;  imeuvia,imeuvib : pair of EUVI images on which the foot points of the model will be displayed.
;  hdreuvia,hdreuvib : fits header structure corresponding to the pair of EUVI images.
;  demo : run in demo mode: will load some example images.
;  showanaglyph : will display in anaglyph mode: red-blue glasses are required. 
;                 Only useful when the separation is less than 10 degrees.
;  eruptiondatein : date of the stereo image pair used as the reference time. 
;                   Useful to compensate solar rotation when fitting a sequence of images.
;  imlasco,hdrlasco : LASCO image and fits header. Add a third eye on the Sun to better constrain the fit.
;  forceinit : set this keyword to force reseting the different GUI parameters.
;  sparaminit : Allow to initialize any of the GUI parameters.
;  imabsunin,imbbsunin : Pass the stereo pair A and B calibrated in Bsun. Useful for fitting the Ne.
;  admin : Run in administrator mode. Allow to sync the GUI with the CME modeling data base.
;          This feature only works within NRL.
;
; OUTPUTS:
;  ssim : returns a structure containing the rendered simulated images: total B polarized B, total electron content.
;  sgui : returns a structure containing all the different parameters of the GUI.
;    sgui.lon : longitude Carrington.
;    sgui.lat : latitude.
;    sgui.rot : tilt angle or rotation around the model axis of symmetry. 0 is parallel to the equator.
;    sgui.han : half angle between the model's feet.
;    sgui.hgt : height, in Rsun.
;    sgui.rat : aspect ratio
;  swire : returns the wireframe, projected in each views.
;    swire.sa.im : wireframe projected in view A
;    swire.sb.im : wireframe projected in view B
;  ocout : returns the wireframe: a set of x,y,z point coordinates.
;
; CALL EXAMPLE:
;  rtsccguicloud,ima,imb,hdra,hdrb
;
; COMMENTS:
;  For additional information and tutorial, see:
;  http://secchi.nrl.navy.mil/wiki/ -> Data Processing and Analysis -> Solar Corona Ray-Tracing Software
;
;-

; ---- compute the feet height, a parameter of the model
function rtsccguicloud_calcfeetheight,leadedgeheight,k,hang
return,leadedgeheight*(1.-k)*cos(hang)/(1.+sin(hang))
end

; ---- display EUVI images
pro rtsccguicloud_dispeuviim,obsrvtry,hdr,im
common com_sgui,oc,sa,sb,sgui,ssim,firstrun,sinit,slasco

nlposabscntr=neutlinecarposabs(sgui.lon,sgui.lat,0.,0.)

nlposabsl2=neutlinecarposabs(sgui.lon,sgui.lat,sgui.han,sgui.rot)
nlposabsl1=neutlinecarposabs(sgui.lon,sgui.lat,sgui.han*0.5,sgui.rot)

nlposabsr2=neutlinecarposabs(sgui.lon,sgui.lat,-sgui.han,sgui.rot)
nlposabsr1=neutlinecarposabs(sgui.lon,sgui.lat,-sgui.han*0.5,sgui.rot)

rtcloud,[[nlposabsl2],[nlposabsl1],[nlposabscntr],[nlposabsr1],[nlposabsr2]],imsize=sgui.imdispsize,scchead=hdr,/flistout,listout=listout,/quiet,listbehind=listbehind

wnd,obsrvtry+4,im,title=(obsrvtry eq 1 ? 'EUVI A' : 'EUVI B')

plots,listout[*,0],/device,psym=1,color=(listbehind[0] eq 1 ? 254 : 255)
plots,listout[*,1],/device,psym=1,color=(listbehind[1] eq 1 ? 254 : 255)
plots,listout[*,2],/device,psym=2,color=(listbehind[2] eq 1 ? 254 : 255)
plots,listout[*,3],/device,psym=1,color=(listbehind[3] eq 1 ? 254 : 255)
plots,listout[*,4],/device,psym=1,color=(listbehind[4] eq 1 ? 254 : 255)


end


; ---- display images
pro rtsccguicloud_displayimages
common com_sgui

wnd,0, bytscl((sgui.ima + sa.im * (sgui.flagwireonoff ? 255 : 0))<255,top=(sgui.flagwireonoff ? 255 : 254)),title='A',/tv
wnd,1,bytscl((sgui.imb + sb.im * (sgui.flagwireonoff ? 255 : 0))<255,top=(sgui.flagwireonoff ? 255 : 254)),title='B',/tv
if sgui.flaglasco then wnd,20,bytscl((sgui.imlasco + slasco.im * (sgui.flagwireonoff ? 255 : 0))<255,top=(sgui.flagwireonoff ? 255 : 254)),title='LASCO',/tv

if size(sgui.hdreuvia,/type) eq 8 then rtsccguicloud_dispeuviim,1,sgui.hdreuvia,sgui.imeuvia
if size(sgui.hdreuvib,/type) eq 8 then rtsccguicloud_dispeuviim,2,sgui.hdreuvib,sgui.imeuvib

if sgui.showanag then begin
    wset,10
    tv,bytscl(sgui.anag,top=254),true=1
endif


end


; ---- recompute the cloud
pro rtsccguicloud_recalccloud
common com_sgui
oc=cmecloud(sgui.han,sgui.hgt,sgui.nbvertaxisp,sgui.nbvertcirup,sgui.rat,sgui.nbvertshell,/distjuncisleadingedge)

end


; ---- compute the neang parameter
function rtsccguicloud_calcneang,lon,lat,rot,carrlonshiftdeg,carrstonyshiftdeg

; print,'lon : ',lon
; print,'carrlonshiftdeg : ',carrlonshiftdeg
; print,'carrstonyshiftdeg : ',carrstonyshiftdeg

;return,[lon+(carrlonshiftdeg-carrstonyshiftdeg)*!dtor,lat,rot]
return,[lon+carrlonshiftdeg*!dtor,lat,rot]
end

; --- recompute the cloud projection
pro rtsccguicloud_calccloudproj
common com_sgui

neang=rtsccguicloud_calcneang(sgui.lon,sgui.lat,sgui.rot,sgui.carrlonshiftdeg,sgui.carrstonyshiftdeg)
rtcloud,oc,sa,imsize=sgui.imdispsize,scchead=sgui.hdra,neang=neang,/fclip,/quiet

rtcloud,oc,sb,imsize=sgui.imdispsize,scchead=sgui.hdrb,neang=neang,/fclip,/quiet

if sgui.flaglasco then rtcloud,oc,slasco,imsize=sgui.imdispsize,scchead=sgui.shdrlasco,neang=neang,/fclip,/quiet

if sgui.showanag then begin
    rtcloud,oc,saanag,imsize=sgui.imdispsize,scchead=sgui.hdra,/quiet,neang=neang
    rtcloud,oc,sbanag,imsize=sgui.imdispsize,scchead=sgui.hdrbanag,/quiet,neang=neang
    
    sgui.anag=mkanaglyph(bytscl(saanag.im),bytscl(sbanag.im))
endif



end


; -------- event handlers
; ------ deal with slider changes
pro event_slidermoved,ev
common com_sgui

; ---- get event name
uname=strmid(widget_info(ev.id,/uname),2,3,/reverse )

; ---- update the struct according to the slide that has been moved
widget_control,ev.id,get_value=val
m=where(tag_names(sgui) eq strupcase(uname),cnt)
if cnt eq 0 or cnt gt 1 then message,'Cannot find the tag !'
sgui.(m[0])=val*(uname eq 'han' or uname eq 'lat' or uname eq 'lon' or uname eq 'rot' ? !dtor : 1.)
if uname eq 'lon' then sgui.lon=sgui.lon-sgui.carrlonshiftdeg*!dtor+sgui.carrstonyshiftdeg*!dtor

; ---- recompute the cloud if parameters changed
if uname eq 'hgt' or uname eq 'rat' or uname eq 'han' then rtsccguicloud_recalccloud
; ---- recompute the projections
rtsccguicloud_calccloudproj
; ---- display
rtsccguicloud_displayimages

end


; ---- recalc of the cloud requested
pro event_changecloud,ev
common com_sgui

; ---- get event name
uname=strmid(widget_info(ev.id,/uname),6)

; ---- update the struct occording to the slide that has been moved
widget_control,ev.id,get_value=val
m=where(tag_names(sgui) eq strupcase(uname),cnt)
if cnt eq 0 or cnt gt 1 then message,'Cannot find the tag !'
sgui.(m[0])=val

; ---- recalculate the cloud
rtsccguicloud_recalccloud
; ---- recompute the projections
rtsccguicloud_calccloudproj
; ---- display
rtsccguicloud_displayimages

end


pro event_msid,ev
return
end


; ---- recalc of the cloud requested
pro event_dbsync,ev
common com_sgui

; ---- get event name
;uname=strmid(widget_info(ev.id,/uname),6)

; ---- get the msid
id_msid=widget_info(ev.top,find_by_uname='wtext_msid')
widget_control,id_msid,get_value=msidval

msidval=strtrim(msidval,2)
if msidval eq '' then begin
	print,'Please enter an msid in the box'
	return
endif


; ---- query the DB
qr="select evid,msid,date_obsa,lon,lat,rot,han,hgt,rat from measure where msid="+msidval+""
dbqrsecchi2struct,qr,sms,norow,db='synomap'

if norow then begin
	print,'No record found for msid '+msidval
	return
endif


; ---- update the different sliders
; -- lon
id=widget_info(ev.top,find_by_uname='wslider_lon')
; - is it carrington or stonyhurst ?
;   sms.lon is in Carr, and deg
;   if the lonslider display Carr, just directly update
;   else convert in Stony before updating
if sgui.carrorstony ne 0 then begin
	; -- the display is in Stonyhurst: convert sms.lon to stonyhurst
	lon=sms.lon-sgui.hdra.crln_obs+sgui.hdra.hgln_obs
endif else lon=sms.lon

widget_control,id,set_value=lon
event_slidermoved,{id:id}

; -- lat
id=widget_info(ev.top,find_by_uname='wslider_lat')
widget_control,id,set_value=sms.lat
event_slidermoved,{id:id}

; -- rot
id=widget_info(ev.top,find_by_uname='wslider_rot')
widget_control,id,set_value=sms.rot
event_slidermoved,{id:id}

; -- han
id=widget_info(ev.top,find_by_uname='wslider_han')
widget_control,id,set_value=sms.han
event_slidermoved,{id:id}

; -- hgt
id=widget_info(ev.top,find_by_uname='wslider_hgt')
widget_control,id,set_value=sms.hgt
event_slidermoved,{id:id}

; -- rat
id=widget_info(ev.top,find_by_uname='wslider_rat')
widget_control,id,set_value=sms.rat
event_slidermoved,{id:id}

return
end








; ---- generate thomson scaterring views for A n B
pro event_generateview,ev
common com_sgui

neang=rtsccguicloud_calcneang(sgui.lon,sgui.lat,sgui.rot,sgui.carrlonshiftdeg,sgui.carrstonyshiftdeg)
hgt=rtsccguicloud_calcfeetheight(sgui.hgt,sgui.rat,sgui.han)
mp=[1.5,sgui.han,hgt,sgui.rat,sgui.nel,0.,0.,0.,sgui.sigin,sgui.sigout]

print,'Computing view SC-A'
rtraytracewcs,sbta,sbpa,snea,modelid=54,imsize=sgui.imdispsize,losrange=sgui.losrange,modparam=mp,neang=neang,scchead=sgui.hdra,losnbp=sgui.losnbp,/progressonly

print,'Computing view SC-B'
rtraytracewcs,sbtb,sbpb,sneb,modelid=54,imsize=sgui.imdispsize,losrange=sgui.losrange,modparam=mp,neang=neang,scchead=sgui.hdrb,losnbp=sgui.losnbp,/progressonly


if sgui.flaglasco then begin
	print,'Computing view LASCO'
	rtraytracewcs,sbtlasco,sbplasco,snelasco,modelid=54,imsize=sgui.imdispsize,losrange=sgui.losrange,modparam=mp,neang=neang,scchead=sgui.shdrlasco,losnbp=sgui.losnbp,/progressonly
endif
print,'Done'


wnd,2,bytscl(alog10(sbta.im > 1e-14),top=254),title='SC-A: Simu',/tv
wnd,3,bytscl(alog10(sbtb.im > 1e-14),top=254),title='SC-B: Simu',/tv
if sgui.flaglasco then wnd,21,bytscl(alog10(sbtlasco.im > 1e-14),top=254),title='LASCO: Simu',/tv

if ~sgui.flaglasco then ssim={a:{sbt:sbta,sbp:sbpa,sne:snea},b:{sbt:sbtb,sbp:sbpb,sne:sneb},mp:mp,neang:neang} else ssim={a:{sbt:sbta,sbp:sbpa,sne:snea},b:{sbt:sbtb,sbp:sbpb,sne:sneb},lasco:{sbt:sbtlasco,sbp:sbplasco,sne:snelasco},mp:mp,neang:neang}

end


; ----- update the simu parameters
pro rtsccguicloud_updatesimuinfo,ev
common com_sgui
idtab=widget_info(ev.id,/parent)
idlabel=widget_info(idtab,find_by_uname='wlabel_msg3')
widget_control,idlabel,set_value='['+strtrim(sgui.losrange[0],2)+','+strtrim(sgui.losrange[1],2)+'],'+strtrim(sgui.losnbp,2)
end
pro event_updatelosrangemin,ev
common com_sgui
widget_control,ev.id,get_value=val
sgui.losrange[0]=long(val)
rtsccguicloud_updatesimuinfo,ev
end
pro event_updatelosrangemax,ev
common com_sgui
widget_control,ev.id,get_value=val
sgui.losrange[1]=long(val)
rtsccguicloud_updatesimuinfo,ev
end
pro event_updatelosnbp,ev
common com_sgui
widget_control,ev.id,get_value=val
sgui.losnbp=long(val)
rtsccguicloud_updatesimuinfo,ev
end

pro event_updatenel,ev
common com_sgui
widget_control,ev.id,get_value=val
sgui.nel=float(val)
rtsccguicloud_updatesimuinfo,ev
end
pro event_updatesigin,ev
common com_sgui
widget_control,ev.id,get_value=val
sgui.sigin=float(val)
rtsccguicloud_updatesimuinfo,ev
end
pro event_updatesigout,ev
common com_sgui
widget_control,ev.id,get_value=val
sgui.sigout=float(val)
rtsccguicloud_updatesimuinfo,ev
end


; ---- sensitivity analysis
pro event_updatesensitrange,ev
common com_sgui
widget_control,ev.id,get_value=val
sgui.sensitrange=float(val)
return
end
; ---- longitude
pro event_sensitlon,ev
common com_sgui

idtab=widget_info(ev.id,/parent)

id=widget_info(idtab,find_by_uname='wtext_sensitrange')
widget_control,id,get_value=val
sensitrange=float(val)

cloudautofit_computewireproj,sgui.han,sgui.hgt,sgui.rat,sgui.lon,sgui.lat,sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp
goodfit=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

;goodfit=sgui.error

boundgf=goodfit*(1.-sensitrange)

step=0.1*!dtor ; step in rad
maxit=250l

; -- scan positive
it=0l
repeat begin
print,'+',it
it++

cloudautofit_computewireproj,sgui.han,sgui.hgt,sgui.rat,sgui.lon+(float(it)*step),sgui.lat,sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp

error=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

endrep until (error le boundgf || it gt maxit)

plonit=it
plonerror=error


; -- scan negative
it=0l
repeat begin
print,'-',it
it++

cloudautofit_computewireproj,sgui.han,sgui.hgt,sgui.rat,sgui.lon-(float(it)*step),sgui.lat,sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp

error=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

endrep until (error le boundgf || it gt maxit)

mlonit=it
mlonerror=error

; ---- update text
id=widget_info(idtab,find_by_uname='wlabel_sensit_lon')
widget_control,id,set_value='lon: +'+strtrim(float(plonit)*step*!radeg,2)+' -'+strtrim(float(mlonit)*step*!radeg,2)+' '+strtrim(plonit,2)+' '+strtrim(mlonit,2)+' '+strtrim(plonerror,2)+' '+strtrim(mlonerror,2)+' '+string(sensitrange*100,form='(I2)')

return
end
; ---- latitude
pro event_sensitlat,ev
common com_sgui

idtab=widget_info(ev.id,/parent)

id=widget_info(idtab,find_by_uname='wtext_sensitrange')
widget_control,id,get_value=val
sensitrange=float(val)

cloudautofit_computewireproj,sgui.han,sgui.hgt,sgui.rat,sgui.lon,sgui.lat,sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp
goodfit=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

;goodfit=sgui.error

boundgf=goodfit*(1.-sensitrange)

step=0.1*!dtor ; step in rad
maxit=100l

; -- scan positive
it=0l
repeat begin
print,'+',it
it++

cloudautofit_computewireproj,sgui.han,sgui.hgt,sgui.rat,sgui.lon,sgui.lat+(float(it)*step),sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp

error=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

endrep until (error le boundgf || it gt maxit)

pit=it
perror=error


; -- scan negative
it=0l
repeat begin
print,'-',it
it++

cloudautofit_computewireproj,sgui.han,sgui.hgt,sgui.rat,sgui.lon,sgui.lat-(float(it)*step),sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp

error=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

endrep until (error le boundgf || it gt maxit)

mit=it
merror=error

; ---- update text
id=widget_info(idtab,find_by_uname='wlabel_sensit_lat')
widget_control,id,set_value='lat: +'+strtrim(float(pit)*step*!radeg,2)+' -'+strtrim(float(mit)*step*!radeg,2)+' '+strtrim(pit,2)+' '+strtrim(mit,2)+' '+strtrim(perror,2)+' '+strtrim(merror,2)+' '+string(sensitrange*100,form='(I2)')

return
end

; ---- rotation
pro event_sensitrot,ev
common com_sgui

idtab=widget_info(ev.id,/parent)

id=widget_info(idtab,find_by_uname='wtext_sensitrange')
widget_control,id,get_value=val
sensitrange=float(val)

cloudautofit_computewireproj,sgui.han,sgui.hgt,sgui.rat,sgui.lon,sgui.lat,sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp
goodfit=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

;goodfit=sgui.error

boundgf=goodfit*(1.-sensitrange)

step=1.*!dtor ; step in rad
maxit=50l

; -- scan positive
it=0l
repeat begin
print,'+',it
it++

cloudautofit_computewireproj,sgui.han,sgui.hgt,sgui.rat,sgui.lon,sgui.lat,sgui.rot+(float(it)*step),{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp

error=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

endrep until (error le boundgf || it gt maxit)

pit=it
perror=error


; -- scan negative
it=0l
repeat begin
print,'-',it
it++

cloudautofit_computewireproj,sgui.han,sgui.hgt,sgui.rat,sgui.lon,sgui.lat,sgui.rot-(float(it)*step),{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp

error=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

endrep until (error le boundgf || it gt maxit)

mit=it
merror=error

; ---- update text
id=widget_info(idtab,find_by_uname='wlabel_sensit_rot')
widget_control,id,set_value='rot: +'+strtrim(float(pit)*step*!radeg,2)+' -'+strtrim(float(mit)*step*!radeg,2)+' '+strtrim(pit,2)+' '+strtrim(mit,2)+' '+strtrim(perror,2)+' '+strtrim(merror,2)+' '+string(sensitrange*100,form='(I2)')

return
end


; ---- height
pro event_sensithgt,ev
common com_sgui

idtab=widget_info(ev.id,/parent)

id=widget_info(idtab,find_by_uname='wtext_sensitrange')
widget_control,id,get_value=val
sensitrange=float(val)

cloudautofit_computewireproj,sgui.han,sgui.hgt,sgui.rat,sgui.lon,sgui.lat,sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp
goodfit=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

;goodfit=sgui.error

boundgf=goodfit*(1.-sensitrange)

step=0.05 ; step in Rsun
maxit=50l

; -- scan positive
it=0l
repeat begin
print,'+',it
it++

cloudautofit_computewireproj,sgui.han,sgui.hgt+(float(it)*step),sgui.rat,sgui.lon,sgui.lat,sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp

error=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

endrep until (error le boundgf || it gt maxit)

pit=it
perror=error


; -- scan negative
it=0l
repeat begin
print,'-',it
it++

cloudautofit_computewireproj,sgui.han,sgui.hgt-(float(it)*step),sgui.rat,sgui.lon,sgui.lat,sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp

error=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

endrep until (error le boundgf || it gt maxit)

mit=it
merror=error

; ---- update text
id=widget_info(idtab,find_by_uname='wlabel_sensit_hgt')
widget_control,id,set_value='hgt: +'+strtrim(float(pit)*step,2)+' -'+strtrim(float(mit)*step,2)+' '+strtrim(pit,2)+' '+strtrim(mit,2)+' '+strtrim(perror,2)+' '+strtrim(merror,2)+' '+string(sensitrange*100,form='(I2)')

return
end



; ---- aspect ratio
pro event_sensitrat,ev
common com_sgui

idtab=widget_info(ev.id,/parent)

id=widget_info(idtab,find_by_uname='wtext_sensitrange')
widget_control,id,get_value=val
sensitrange=float(val)

cloudautofit_computewireproj,sgui.han,sgui.hgt,sgui.rat,sgui.lon,sgui.lat,sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp
goodfit=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

;goodfit=sgui.error

boundgf=goodfit*(1.-sensitrange)

step=0.01 ; step
maxit=50l

; -- scan positive
it=0l
repeat begin
print,'+',it
it++

cloudautofit_computewireproj,sgui.han,sgui.hgt,sgui.rat+(float(it)*step),sgui.lon,sgui.lat,sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp

error=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

endrep until (error le boundgf || it gt maxit || sgui.rat+(float(it+1)*step) ge 0.8)

pit=it
perror=error


; -- scan negative
it=0l
repeat begin
print,'-',it
it++

cloudautofit_computewireproj,sgui.han,sgui.hgt,sgui.rat-(float(it)*step),sgui.lon,sgui.lat,sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp

error=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

endrep until (error le boundgf || it gt maxit || sgui.rat-(float(it+1)*step) le 0.01)

mit=it
merror=error

; ---- update text
id=widget_info(idtab,find_by_uname='wlabel_sensit_rat')
widget_control,id,set_value='rat: +'+strtrim(float(pit)*step,2)+' -'+strtrim(float(mit)*step,2)+' '+strtrim(pit,2)+' '+strtrim(mit,2)+' '+strtrim(perror,2)+' '+strtrim(merror,2)+' '+string(sensitrange*100,form='(I2)')

return
end



; ---- half angle
pro event_sensithan,ev
common com_sgui

idtab=widget_info(ev.id,/parent)

id=widget_info(idtab,find_by_uname='wtext_sensitrange')
widget_control,id,get_value=val
sensitrange=float(val)

cloudautofit_computewireproj,sgui.han,sgui.hgt,sgui.rat,sgui.lon,sgui.lat,sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp
goodfit=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

;goodfit=sgui.error

boundgf=goodfit*(1.-sensitrange)

step=1.*!dtor ; step in rad
maxit=100l

; -- scan positive
it=0l
repeat begin
print,'+',it
it++

cloudautofit_computewireproj,sgui.han+(float(it)*step),sgui.hgt,sgui.rat,sgui.lon,sgui.lat,sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp

error=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

endrep until (error le boundgf || it gt maxit || sgui.han+(float(it+1)*step) ge 50.*!dtor)

pit=it
perror=error


; -- scan negative
it=0l
repeat begin
print,'-',it
it++

cloudautofit_computewireproj,sgui.han-(float(it)*step),sgui.hgt,sgui.rat,sgui.lon,sgui.lat,sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp

error=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

endrep until (error le boundgf || it gt maxit || sgui.han-(float(it+1)*step) le 0.)

mit=it
merror=error

; ---- update text
id=widget_info(idtab,find_by_uname='wlabel_sensit_han')
widget_control,id,set_value='han: +'+strtrim(float(pit)*step*!radeg,2)+' -'+strtrim(float(mit)*step*!radeg,2)+' '+strtrim(pit,2)+' '+strtrim(mit,2)+' '+strtrim(perror,2)+' '+strtrim(merror,2)+' '+string(sensitrange*100,form='(I2)')

return
end





; ---- format the SC position, in the right coordinate system
function rtsccguicloud_formatscpos,hdr,carrorstony
if carrorstony eq 0 then begin
	; -- display Carrington coords
	lonobs=hdr.crln_obs
	latobs=hdr.crlt_obs
endif else begin
	; -- display Stonyhurst coords
	lonobs=hdr.hgln_obs
	latobs=hdr.hglt_obs
endelse

sc=strmid(hdr.obsrvtry,0,1,/reverse)
return,'SC-'+sc+' LON: '+string(lonobs,form='(F6.1)')+' , LAT: '+string(latobs,form='(F5.1)')
end


; -- compute the corresponding carrington longitude shift
pro rtsccguicloud_computecarrlonshift_nocommon,hdra,eruptiondate,carrorstony,carrlonshiftdeg,carrstonyshiftdeg
; -- carrington rotation period is 27.2753 days
carrlonshiftdeg=-mjd2day(submjd(anytim(hdra.date_obs,/mjd),anytim(eruptiondate,/mjd)))/27.2753*360.

if carrorstony eq 1 then carrstonyshiftdeg=hdra.crln_obs-hdra.hgln_obs else carrstonyshiftdeg=0.

;carrlonshiftdeg+=carrstonyshiftdeg

return
end

pro rtsccguicloud_computecarrlonshift
common com_sgui
rtsccguicloud_computecarrlonshift_nocommon,sgui.hdra,sgui.eruptiondate,sgui.carrorstony,carrlonshiftdeg,carrstonyshiftdeg
sgui.oldcarrlonshiftdeg=sgui.carrlonshiftdeg
sgui.carrlonshiftdeg=carrlonshiftdeg
sgui.carrstonyshiftdeg=carrstonyshiftdeg
return
end


function rtsccguicloud_calclonslidervalues,lon,carrlonshiftdeg,carrstonyshiftdeg

sliderlon=lon*!radeg+carrlonshiftdeg-carrstonyshiftdeg
if sliderlon lt 0. then sliderlon+=360.
if sliderlon ge 360. then sliderlon-=360.

return,sliderlon
end


pro rtsccguicloud_updatecarrlonshiftandslider
common com_sgui
; -- compute corresponding carrington longitude shift
rtsccguicloud_computecarrlonshift
; -- update the longitude slider value
r=widget_info(sgui.wbase,find_by_uname='wslider_lon')
widget_control,r,get_value=val

sliderlon=rtsccguicloud_calclonslidervalues(sgui.lon,sgui.carrlonshiftdeg,sgui.carrstonyshiftdeg)
widget_control,r,set_value=[sliderlon,0.,360.]

; ---- recompute the projections
rtsccguicloud_calccloudproj
; ---- display
rtsccguicloud_displayimages
end


; ---- use different eruption date than the data header
pro event_updateeruptiondate,ev
common com_sgui
widget_control,ev.id,get_value=val
sgui.eruptiondate=strtrim(val,2)
; -- update carrlonshift value and the longitude slider
rtsccguicloud_updatecarrlonshiftandslider
end


; ---- switch coord system
function event_carrstony,ev
common com_sgui

; -- switch the flag
oldcarrosstony=sgui.carrorstony
sgui.carrorstony=(ev.value eq 'stony' && ev.select eq 1)

; -- return if no change
if oldcarrosstony eq sgui.carrorstony then return,1b

; -- update all the widget coordinate outputs
; - longitude slider
rtsccguicloud_updatecarrlonshiftandslider

; - SC A position
idlabel=widget_info(sgui.wbase,find_by_uname='wlabel_scapos')
widget_control,idlabel,set_value=rtsccguicloud_formatscpos(sgui.hdra,sgui.carrorstony)
; - SC B position
idlabel=widget_info(sgui.wbase,find_by_uname='wlabel_scbpos')
widget_control,idlabel,set_value=rtsccguicloud_formatscpos(sgui.hdrb,sgui.carrorstony)


return,1b
end


; ---- wire frame on/off
pro event_wireonoff,ev
common com_sgui
sgui.flagwireonoff=~sgui.flagwireonoff
widget_control,ev.id,set_value=(sgui.flagwireonoff ? 'Wire Off' : 'Wire On')
rtsccguicloud_displayimages
return
end


; ---- interactively draw the contours on A and B
pro event_drawcontour,ev
common com_sgui
idlabel=widget_info(sgui.wbase,find_by_uname='wlabel_msgcontour')

wnd,16,bytscl(sgui.ima,top=254),/tv,title='Draw Contour A'
wnd,17,bytscl(sgui.imb,top=254),/tv,title='Draw Contour B'

widget_control,idlabel,set_value='Draw contour in view A'
repeat begin
	wnd,16,bytscl(sgui.ima,top=254),/tv,title='Draw Contour A'
	contourA=draw(/quiet)
	ans=DIALOG_MESSAGE('Contour A ok ?',/question,/center)
endrep until (ans eq 'Yes')
widget_control,idlabel,set_value='Draw contour in view B'
repeat begin
	wnd,17,bytscl(sgui.imb,top=254),/tv,title='Draw Contour B'
	contourB=draw(/quiet)
	ans=DIALOG_MESSAGE('Contour B ok ?',/question,/center)
endrep until (ans eq 'Yes')
widget_control,idlabel,set_value='Computing contour images and masks...'

; ---- compute distance map for the A and B contours
hglghtcontfactor=1.
cloudautofit_computecontourmap,contourA,contourB,sgui.hdra,sgui.hdrb,imdistA,imdistB,masksunA,masksunB,sumA,sumB,imcontA,imcontB,hglghtcontfactor=hglghtcontfactor

sgui.imdistA=imdistA
sgui.imdistB=imdistB
sgui.masksunA=ptr_new(masksunA)
sgui.masksunB=ptr_new(masksunB)
sgui.sumA=sumA
sgui.sumB=sumB
sgui.contourA=ptr_new(contourA)
sgui.contourB=ptr_new(contourB)

; ---- compute smaller resolution dist map to speed up the auto fit
imsizeautofit=long(512.*2.^sgui.autofitresolution)

imcontAreb=rebin(float(imcontA),imsizeautofit,imsizeautofit)
ma=where(imcontAreb gt 0.,cntA)
imdistAreb=rebin(imdistA,imsizeautofit,imsizeautofit)
masksunAreb=rebin(masksunA,imsizeautofit,imsizeautofit)
if cnta gt 0 then begin
	imcontAreb[ma]=1.
	imdistAreb[ma]=hglghtcontfactor
endif

imcontBreb=rebin(float(imcontB),imsizeautofit,imsizeautofit)
mb=where(imcontBreb gt 0.,cntB)
imdistBreb=rebin(imdistB,imsizeautofit,imsizeautofit)
masksunBreb=rebin(masksunB,imsizeautofit,imsizeautofit)
if cntb gt 0 then begin
	imcontBreb[mb]=1.
	imdistBreb[mb]=hglghtcontfactor
endif


sgui.imdistAreb=imdistAreb
sgui.imdistBreb=imdistBreb
sgui.masksunAreb=ptr_new(masksunAreb)
sgui.masksunBreb=ptr_new(masksunBreb)
sgui.sumAreb=cntA
sgui.sumBreb=cntB

widget_control,idlabel,set_value='Done !'

return
end


pro event_updateprofiles,ev
common com_sgui
common com_imbsun,imabsun,imbbsun

idlabel=widget_info(sgui.wbase,find_by_uname='wlabel_msgprofiles')
widget_control,idlabel,set_value='Updating profiles...'

; ---- get profile from data
profdataA=prline(imabsun,sgui.ptprofa[*,0],sgui.ptprofa[*,1],512,sprofa)
profdataB=prline(imbbsun,sgui.ptprofb[*,0],sgui.ptprofb[*,1],512,sprofb)

; ---- get profile from simu
neang=rtsccguicloud_calcneang(sgui.lon,sgui.lat,sgui.rot,sgui.carrlonshiftdeg,sgui.carrstonyshiftdeg)
hgt=rtsccguicloud_calcfeetheight(sgui.hgt,sgui.rat,sgui.han)
mp=[1.5,sgui.han,hgt,sgui.rat,sgui.nel,0.,0.,0.,sgui.sigin,sgui.sigout]

; -- compute the ROI
set_plot,'z'
device,set_resolution=[512,512]
erase
plots,sgui.ptprofa[*,0],/device
plots,sgui.ptprofa[*,1],/device,/continue
roia=tvrd()
roia=dilate(roia,replicate(1,5,5))

erase
plots,sgui.ptprofb[*,0],/device
plots,sgui.ptprofb[*,1],/device,/continue
roib=tvrd()
roib=dilate(roib,replicate(1,5,5))

setx


print,'Computing profile SC-A'
rtraytracewcs,sbta,sbpa,snea,modelid=54,imsize=sgui.imdispsize,losrange=sgui.losrange,modparam=mp,neang=neang,scchead=sgui.hdra,losnbp=256,/quiet,roi=roia

profsimuA=prline(sbta.im,sgui.ptprofa[*,0],sgui.ptprofa[*,1],512)

print,'Computing profile SC-B'
rtraytracewcs,sbtb,sbpb,sneb,modelid=54,imsize=sgui.imdispsize,losrange=sgui.losrange,modparam=mp,neang=neang,scchead=sgui.hdrb,losnbp=256,/quiet,roi=roib

profsimuB=prline(sbtb.im,sgui.ptprofb[*,0],sgui.ptprofb[*,1],512)

wnd,28,700,900
!p.multi=[0,1,2]
plot,sprofa,profdataA,title='A',/ylog,yrange=[1e-12,1e-10]
oplot,sprofa,profsimuA

plot,sprofb,profdataB,title='B',/ylog,yrange=[1e-12,1e-10]
oplot,sprofb,profsimuB


!p.multi=0

widget_control,idlabel,set_value='Done...'

return
end

pro event_drawprofiles,ev
common com_sgui
idlabel=widget_info(sgui.wbase,find_by_uname='wlabel_msgprofiles')

wnd,26,bytscl(sgui.ima,top=254),/tv,title='Draw Profile A'
wnd,27,bytscl(sgui.imb,top=254),/tv,title='Draw Profile B'

widget_control,idlabel,set_value='Draw profile in view A'
repeat begin
	wnd,26,bytscl(sgui.ima,top=254),/tv,title='Draw Profile A'
	ptprofileA=clickline()
	ans=DIALOG_MESSAGE('Profile A ok ?',/question,/center)
endrep until (ans eq 'Yes')
widget_control,idlabel,set_value='Draw profile in view B'
repeat begin
	wnd,27,bytscl(sgui.imb,top=254),/tv,title='Draw Profile B'
	ptprofileB=clickline()
	ans=DIALOG_MESSAGE('Profile B ok ?',/question,/center)
endrep until (ans eq 'Yes')

sgui.ptprofa=ptprofileA
sgui.ptprofb=ptprofileB

; ---- update profiles
event_updateprofiles,ev

return
end








function rtsccguicloud_cloudautofit_error,p
common com_sgui
common com_autofit,sautofit

; ---- inc call counter
sautofit.iter++
;if sautofit.iter mod 50 eq 0 then print,'iter: ',sautofit.iter
widget_control,sautofit.idlabeliter,set_value='Iterations : '+strtrim(sautofit.iter,2)

; ---- extract the function parameters
lon=p[0]
lat=p[1]
hgt=p[2]

; ---- compute the wire-frame
cloudautofit_computewireproj,sgui.han,hgt,sgui.rat,lon,lat,sgui.rot,{resolution:sgui.autofitresolution,imsidesize:long(([512,512]*2.^sgui.autofitresolution)),hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunAreb,masksunB:sgui.masksunBreb},wireprojA,wireprojB,/nodisp

; ---- limit the research domain
prior_hgt=sigmoid(hgt,1.,2.,2.)*sigmoid(hgt,1.,25.,-2.)
prior_lat=float(lat ge -!pi/2. && lat le !pi/2.)
prior=prior_hgt*prior_lat

; ---- compute the error
error=-((total(wireprojA* sgui.imdistAreb)/sgui.sumAreb+total(wireprojB* sgui.imdistBreb)/sgui.sumBreb))/2.+(1.-prior)
widget_control,sautofit.idlabelerror,set_value='Fit : '+strtrim(-error*100,2)+' %'


return,error
end


; ---- evaluate the error using the user defined contour images
pro event_evalerror,ev,nodisp=nodisp
common com_sgui

if sgui.sumA eq 0. or sgui.sumB eq 0. then begin
	ans=DIALOG_MESSAGE('Please define the CME contours using the "Contour" tab. ',/center,/information)
	return
endif


cloudautofit_computewireproj,sgui.han,sgui.hgt,sgui.rat,sgui.lon,sgui.lat,sgui.rot,{resolution:0,imsidesize:sgui.imdispsize,hdra:sgui.hdra,hdrb:sgui.hdrb,masksunA:sgui.masksunA,masksunB:sgui.masksunB},wireprojA,wireprojB,/nodisp

error=((total(wireprojA* sgui.imdistA)/sgui.sumA+total(wireprojB* sgui.imdistB)/sgui.sumB))/2.

sgui.error=error

idlabel=widget_info(sgui.wbase,find_by_uname='wlabel_error')

widget_control,idlabel,set_value='Fit : '+strtrim(error*100,2)+' %'
if ~keyword_set(nodisp) then begin
	wnd,18,bytscl(wireprojA+ sgui.imdistA),title='A: Binary contour'
	wnd,19,bytscl(wireprojB+ sgui.imdistB),title='B: Binary contour'
endif
return
end

; ------ Auto Fit
pro event_autofit,ev
common com_sgui
common com_autofit,sautofit

if sgui.sumAreb eq 0. or sgui.sumBreb eq 0. then begin
	ans=DIALOG_MESSAGE('Please define the CME contours using the "Contour" tab. ',/center,/information)
	return
endif

idlabeliter=widget_info(sgui.wbase,find_by_uname='wlabel_autofit_iter')
idlabelerror=widget_info(sgui.wbase,find_by_uname='wlabel_autofit_error')
idlabelstatus=widget_info(sgui.wbase,find_by_uname='wlabel_autofit_status')


; ---- init powell minimizer
sautofit={iter:0L,idlabeliter:idlabeliter,idlabelerror:idlabelerror}
xi=identity(3)
ftol=1e-3
pinit1=[sgui.lon,sgui.lat,sgui.hgt]
p1=pinit1
widget_control,idlabelstatus,set_value='Status : running...'
powell,p1,xi,ftol,fmin,'rtsccguicloud_cloudautofit_error'

widget_control,idlabelstatus,set_value='Status : done !'

; ---- update the GUI sliders
sgui.lon=p1[0]
sgui.lat=p1[1]
sgui.hgt=p1[2]

; print,'lon : ',sgui.lon*!radeg
; print,'lat : ',sgui.lat*!radeg
; print,'hgt : ',sgui.hgt


; -- lat
idlabel=widget_info(sgui.wbase,find_by_uname='wslider_lat')
widget_control,idlabel,set_value=sgui.lat*!radeg

; -- hgt
idlabel=widget_info(sgui.wbase,find_by_uname='wslider_hgt')
widget_control,idlabel,set_value=sgui.hgt

rtsccguicloud_updatecarrlonshiftandslider

; ---- recompute the projections
rtsccguicloud_calccloudproj
; ---- display
rtsccguicloud_displayimages

return
end




; ------ main handler
pro widget1_event,ev
common com_sgui
uname=widget_info(ev.id,/uname)

if strmid(uname,0,7) eq 'wslider' then event_slidermoved,ev

if uname eq 'quit' then begin
	; ---- eval the error if a contour is drawn
	if sgui.sumA gt 0. and sgui.sumB gt 0. then begin
		event_evalerror,ev,/nodisp
	endif
	widget_control,ev.top,/destroy
	widget_control,sgui.groupleader,/destroy
        sgui.quit=1
endif
if uname eq 'done' then begin
	; ---- eval the error if a contour is drawn
	if sgui.sumA gt 0. and sgui.sumB gt 0. then begin
		event_evalerror,ev,/nodisp
	endif
	widget_control,ev.top,/destroy
	widget_control,sgui.groupleader,/destroy
        sgui.quit=0
endif
return
end


; ------ Main prog
pro rtsccguicloud,ima,imb,hdra,hdrb,$
        imdispsize=imdispsize,maxheight=maxheight,$
        ssim=ssimout,sgui=sguiout,$
        imeuvia=imeuvia,hdreuvia=hdreuvia,imeuvib=imeuvib,hdreuvib=hdreuvib,$
        demo=demo,showanaglyph=showanaglyph,eruptiondatein=eruptiondatein,$
        imlasco=imlasco,hdrlasco=hdrlasco,$
        forceinit=forceinit,swire=swire,$
        sparaminit=sparaminit,ocout=ocout,$
	    imabsunin=imabsunin,imbbsunin=imbbsunin,admin=admin,modal=modal
	    
common com_sgui
common com_imbsun,imabsun,imbbsun
if keyword_set(imabsunin) then imabsun=imabsunin
if keyword_set(imbbsunin) then imbbsun=imbbsunin

resolve_routine,'cmecloud',/either,/compile_full

; ---- set default values
if n_elements(imdispsize) eq 0 then imdispsize=[512L,512L]

; ---- load demo images if demo mode requested
if keyword_set(demo) then begin
    rtinitenv
    sdemo=readstruct(getenv('RT_DATAPATH')+path_sep()+'rtsccguiclouddemodata.xdr',/ignorestructnames)
    ima=sdemo.ima
    imb=sdemo.imb
    hdra=sdemo.hdra
    hdrb=sdemo.hdrb
    imeuvia=sdemo.imeuvia
    imeuvib=sdemo.imeuvib
    hdreuvia=sdemo.hdreuvia
    hdreuvib=sdemo.hdreuvib
endif


; ---- deal with LASCO image if present
;if n_elements(imlasco) ne 0 then begin
if keyword_set(imlasco) then begin
	if n_elements(hdrlasco) eq 0 then message,'LASCO image must be passed with a fits header'
	if datatype(hdrlasco) ne 'STC' then shdrlasco=fitshead2struct(hdrlasco,/wcs) else shdrlasco=hdrlasco
	flaglasco=1b
endif else flaglasco=0b

; ---- init object containing info to be passed to event handler
if n_elements(firstrun) eq 0 then firstrun=1b
if firstrun || keyword_set(forceinit) then begin
	; ---- initialize the CME cloud
	foo=fltarr(imdispsize[0],imdispsize[1])
	autofitresolution=-2
	fooreb=fltarr(128,128)
	sinit={nbvertshell:50,nbvertcirup:20,nbvertaxisp:5,han:30.*!dtor,hgt:2.,rat:0.4,lon:0.,lat:0.,rot:0.,flagwireonoff:1b,$
	    	carrorstony:0b,imdistA:foo,imdistB:foo,masksunA:ptr_new(),masksunB:ptr_new(),sumA:0.,sumB:0.,error:0.,contourA:ptr_new(),$
		contourB:ptr_new(),imdistAreb:fooreb,imdistBreb:fooreb,masksunAreb:ptr_new(),masksunBreb:ptr_new(),sumAreb:0.,sumBreb:0.,$
		autofitresolution:autofitresolution,sensitrange:0.1,nel:1e5,sigin:0.1,sigout:0.1,ptprofA:[[0.,0],[0,0]],ptprofB:[[0.,0],[0,0]],$
		losrange:[-10,10],losnbp:64}
	firstrun=0b
endif

; ---- force parameters passed by the user
if n_elements(sparaminit) ne 0 then begin
	sptags=tag_names(sparaminit)
	nbsptags=n_tags(sparaminit)
	sinittags=tag_names(sinit)
	for i=0l,nbsptags-1 do begin
		m=where(sptags[i] eq sinittags,cnt)
		if cnt eq 1 then sinit.(m[0])=sparaminit.(i)
	endfor
endif

; ---- deal with eruption date
if n_elements(eruptiondatein) eq 0 then eruptiondate=hdra.date_obs else eruptiondate=eruptiondatein
rtsccguicloud_computecarrlonshift_nocommon,hdra,eruptiondate,sinit.carrorstony,carrlonshiftdeg,carrstonyshiftdeg

; ---- generate first view of the cloud using initial parameters
oc=cmecloud(sinit.han,sinit.hgt,sinit.nbvertaxisp,sinit.nbvertcirup,sinit.rat,sinit.nbvertshell,/distjuncisleadingedge)

neang=rtsccguicloud_calcneang(sinit.lon,sinit.lat,sinit.rot,carrlonshiftdeg,carrstonyshiftdeg)
rtcloud,oc,sa,imsize=imdispsize,scchead=hdra,neang=neang,/fclip,/quiet
rtcloud,oc,sb,imsize=imdispsize,scchead=hdrb,neang=neang,/fclip,/quiet
if flaglasco then rtcloud,oc,slasco,imsize=imdispsize,scchead=shdrlasco,/quiet,neang=neang,/fclip


; ---- rebytscale the images
ima2=bytscl(ima,top=254)
imb2=bytscl(imb,top=254)
if flaglasco then imlasco2=bytscl(imlasco,top=254)

; ---- make anaglyph if requested
if keyword_set(showanaglyph) then begin

    ; ---- change dist and center: set SC B on SC A 
    hdrbanag=hdra

    hdrbanag.crln_obs=hdrb.crln_obs
    hdrbanag.crlt_obs=hdrb.crlt_obs

    rtcloud,oc,saanag,imsize=imdispsize,scchead=hdra,/quiet,neang=neang
    rtcloud,oc,sbanag,imsize=imdispsize,scchead=hdrbanag,/quiet,neang=neang

    anag=mkanaglyph(bytscl(saanag.im),bytscl(sbanag.im))
    
    wnd,10,imdispsize[0],imdispsize[1],title='Cloud Anaglyph'

endif

; ---- user maxheight if passed
if keyword_set(maxheight) then maxh=maxheight else maxh=24.

; ---- check presence of euvi images
if keyword_set(imeuvia) xor keyword_set(hdreuvia) then message,'EUVI-A image must be passed with a fits header'
if keyword_set(imeuvib) xor keyword_set(hdreuvib) then message,'EUVI-B image must be passed with a fits header'

; ---- re-init object containing info to be passed to event handler
sgui={ima:ima2,imb:imb2,hdra:hdra,hdrb:hdrb,lon:sinit.lon,lat:sinit.lat,rot:sinit.rot,han:sinit.han,hgt:sinit.hgt,rat:sinit.rat,$
    nbvertaxisp:sinit.nbvertaxisp,nbvertcirup:sinit.nbvertcirup,nbvertshell:sinit.nbvertshell,imdispsize:imdispsize,losrange:sinit.losrange,$
    losnbp:sinit.losnbp,nel:sinit.nel,sigin:sinit.sigin,sigout:sinit.sigout,imeuvia:(keyword_set(imeuvia) ? bytscl(imeuvia,top=254) : 0),$
    imeuvib:(keyword_set(imeuvib) ? bytscl(imeuvib,top=254) : 0),hdreuvia:(keyword_set(hdreuvia) ? hdreuvia : 0),hdreuvib:(keyword_set(hdreuvib) ? hdreuvib : 0),$
    showanag:keyword_set(showanaglyph),anag:(keyword_set(showanaglyph) ? anag : 0),hdrbanag:(keyword_set(showanaglyph) ? hdrbanag : 0),eruptiondate:eruptiondate,$
    carrlonshiftdeg:0d,wbase:0L,flaglasco:flaglasco,imlasco:(flaglasco ? imlasco2 : 0),shdrlasco:(flaglasco ? shdrlasco : 0),flagwireonoff:sinit.flagwireonoff,$
    carrorstony:sinit.carrorstony,carrstonyshiftdeg:0.,oldcarrlonshiftdeg:0d,imdistA:sinit.imdistA,imdistB:sinit.imdistB,masksunA:sinit.masksunA,$
    masksunB:sinit.masksunB,sumA:sinit.sumA,sumB:sinit.sumB,error:0.,contourA:sinit.contourA,contourB:sinit.contourB,imdistAreb:sinit.imdistAreb,$
    imdistBreb:sinit.imdistBreb,masksunAreb:sinit.masksunAreb,masksunBreb:sinit.masksunBreb,sumAreb:sinit.sumAreb,sumBreb:sinit.sumBreb,autofitresolution:sinit.autofitresolution,$
    sensitrange:sinit.sensitrange,ptprofa:sinit.ptprofa,ptprofb:sinit.ptprofb,groupleader:-1,quit:0}

; ---- update carr lon shift in sgui
rtsccguicloud_computecarrlonshift

; ---- modify color table
tvlct,rrr0,ggg0,bbb0,/get
loadct,0
tvlct,rrr,ggg,bbb,/get
rrr[255]=0
bbb[255]=0
;ggg[255]=0
tvlct,rrr,ggg,bbb

; ---- display
rtsccguicloud_displayimages

; ---- initialize widget window
groupleader = Widget_Base(Map=0)
Widget_Control, groupleader, /Realize
wbase=widget_base(RESOURCE_NAME = 'base',title='STEREO Cloud',/column,/BASE_ALIGN_TOP,MODAL = keyword_set(modal),group_leader=groupleader)

wtab=widget_tab(wbase,/align_left)
sgui.groupleader=groupleader
sgui.wbase=wbase

; ---- tab 1: positionning
wbase_tab1=widget_base(wtab,title='Position',/column)

sliderlon=rtsccguicloud_calclonslidervalues(sgui.lon,sgui.carrlonshiftdeg,sgui.carrstonyshiftdeg)
wslider_lon=cw_fslider(wbase_tab1,minimum=0.,maximum=360.,value=sliderlon,title='Longitude',uname='wslider_lon',/drag,/edit,scroll=1.,xsize=360)
wslider_lat=cw_fslider(wbase_tab1,minimum=-90,maximum=90,value=sgui.lat*!radeg,title='Latitude',uname='wslider_lat',/drag,/edit,scroll=1.,xsize=360)
wslider_rot=cw_fslider(wbase_tab1,minimum=-90,maximum=90,value=sgui.rot*!radeg,title='Tilt Angle',uname='wslider_rot',/drag,/edit,scroll=1.,xsize=360)
wslider_hgt=cw_fslider(wbase_tab1,minimum=1.,maximum=maxh,value=sgui.hgt,title='Height',uname='wslider_hgt',/drag,/edit,scroll=1.,xsize=360)
wslider_rat=cw_fslider(wbase_tab1,minimum=0.01,maximum=1.,value=sgui.rat,title='Ratio',uname='wslider_rat',/drag,/edit,xsize=360)
wslider_han=cw_fslider(wbase_tab1,minimum=0.,maximum=90.,value=sgui.han*!radeg,title='Half Angle',uname='wslider_han',/drag,/edit,scroll=0.5,xsize=360)
wlabel_eruptiondate=widget_label(wbase_tab1,value='Eruption Date',uname='wlabel_eruptiondate')
wtext_eruptiondate=widget_text(wbase_tab1,/editable,value=eruptiondate,uname='wtext_eruptiondate',event_pro='event_updateeruptiondate')


; ---- tab 2: cloud
wbase_tab2=widget_base(wtab,title='Cloud',/column)
wlabel_nbvertshell=widget_label(wbase_tab2,value='# Points Axis',uname='wlabel_nbvertshell')
wtext_nbvertshell=widget_text(wbase_tab2,/editable,value=strtrim(sgui.nbvertshell,2),uname='wtext_nbvertshell',event_pro='event_changecloud')

wlabel_nbvertcirup=widget_label(wbase_tab2,value='# Points Shell',uname='wlabel_nbvertcirup')
wtext_nbvertcirup=widget_text(wbase_tab2,/editable,value=strtrim(sgui.nbvertcirup,2),uname='wtext_nbvertcirup',event_pro='event_changecloud')

wlabel_nbvertaxisp=widget_label(wbase_tab2,value='# Points Leg Axis',uname='wlabel_nbvertaxisp')
wtext_nbvertaxisp=widget_text(wbase_tab2,/editable,value=strtrim(sgui.nbvertaxisp,2),uname='wtext_nbvertaxisp',event_pro='event_changecloud')

wlabel_msg1=widget_label(wbase_tab2,value='Press ENTER when done',uname='wlabel_msg1')


; ---- tab 3: simu parameters 
wbase_tab3=widget_base(wtab,title='Simu',/column)
wlabel_losrangemin=widget_label(wbase_tab3,value='LOS Range Mini')
wtext_losrangemin=widget_text(wbase_tab3,/editable,value=strtrim(sgui.losrange[0],2),uname='wtext_losrangemin',event_pro='event_updatelosrangemin',/kbrd_focus_events)
wlabel_losrangemax=widget_label(wbase_tab3,value='LOS Range Maxi')
wtext_losrangemax=widget_text(wbase_tab3,/editable,value=strtrim(sgui.losrange[1],2),uname='wtext_losrangemax',event_pro='event_updatelosrangemax',/kbrd_focus_events)
wlabel_losnbp=widget_label(wbase_tab3,value='LOS Number of Points')
wtext_losnbp=widget_text(wbase_tab3,/editable,value=strtrim(sgui.losnbp,2),uname='wtext_losnbp',event_pro='event_updatelosnbp',/kbrd_focus_events)

wlabel_nel=widget_label(wbase_tab3,value='Ne')
wtext_nel=widget_text(wbase_tab3,/editable,value=strtrim(sgui.nel,2),uname='wtext_ne',event_pro='event_updatenel',/kbrd_focus_events)
wlabel_sigin=widget_label(wbase_tab3,value='Sigma in')
wtext_sigin=widget_text(wbase_tab3,/editable,value=strtrim(sgui.sigin,2),uname='wtext_sigin',event_pro='event_updatesigin',/kbrd_focus_events)
wlabel_sigout=widget_label(wbase_tab3,value='Sigma out')
wtext_sigout=widget_text(wbase_tab3,/editable,value=strtrim(sgui.sigout,2),uname='wtext_sigout',event_pro='event_updatesigout',/kbrd_focus_events)



wlabel_msg3=widget_label(wbase_tab3,value='['+strtrim(sgui.losrange[0],2)+','+strtrim(sgui.losrange[1],2)+'],'+strtrim(sgui.losnbp,2),uname='wlabel_msg3')

wlabel_msg2=widget_label(wbase_tab3,value='Press ENTER when done',uname='wlabel_msg2')

; ---- tab 4: Contour definition
wbase_tab4=widget_base(wtab,title='Contour',/column)
wbutton_drawcontour=widget_button(wbase_tab4,value='Draw Contours',uname='wbutton_drawcontour',event_pro='event_drawcontour')

wlabel_msgcontour=widget_label(wbase_tab4,value='Message here...',uname='wlabel_msgcontour',/dynamic_resize)

; ---- tab 5: Auto fit
wbase_tab5=widget_base(wtab,title='Auto Fit',/column)
wbutton_autofit=widget_button(wbase_tab5,value='Start Auto Fit',uname='wbutton_autofit',event_pro='event_autofit')

wlabel_autofit_comments=widget_label(wbase_tab5,value='Only Lon, Lat and Height are free parameters.',uname='wlabel_autofit_comments')

wlabel_autofit_status=widget_label(wbase_tab5,value='Status: stopped',uname='wlabel_autofit_status',/dynamic_resize)
wlabel_autofit_iter=widget_label(wbase_tab5,value='Iterations: 0',uname='wlabel_autofit_iter',/dynamic_resize)
wlabel_autofit_error=widget_label(wbase_tab5,value='Fit: ?',uname='wlabel_autofit_error',/dynamic_resize)

; ---- tab 6: Sensitivity
wbase_tab6=widget_base(wtab,title='Sensit.',/column)
wlabel_sensitrange=widget_label(wbase_tab6,value='Sensitivity range : [0..1]')
wtext_sensitrange=widget_text(wbase_tab6,/editable,value=strtrim(sgui.sensitrange,2),uname='wtext_sensitrange',event_pro='event_updatesensitrange',/kbrd_focus_events)

wbutton_sensitstartlon=widget_button(wbase_tab6,value='Start Lon',uname='wbutton_sensitstartlon',event_pro='event_sensitlon')
wbutton_sensitstartlat=widget_button(wbase_tab6,value='Start Lat',uname='wbutton_sensitstartlat',event_pro='event_sensitlat')
wbutton_sensitstartrot=widget_button(wbase_tab6,value='Start Rot',uname='wbutton_sensitstartrot',event_pro='event_sensitrot')
wbutton_sensitstarthgt=widget_button(wbase_tab6,value='Start Hgt',uname='wbutton_sensitstarthgt',event_pro='event_sensithgt')
wbutton_sensitstartrat=widget_button(wbase_tab6,value='Start Rat',uname='wbutton_sensitstartrat',event_pro='event_sensitrat')
wbutton_sensitstarthan=widget_button(wbase_tab6,value='Start han',uname='wbutton_sensitstarthan',event_pro='event_sensithan')


wlabel_sensit_status=widget_label(wbase_tab6,value='Status: stopped',uname='wlabel_sensit_status',/dynamic_resize)
wlabel_sensit_lon=widget_label(wbase_tab6,value='lon: ',uname='wlabel_sensit_lon',/dynamic_resize)
wlabel_sensit_lat=widget_label(wbase_tab6,value='lat: ',uname='wlabel_sensit_lat',/dynamic_resize)
wlabel_sensit_rot=widget_label(wbase_tab6,value='rot: ',uname='wlabel_sensit_rot',/dynamic_resize)
wlabel_sensit_hgt=widget_label(wbase_tab6,value='hgt: ',uname='wlabel_sensit_hgt',/dynamic_resize)
wlabel_sensit_rat=widget_label(wbase_tab6,value='rat: ',uname='wlabel_sensit_rat',/dynamic_resize)
wlabel_sensit_han=widget_label(wbase_tab6,value='han: ',uname='wlabel_sensit_han',/dynamic_resize)


; ---- tab 7: Profile definition
if keyword_set(admin) then begin
	wbase_tab7=widget_base(wtab,title='Profile',/column)
	wbutton_drawprofile=widget_button(wbase_tab7,value='Draw Profiles',uname='wbutton_drawprofiles',event_pro='event_drawprofiles')

	wlabel_msgprofiles=widget_label(wbase_tab7,value='Message here...',uname='wlabel_msgprofiles',/dynamic_resize)

	wbutton_updateprofiles=widget_button(wbase_tab7,value='Update profiles',uname='wbutton_updateprofiles',event_pro='event_updateprofiles')
endif

; ---- main window widgets
wlist_carrstony=cw_bgroup(wbase,['Carrington','Stonyhurst'],button_uvalue=['carr','stony'],column=2,set_value=sgui.carrorstony,ids=[0,1],/exclusive,event_func='event_carrstony')

wbutton_wireonoff=widget_button(wbase,value='Wire Off',uname='wbutton_wireonoff',event_pro='event_wireonoff')

wbutton_evalerror=widget_button(wbase,value='Eval. Fit',uname='wbutton_evalerror',event_pro='event_evalerror')
wlabel_error=widget_label(wbase,value='Fit : ?',uname='wlabel_error',/dynamic_resize)


wlabel_scapos=widget_label(wbase,value=rtsccguicloud_formatscpos(hdra,sgui.carrorstony),uname='wlabel_scapos')
wlabel_scadate=widget_label(wbase,value='Date Obs: '+hdra.date_obs)

wlabel_scbpos=widget_label(wbase,value=rtsccguicloud_formatscpos(hdrb,sgui.carrorstony),uname='wlabel_scbpos')
wlabel_scbdate=widget_label(wbase,value='Date Obs: '+hdrb.date_obs)


if keyword_set(admin) then begin
	wbutton_dbsync=widget_button(wbase,value='Sync with DB',uname='wbutton_dbsync',event_pro='event_dbsync')
	wtext_msid=widget_text(wbase,/editable,value='',uname='wtext_msid',event_pro='event_dbsync')
endif

wbutton_go=widget_button(wbase,value='Generate View',uname='wbutton_go',event_pro='event_generateview')

brow=widget_base(wbase,/ROW)
wbutton_quit=widget_button(brow,value='Done',uname='done')
wbutton_quit=widget_button(brow,value='Quit',uname='quit')

widget_control,wbase,/realize

; ---- run event handler
xmanager,'rtsccguicloud',wbase,EVENT_HANDLER='widget1_event',group_leader=group

; ---- copy simulation in the output variable
if n_elements(ssim) ne 0 then ssimout=ssim
sguiout=sgui
sinit={nbvertshell:sgui.nbvertshell,nbvertcirup:sgui.nbvertcirup,nbvertaxisp:sgui.nbvertaxisp,han:sgui.han,hgt:sgui.hgt,rat:sgui.rat,lon:sgui.lon,lat:sgui.lat,rot:sgui.rot,flagwireonoff:sgui.flagwireonoff,$
    carrorstony:sgui.carrorstony,imdistA:sgui.imdistA,imdistB:sgui.imdistB,masksunA:sgui.masksunA,masksunB:sgui.masksunB,sumA:sgui.sumA,sumB:sgui.sumB,contourA:sgui.contourA,contourB:sgui.contourB,$
    imdistAreb:sgui.imdistAreb,imdistBreb:sgui.imdistBreb,masksunAreb:sgui.masksunAreb,masksunBreb:sgui.masksunBreb,sumAreb:sgui.sumAreb,sumBreb:sgui.sumBreb,autofitresolution:sgui.autofitresolution,$
    sensitrange:sgui.sensitrange,nel:sgui.nel,sigin:sgui.sigin,sigout:sgui.sigout,ptprofa:sgui.ptprofa,ptprofb:sgui.ptprofb,losrange:sgui.losrange,losnbp:sgui.losnbp}

if flaglasco then swire={sa:sa,sb:sb,slasco:slasco} else swire={sa:sa,sb:sb}

; ---- return 3D cloud if requested
ocout=oc

; ---- restore color table
tvlct,rrr0,ggg0,bbb0
return
end


; ---- tests routine
pro rtsccguicloud_test

; ----- COR2 example

; event: 20070605_072230
; preevent: 20070604_132230

; ---- get the event and pre-event image
rtinitenv,forcelibfile='/home/thernis/idl/bin/secchi/cpp/scraytrace/optimized/src/.libs/libraytrace.so'
eventtrip=sccdiskpath((sccgetpoltriplet('2007/06/05 07:22:30',1,1,addfields=',exptime')).filename)
preevtrip=sccdiskpath((sccgetpoltriplet('2007/06/04 13:22:30',1,1,addfields=',exptime')).filename)

eventtripb=sccdiskpath((sccgetpoltriplet('2007/06/05 07:22:30',2,1,addfields=',exptime')).filename)
preevtripb=sccdiskpath((sccgetpoltriplet('2007/06/04 13:22:30',2,1,addfields=',exptime')).filename)


; -- event 2
eventtrip=sccdiskpath((sccgetpoltriplet('2007/05/15 23:53:30',1,1,addfields=',exptime')).filename)
preevtrip=sccdiskpath((sccgetpoltriplet('2007/05/15 18:22:30',1,1,addfields=',exptime')).filename)

eventtripb=sccdiskpath((sccgetpoltriplet('2007/05/15 23:53:30',2,1,addfields=',exptime')).filename)
preevtripb=sccdiskpath((sccgetpoltriplet('2007/05/15 18:52:30',2,1,addfields=',exptime')).filename)

eveuvia=sccdiskpath((sccgetnearestimage('2007/05/15 23:52:15',1,3))[0].filename)
eveuvib=sccdiskpath((sccgetnearestimage('2007/05/15 23:53:15',2,3))[0].filename)
secchi_prep,eveuvia,heuvia,imeuvia,/PRECOMMCORRECT_ON;,/rotate_on,/rotinterp_on,/rotcubic_on
secchi_prep,eveuvib,heuvib,imeuvib,/PRECOMMCORRECT_ON;,/rotate_on,/rotinterp_on,/rotcubic_on


; ---- compute total B
secchi_prep,eventtrip,hdrevent,imevent,/polariz_on,/rotate_on,/PRECOMMCORRECT_ON,/rotinterp_on,/silent
secchi_prep,preevtrip,hdrpreev,impreev,/polariz_on,/rotate_on,/PRECOMMCORRECT_ON,/rotinterp_on,/silent

secchi_prep,eventtripb,hdreventb,imeventb,/polariz_on,/rotate_on,/PRECOMMCORRECT_ON,/rotinterp_on,/silent
secchi_prep,preevtripb,hdrpreevb,impreevb,/polariz_on,/rotate_on,/PRECOMMCORRECT_ON,/rotinterp_on,/silent

m=get_smask(hdrevent)
mb=get_smask(hdreventb)
wnd,0,alog10(m*(imevent-impreev) > 1e-12 < 1e-10),.25
wnd,1,alog10(mb*(imeventb-impreevb) > 1e-12 < 1e-10),.25

ima=bytscl(rebin(alog10(m*(imevent-impreev) > 1e-12 < 1e-10),512,512))
imb=bytscl(rebin(alog10(mb*(imeventb-impreevb) > 1e-12 < 1e-10),512,512))

hdra=hdrevent
hdrb=hdreventb

rtsccguicloud,ima,imb,hdra,hdrb

; ----- HI1 example
eventa=sccdiskpath((sccgetnearestimage('2007/04/20 00:00:00',1,5))[0].filename)
eventb=sccdiskpath((sccgetnearestimage('2007/04/20 00:00:00',2,5))[0].filename)

; -- read bckground image
sb=readstruct('/home/thernis/work/starremove/immedhi1AApr2007.xdr')

secchi_prep,eventa,hdra,ima,/PRECOMMCORRECT_ON ,/SHUTTERLESS_OFF,/silent
secchi_prep,eventb,hdrb,imb,/PRECOMMCORRECT_ON ,/SHUTTERLESS_OFF,/silent

imhi1a=rebin(bytscl(ima-smooth(ima,10),0,0.05),512,512)
imhi1b=rebin(bytscl(imb-smooth(imb,10),0,0.05),512,512)

imhi1a=bytscl(rebin(alog10(ima>1),512,512))
imhi1b=bytscl(rebin(alog10(imb>1),512,512))


rtsccguicloud,imhi1a,imhi1b,hdra,hdrb,maxheight=100.


return
end

;  $Log: rtsccguicloud.pro,v $
;  Revision 1.33  2011/09/23 19:08:57  thernis
;  Fix bug when calling demo mode
;
;  Revision 1.32  2010-09-08 18:24:36  mcnutt
;  added a done button and set sgui.quit to 1 if quit button is used
;
;  Revision 1.31  2010/09/08 15:59:41  thernis
;  Merge few comments with latest version
;
;  Revision 1.30  2010-09-08 14:33:12  mcnutt
;  remembers ssim values
;
;  Revision 1.29  2010/08/26 13:23:59  mcnutt
;  added KEYWORD_NULL_VALUE to call to fitshead2wcs
;
;  Revision 1.28  2010/08/12 15:23:29  mcnutt
;  added modal keyword
;
;  Revision 1.27  2009/03/17 15:41:30  thernis
;  - Add more comments in the header
;  - Implement Ne fitting features, but only available in admin mode for testing
;
;  Revision 1.26  2009/02/05 19:59:06  thernis
;  - Implement sensitivity analysis
;  - Implement brightness view for LASCO
;  - Allow changing the Ne for brightness computation
;  - Implement a keyword to return the model wireframe
;
;  Revision 1.25  2008/10/17 15:14:37  thernis
;  First commit of multi-model fitting gui
;
;  Revision 1.24  2008/07/03 16:23:55  thernis
;  Implement sparaminit keyword to allow the user to pass initialization of the model parameter
;
;  Revision 1.23  2008/06/27 18:08:14  thernis
;  Implement Stonyhurst corrdinate system
;
;  Revision 1.22  2008/06/26 15:44:49  thernis
;  Enlarge the sliders to lower the sensitivity
;
;  Revision 1.21  2008/06/26 15:28:45  thernis
;  Implement swire keyword to be able to get back the projection of the wireframe model
;
;  Revision 1.20  2008/06/10 20:28:02  thernis
;  - Fix neang calculation: take into account the eruption date correction
;  - Implement wire on/off button
;
;  Revision 1.19  2008/06/09 21:55:17  thernis
;  - fix eruption misspelling
;  - fix longitude shift computation function of eruptiondate
;
;  Revision 1.18  2008/06/06 17:44:42  thernis
;  Allow to use a SOHO (EIT or LASCO) image as a 3rd view.
;
;  Revision 1.17  2008/05/20 19:06:23  thernis
;  - remove one unuseful function
;  - fix feet height for the rendering
;
;  Revision 1.16  2008/05/16 20:10:43  thernis
;  Fix the height parameter slider to the true height of the croissant model
;
;  Revision 1.15  2008/04/01 18:37:01  thernis
;  Fix initial and remembering values bug
;
;  Revision 1.14  2008/03/31 13:53:15  thernis
;  Remember previous values when you quit and rerun.
;
;  Revision 1.13  2008/03/03 18:46:12  thernis
;  Add ignorestructname keyword in readstruct.pro call to avoid problem with new definition of SECCHI_HDR_STRUCT
;
;  Revision 1.12  2007/09/28 20:31:56  thernis
;  Move erruptiondatein keyword default value setting
;
;  Revision 1.11  2007/07/25 21:47:49  thernis
;  - Implement anaglyph view of the cloud
;  - Add date_obs in the GUI
;  - Update labels in GUI->cloud tab
;
;  Revision 1.10  2007/07/25 19:51:06  thernis
;  Uniformize angle units to radian
;
;  Revision 1.9  2007/07/25 19:41:46  thernis
;  Implement visu of neutral line orientation in the EUVI views
;
;  Revision 1.8  2007/07/25 14:46:01  thernis
;  Initialize the scraytrace environment variable in the demo mode
;
;  Revision 1.7  2007/07/24 18:04:14  thernis
;  Max half angle is 90 deg, not 180 deg
;
;  Revision 1.6  2007/07/24 15:22:15  thernis
;  secchi_prep needs to update the hdr in the hi1 example
;
;  Revision 1.5  2007/07/23 20:54:01  thernis
;  Implement a not too wordy progression
;
;  Revision 1.4  2007/07/23 20:22:14  thernis
;  - Implement demo mode
;  - Fix wrong window number for EUVI display
;
;  Revision 1.3  2007/07/19 22:23:58  thernis
;  - Implement visualization of SR on EUVI images
;  - Remove offset of 180 degrees to comply with new CBasis.cpp
;
;  Revision 1.2  2007/07/19 19:19:29  thernis
;  Forgot to change the procedure name
;
;  Revision 1.1  2007/07/19 19:18:38  thernis
;  First commit
;
