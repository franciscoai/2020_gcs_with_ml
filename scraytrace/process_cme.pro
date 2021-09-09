;+
;$Id: process_cme.pro,v 1.20 2012/07/13 14:43:21 mcnutt Exp $
;
; Project     : SECCHI/LASCO
;                   
; Name        : process_cme
;               
; Purpose     : wigdet tool for input to guicloud
;               
; Explanation : To select input image for rtsccguicloud
;                     
; Use         : IDL> process_cme,/secchi,/lasco
;
; Optional Inputs:
;
; output: selected files cme ht fits, and param file
;
; KEYWORDS: secchi = structure list of images from scclister if keyword set without list scclister will be called, secchi images change selected inside widget	
;          lasco = a list of lasco images , if keyword set without list it will call wlister keyword must be set to use lasco images.
;               
; Calls       : rtsccguicloud.pro
;
; Side effects: Some widget issues still need to be resolved.
;               
; Category    : Image Display.  CME.
;               
; Written     : Lynn Simpson NRL 2010-08-12
;               
; See Also    : 
;	
;-            
;$Log: process_cme.pro,v $
;Revision 1.20  2012/07/13 14:43:21  mcnutt
;removed stops
;
;Revision 1.19  2012/07/13 11:26:47  mcnutt
;corrected backgrounds when keyword set selon
;
;Revision 1.18  2012/05/24 18:59:44  mcnutt
;change maxheight to 100 rtsccguicloud
;
;Revision 1.17  2012/05/23 17:36:43  mcnutt
;creates a difference image when processing none mass fits images sellon=1
;
;Revision 1.16  2012/05/21 11:58:36  mcnutt
;added maxheight=60 keyword to rtsccguicloud call
;
;Revision 1.15  2012/05/16 15:33:01  mcnutt
;added mpw tag to ppcme structure
;
;Revision 1.14  2012/05/11 18:39:47  mcnutt
;changed image list from droplist to combobox for scroll bars
;
;Revision 1.13  2012/04/30 17:10:25  mcnutt
;corrected files when called from make_cme_mass_fits
;
;Revision 1.12  2012/04/27 14:06:52  mcnutt
;added total mass  output and plot option and changed widget display
;
;Revision 1.11  2012/04/11 13:28:26  mcnutt
;corrected input dir
;
;Revision 1.10  2011/05/13 20:34:17  mcnutt
;corrected file list in play movie
;
;Revision 1.9  2011/03/02 19:07:42  mcnutt
;corrected error if tels files are found
;
;Revision 1.8  2011/02/25 15:52:28  mcnutt
;fixed play movie and ht files
;
;Revision 1.7  2011/02/17 20:57:35  mcnutt
;revamped to include HI1 and COR1 not fully play movie does work yet
;
;Revision 1.6  2010/09/10 12:37:12  avourlid
;removed SPAWN commands to reduce OS-related errors, AV
;
;Revision 1.5  2010/09/08 18:22:07  mcnutt
;added lasco select drop menu
;
;Revision 1.4  2010/09/08 15:01:45  mcnutt
;corrected movie display
;
;Revision 1.3  2010/08/31 15:13:49  mcnutt
;write png files for swire and wlrt windows used with generic_movie to display cme fits
;
;Revision 1.2  2010/08/26 13:24:44  mcnutt
;beta2
;
;Revision 1.1  2010/08/12 15:06:14  mcnutt
;Beta version
;

function pcme_diff_image,ffile,bkg,outsize=outsize,hdr=hdr,notsecchi=notsecchi
  if ~keyword_set(notsecchi) then begin
    secchi_prep,bkg,hdr1,img1,outsize=outsize,/CALFAC_OFF, /NOCALFAC, /calimg_off
    secchi_prep,ffile,hdr,img2,outsize=outsize,/CALFAC_OFF, /NOCALFAC, /calimg_off
  endif else begin
     img1=mk_img(bkg,bmin,bmax,hdr1,pan=.5,/roll, /NO_DISPLAY)
     img2=mk_img(ffile,bmin,bmax,hdr2,pan=.5,/roll, /NO_DISPLAY)
     hdr=fitshead2struct(hdr2,/dash2underscore)
  endelse
  return,img2-img1
end

pro process_cme_event,ev

common pprocesscme,ppcme,htarr,rtarr,sellon_sgui

   if(where(tag_names(ev) eq 'VALUE') gt -1)then input=ev.Value else WIDGET_CONTROL, ev.id, GET_UVALUE=input
   WIDGET_CONTROL, ev.id, GET_UVALUE=test
   IF (DATATYPE(test) EQ 'STR') THEN input = test
   z=where(ppcme.outfiles eq input)
   if z(0) gt -1 then input='outf'

   IF strpos(input,'_') ge 0 then begin
     telsel=strmid(input,strpos(input,'_')+1,strlen(input))
     nw=where(ppcme.tels eq telsel) & nw=nw(0)
     input=strmid(input,0,strpos(input,'_'))
   endif
    CASE (input) OF

        'CMED' : ppcme.dir=ev.value

	'SSEL' :  BEGIN 	
                     WIDGET_CONTROL, ppcme.cmebase , /DESTROY 
                     if ppcme.ofiles(0) eq 1 and ppcme.outfile gt -1 then FREE_LUN,ppcme.outfile
                     if ppcme.ofiles(1) eq 1  and ppcme.outfile1 gt -1 then Free_lun,ppcme.outfile1
                     ppcme.outfile=-1
		     ppcme.outfile1=-1
                     ppcme.htfile=''
		     ppcme.rtfile=''
                     undefine,htarr
                     undefine,rtarr
                     process_cme,ppcme.dir
                     return
    	    END

         'FSEL' : begin
                    ppcme.runsel(nw)= ev.index
                    for nw2=0,n_Elements(ppcme.tels)-1 do begin
                       if nw2 ne nw and strpos(ppcme.ifiles(nw2,0),'Not selected') eq -1 then begin
                         date1=reform(ppcme.dates(nw2,*)) & date1=date1(where(date1 gt 0))
			 mn = min(abs(date1-ppcme.dates(nw,ev.index)),dex)
                         WIDGET_CONTROL,ppcme.sfiles(nw2),SET_COMBOBOX_SELECT= dex
                         ppcme.runsel(nw2)= dex
                       endif
                     endfor
                     tmp=where(strpos(telsel,'cor2') gt -1,iscor2)
                     tmp=where(strpos(telsel,'cor1') gt -1,iscor1)
                     tmp=where(strpos(telsel,'c2') gt -1,isc2)
                     tmp=where(strpos(telsel,'c3') gt -1,isc3)
                     tmp=where(strpos(telsel,'hi1') gt -1,ishi1)
                     if ishi1 or iscor1 or iscor2 then begin
                        if ishi1 then ind=where(ppcme.usesv eq 'hi1')
                        if iscor1 then ind=where(ppcme.usesv eq 'cor1')
                        if iscor2 then ind=where(ppcme.usesv eq 'cor2')
                        ppcme.usesecchi=ppcme.usesv(ind(0))
                        widget_control,ppcme.uses,SET_COMBOBOX_SELECT=ind(0)
		    endif 
                     if isc2 or isc3 then begin
                        ind=where(ppcme.uselv eq telsel)
                        ppcme.uselasco=ppcme.uselv(ind(0))
                        widget_control,ppcme.usel,SET_COMBOBOX_SELECT=ind(0)
		    endif 
    	    END

        'BMINT' : ppcme.bminmax(nw,0)=ev.value
        'BMAXT' : ppcme.bminmax(nw,1)=ev.value
        'SCLS':   ppcme.fscales(nw)=ppcme.scls(ev.index)

        'SELMINMAX' : begin
		    if ppcme.tels(nw) eq 'c2' or ppcme.tels(nw) eq 'c3' then notsecchi=1 else notsecchi=0
                    if ppcme.sellong eq 1 then image_in=pcme_diff_image(ppcme.ifiles(nw,[ppcme.runsel(nw)]),ppcme.backgrounds(nw),outsize=512,notsecchi=notsecchi) else $
		    	image_in=sccreadfits(ppcme.ifiles(nw,[ppcme.runsel(nw)]),hdra)
                    bmin=ppcme.bminmax(nw,0) & bmax=ppcme.bminmax(nw,1) & scale=ppcme.fscales(nw)
                    ima=select_bytscl(image_in,bmin,bmax,scale,/modal)
                    ppcme.bminmax(nw,*)=[bmin,bmax] & ppcme.fscales(nw)=scale
                  ;  widget_control,ppcme.bmint(nw),set_value=ppcme.bminmax(nw,0)
                  ;  widget_control,ppcme.bmaxt(nw),set_value=ppcme.bminmax(nw,1)
                  ;  WIDGET_CONTROL,ppcme.sclst(nw),SET_DROPLIST_SELECT= where(ppcme.scls eq ppcme.fscales(nw))
    	    END

         'outf' : ppcme.ofiles(z)=ev.select

         'usel':  ppcme.uselasco=ppcme.uselv(ev.index)
         'uses':  ppcme.usesecchi=ppcme.usesv(ev.index)
 
	'RUN' :  BEGIN 	
              ;--Create fake HT file
                fsela=[where(ppcme.tels eq ppcme.usesecchi+'a'),ppcme.runsel(where(ppcme.tels eq ppcme.usesecchi+'a'))]
                file_a=ppcme.ifiles(fsela(0),fsela(1))
                ascale=ppcme.fscales(where(ppcme.tels eq ppcme.usesecchi+'a'))
                   abminmax=ppcme.bminmax(where(ppcme.tels eq ppcme.usesecchi+'a'),*)
                fselb=[where(ppcme.tels eq ppcme.usesecchi+'b'),ppcme.runsel(where(ppcme.tels eq ppcme.usesecchi+'b'))]
                file_b=ppcme.ifiles(fselb(0),fselb(1))
                bscale=ppcme.fscales(where(ppcme.tels eq ppcme.usesecchi+'b'))
                   bbminmax=ppcme.bminmax(where(ppcme.tels eq ppcme.usesecchi+'b'),*)

                hdra='' & hdrb='' & hdrlasco=''
 
                if ppcme.sellong eq 1 then imga=pcme_diff_image(file_a,ppcme.backgrounds(fsela(0)),outsize=512,hdr=hdra) else imga=sccreadfits(file_A,hdra)
                  IF ascale eq 'LOG' THEN image=ALOG10(imga+3)>0 else $
                    IF ascale eq 'SQRT' THEN image = SQRT((imga)>0) else image=imga
                  bmin=abminmax(0) & bmax=abminmax(1) 
                  ima=rebin( BYTE(((image > bmin < bmax)-bmin)*(!D.TABLE_SIZE-1)/(bmax-bmin)),512,512)
		   
                if ppcme.sellong eq 1 then imgb=pcme_diff_image(file_b,ppcme.backgrounds(fselb(0)),outsize=512,hdr=hdrb) else imgb=sccreadfits(file_B,hdrb)
                  IF bscale eq 'LOG' THEN image=ALOG10(imgb+3)>0 else $
                    IF bscale eq 'SQRT' THEN image = SQRT((imgb)>0) else image=imgb
                  bmin=bbminmax(0) & bmax=bbminmax(1) 
                  imb= rebin(BYTE(((image > bmin < bmax)-bmin)*(!D.TABLE_SIZE-1)/(bmax-bmin)),512,512)

                if ppcme.uselasco ne 'no' then begin
                   fsell=[where(ppcme.tels eq ppcme.uselasco),ppcme.runsel(where(ppcme.tels eq ppcme.uselasco))]
		   file_l=ppcme.ifiles(fsell(0),fsell(1)) 
                   if ppcme.sellong eq 1 then imgl=pcme_diff_image(file_l,ppcme.backgrounds(fsell(0)),outsize=512,hdr=hdrlasco,/notsecchi) else imgl=sccreadfits(file_l,hdrlasco)
                     lscale=ppcme.fscales(where(ppcme.tels eq ppcme.uselasco))
                     lbminmax=ppcme.bminmax(where(ppcme.tels eq ppcme.uselasco),*)
                     IF lscale eq 'LOG' THEN image=ALOG10(imgl+3)>0 else $
                        IF lscale eq 'SQRT' THEN image = SQRT((imgl)>0) else image=imgl
                        bmin=lbminmax(0) & bmax=lbminmax(1) 
                        imlasco= rebin(BYTE(((image > bmin < bmax)-bmin)*(!D.TABLE_SIZE-1)/(bmax-bmin)),512,512)
                 endif
                ;--Run raytrace
                 rtsccguicloud,ima,imb,hdra,hdrb,imlasco=imlasco,hdrlasco=hdrlasco,ssim=ssim,sgui=sgui,swire=swire,/modal,maxheight=100

    	         sellon_sgui=sgui

                 if sgui.quit ne 1 and ppcme.sellong eq 0 then begin

                   ppcme.tmass(fsela(0),fsela(1))=total(imga(where(rebin(ssim.a.sbt.im,n_elements(imga(*,0)),n_elements(imga(0,*))) gt 0)))
                   ppcme.tmass(fselb(0),fselb(1))=total(imgb(where(rebin(ssim.b.sbt.im,n_elements(imgb(*,0)),n_elements(imgb(0,*))) gt 0)))
                   if ppcme.uselasco ne 'no' then ppcme.tmass(fsell(0),fsell(1))=total(imgl(where(rebin(ssim.lasco.sbt.im,n_elements(imgl(*,0)),n_elements(imgl(0,*))) gt 0)))

                   ppcme.cloudmass(fsela(0),fsela(1))=total(ssim.a.sbt.im)
                   ppcme.cloudmass(fselb(0),fselb(1))=total(ssim.b.sbt.im)
                   if ppcme.uselasco ne 'no' then ppcme.cloudmass(fsell(0),fsell(1))=total(ssim.lasco.sbt.im)

                 if ppcme.ofiles(0) eq 1 and ppcme.outfile eq -1 then begin
                    ppcme.htfile = ppcme.outdir+'WLRT_'+strmid(hdra.date_obs,0,10)+'.ht'
                    OPENW,outfile,ppcme.htfile,/get_lun,_extra=ex
                    PRINTF,outfile,'#VERSION=5'
                    PRINTF,outfile,'#DATE-OBS: ',strmid(hdra.date_obs,0,10)
                    PRINTF,outfile,'#TIME-OBS: ',strmid(hdra.date_obs,11,12)
                    PRINTF,outfile,'#SPACECRAFT: ',strmid(hdra.obsrvtry,7,1)
                    PRINTF,outfile,'#DETECTOR: ',hdra.detector
                    PRINTF,outfile,'#FILTER:   ',hdra.filter
                    PRINTF,outfile,'#POLAR:    ',hdra.polar
                    PRINTF,outfile,'#OBSERVER: ',getenv('USER')
                    PRINTF,outfile,'#IMAGE_TYPE: ','Raytrace Model'
                    PRINTF,outfile,'#UNITS: ','R/Rsun'
                    PRINTF,outfile,'#PLATESCALE: ',hdra.cdelt1   
                    PRINTF,outfile,'#SUBFIELD[0,0]: ',''
                    PRINTF,outfile,'#COMMENT: ', ''
                    PRINTF,outfile,'#FEAT_CODE: ','0=LE'
                    PRINTF,outfile,'# HEIGHT    DATE     TIME   ANGLE  TEL  FC  COL  ROW'
                    ppcme.outfile=outfile
		    close,ppcme.outfile
                  endif


              ;--create Raytrace parameter file
                  if ppcme.ofiles(1) eq 1  and ppcme.outfile1 eq -1 then begin
                    ppcme.rtfile = ppcme.outdir+'WLRT_'+strmid(hdra.date_obs,0,10)+'.rt'
                    OPENW,outfile1,ppcme.rtfile,/get_lun,_extra=ex
                    PRINTF,outfile1,'#   DATE     TIME     LON     LAT     ROT      HEIGHT RATIO  HALF ANGLE LOS RANGE  NBP'
                    ppcme.outfile1=outfile1
		    close,ppcme.outfile1
                  endif


                 if ppcme.ofiles(2) eq 1 and datatype(ssim) eq 'STC' then begin
		           writefits,ppcme.outdir+'wlrt_'+hdra.filename,ssim.A.sbt.im,ssim.A.sbt.FITSHEAD
                           writefits,ppcme.outdir+'wlrt_'+hdrb.filename,ssim.B.sbt.im,ssim.B.sbt.FITSHEAD
                           if ppcme.uselasco ne 'no' then   writefits,ppcme.outdir+'wlrt_'+hdrlasco.filename,ssim.lasco.sbt.im,ssim.lasco.sbt.FITSHEAD
                 endif
                 if ppcme.ofiles(3) eq 1 and datatype(swire) eq 'STC' then begin
		           writefits,ppcme.outdir+'swire_'+hdra.filename,swire.sa.im,ssim.A.sbt.FITSHEAD
                           writefits,ppcme.outdir+'swire_'+hdrb.filename,swire.sb.im,ssim.B.sbt.FITSHEAD
                           if ppcme.uselasco ne 'no' then   writefits,ppcme.outdir+'swire_'+hdrlasco.filename,swire.slasco.im,ssim.lasco.sbt.FITSHEAD
                 endif

;                 if ppcme.ofiles(4) eq 1 and datatype(ssim) eq 'STC' then begin
                          Device, Window_State=oWin
                           wset,0
		           filen=ppcme.outdir+'mass_'+strmid(hdra.filename,0,strpos(hdra.filename,'.fts'))
                           img=fTVREAD(/png, filename=filen,/noprompt)  
                           wset,1
		           filen=ppcme.outdir+'mass_'+strmid(hdrb.filename,0,strpos(hdrb.filename,'.fts'))
                           img=fTVREAD(/png, filename=filen,/noprompt)  
                           if ppcme.uselasco ne 'no' then  begin
                             wset,20
		             filen=ppcme.outdir+'mass_'+strmid(hdrlasco.filename,0,strpos(hdrlasco.filename,'.fts'))
                             img=fTVREAD(/png, filename=filen,/noprompt)  
                    	  endif
                           if owin(2) eq 1 then begin
                             wset,2
	  	             filen=ppcme.outdir+'wlrt_'+strmid(hdra.filename,0,strpos(hdra.filename,'.fts'))
                             img=fTVREAD(/png, filename=filen,/noprompt)  
                             wset,3
		             filen=ppcme.outdir+'wlrt_'+strmid(hdrb.filename,0,strpos(hdrb.filename,'.fts'))
                             img=fTVREAD(/png, filename=filen,/noprompt)  
                             if ppcme.uselasco ne 'no' then  begin
                               wset,21
		               filen=ppcme.outdir+'wlrt_'+strmid(hdrlasco.filename,0,strpos(hdrlasco.filename,'.fts'))
                               img=fTVREAD(/png, filename=filen,/noprompt)  
                    	    endif
                    	  endif
 
                  hdr_date_obs = strmid(hdra.date_obs,0,10)+'T'+strmid(hdra.date_obs,11,8)

		  if ppcme.ofiles(0) eq 1 then begin
		    line=string(sgui.hgt,hdr_date_obs,sgui.lat*!radeg,sgui.hdra.detector,0,0,0,'(F8.3,A20,f6.1,a5,2x,i3,2i5)')
		    if datatype(htarr) eq 'UND' then htarr=line else begin
                         z=where(strpos(rtarr,hdr_date_obs) gt 0)
		         if z(0) gt -1 then htarr(z)=line else htarr=[htarr,line]
                    endelse
                  endif
                 
		 if ppcme.ofiles(1) eq 1 then  begin
		    line=string(hdr_date_obs,sgui.lon*!radeg,sgui.lat*!radeg,sgui.rot*!radeg,sgui.hgt,sgui.rat,sgui.han*!radeg,sgui.losrange[0],sgui.losrange[1],sgui.losnbp,'(a20,4f8.3,f8.4,f8.3,2i5,i4)')		 
		    if datatype(rtarr) eq 'UND' then rtarr=line else begin
                         z=where(strpos(rtarr,hdr_date_obs) gt 0)
		         if z(0) gt -1 then rtarr(z)=line else rtarr=[rtarr,line]
                    endelse
                  endif

                 ppcme.done_list(fsela(0),fsela(1))=' * '
                 ppcme.done_list(fselb(0),fselb(1))=' * '
                 if ppcme.uselasco ne 'no' then begin
		    ppcme.done_list(fsell(0),fsell(1)) =' * '
                 endif
                       for nw =0,n_Elements(ppcme.tels)-1 do begin
                         z=where(ppcme.ifiles(nw,*) ne '')
                         if z(0) ne -1 then begin
                           ufiles=reform(ppcme.ifiles(nw,z))+reform(ppcme.done_list(nw,z))
                           WIDGET_CONTROL,ppcme.sfiles(nw),SET_value= strmid(ufiles,strlen(ppcme.ifiles(nw,0))-25,28)
                           WIDGET_CONTROL,ppcme.sfiles(nw),SET_COMBOBOX_SELECT= ppcme.runsel(nw)
                         endif
                       endfor

               endif

    	    END

        'ODIR' : ppcme.outdir=ev.value

        'pickfile' : BEGIN
	       file = PICKFILE( TITLE='Select output directory',GET_PATH=path)
		    if strmid(path,0,1,/reverse_offset) ne '/' then path=path+'/'
                    ppcme.outdir=path
                    widget_control,ppcme.odir,set_value=ppcme.outdir

    	    END

	'PLAY' :  BEGIN 
             j=where(ppcme.tels eq 'cor1a',cnt)
	     if cnt gt 0 then z1=where(reform(ppcme.done_list(j(0),*)) eq ' * ',cnt)   ; select cor1a files
	     if cnt gt 0 then done_list=reform(ppcme.ifiles(j(0),z1)) else done_list=''
             j=where(ppcme.tels eq 'cor2a',cnt)
             if cnt gt 0 then z1=where(reform(ppcme.done_list(j(0),*)) eq ' * ',cnt)   ; select cor2a files
	     if cnt gt 0 then done_list=[done_list,reform(ppcme.ifiles(j(0),z1))] else done_list=done_list
             j=where(ppcme.tels eq 'hi1a',cnt)
             if cnt gt 0 then z1=where(reform(ppcme.done_list(j(0),*)) eq ' * ',cnt)   ; select hi1a files
	     if cnt gt 0 then done_list=[done_list,reform(ppcme.ifiles(j(0),z1))] else done_list=done_list
             dt=where(done_list ne '',cnt)
             if cnt gt 0 then done_list=done_list(dt)
	     done_list=strmid(done_list,strlen(ppcme.dir),strlen(done_list(0))-strlen(ppcme.dir))
             if strpos(done_list(0),'/') eq 0 then done_list=strmid(done_list,1,strlen(done_list(0))-1)
	     z=where(done_list ne ' ')
             if z(0) gt -1 then begin
               play_list=strmid(done_list(z),0,15)
               ptime=double(strmid(play_list,0,8)+strmid(play_list,9,6))
               if ppcme.uselasco ne 'no' then begin
                 names=strarr(n_Elements(play_list),3,2)
                 p1=0
                   lasco=file_search(ppcme.outdir+'wlrt_'+'*L.png')
                   ltime=strmid(lasco,strpos(lasco(0),'.png')-21,15)
                   ltime=double(strmid(ltime,0,8)+strmid(ltime,9,6))
		   for i=0,n_Elements(play_list)-1 do begin
		      names(i,0,p1)=file_search(ppcme.outdir+'wlrt_'+play_list(i)+'*B.png')
                      names(i,1,p1)=lasco(find_closest(ptime(i),ltime))
		      names(i,2,p1)=file_search(ppcme.outdir+'wlrt_'+play_list(i)+'*A.png')
	           endfor
                  p1=p1+1
                   lasco=file_search(ppcme.outdir+'mass_'+'*L.png')
                   ltime=strmid(lasco,strpos(lasco(0),'.png')-21,15)
                   ltime=double(strmid(ltime,0,8)+strmid(ltime,9,6))
		   for i=0,n_Elements(play_list)-1 do begin
		      names(i,0,p1)=file_search(ppcme.outdir+'mass_'+play_list(i)+'*B.png')
                      names(i,1,p1)=lasco(find_closest(ptime(i),ltime))
		      names(i,2,p1)=file_search(ppcme.outdir+'mass_'+play_list(i)+'*A.png')
	           endfor
                endif else begin
                  names=strarr(n_Elements(play_list),2,2)
                  for i=0,n_Elements(play_list)-1 do begin
                       p1=0
		       names(i,0,p1)=file_search(ppcme.outdir+'wlrt_'+play_list(i)+'*B.png')
                       names(i,1,p1)=file_search(ppcme.outdir+'wlrt_'+play_list(i)+'*A.png')
		       p1=p1+1
                       names(i,0,p1)=file_search(ppcme.outdir+'mass_'+play_list(i)+'*B.png')
                       names(i,1,p1)=file_search(ppcme.outdir+'mass_'+play_list(i)+'*A.png')
 	          endfor
	        endelse
                generic_movie,names           
              endif
    	    END

	'HTPLOT' :  BEGIN 
               if ppcme.ofiles(0) eq 1 and ppcme.outfile gt -1 then begin
                   openw,ppcme.outfile,ppcme.htfile,/append
                   for i =0,n_Elements(htarr)-1 do printf,ppcme.outfile,htarr(i)
		   close,ppcme.outfile
	           FREE_LUN,ppcme.outfile
               endif
               if ppcme.ofiles(1) eq 1  and ppcme.outfile1 gt -1 then begin
                   openw,ppcme.outfile1,ppcme.rtfile,/append
                   for i =0,n_Elements(rtarr)-1 do printf,ppcme.outfile1,rtarr(i)
		   close,ppcme.outfile1
	           Free_lun,ppcme.outfile1
               endif
               ;ppcme.htfile = ''
               ppcme.outfile = -1
               ;ppcme.rtfile = ''
               ppcme.outfile1 = -1
               sXPLOT_HT, ppcme.htfile
    	    END

        'PLOTMASS' : begin
               device,decompose=1
               if ppcme.mpw eq -1 then begin
	         window,xsize=800,ysize=600,retain=2
                 ppcme.mpw=!d.window
               endif else wset,ppcme.mpw
               xrange=[min(ppcme.dates(where(ppcme.dates gt 0))),max(ppcme.dates(where(ppcme.dates gt 0)))]
               trange=utc2str(tai2utc(xrange))
               xpos=min(ppcme.dates(where(ppcme.dates gt 0)))+(30*60)
               yrange=[min(ppcme.tmass(where(ppcme.tmass gt 0))),max(ppcme.tmass(where(ppcme.tmass gt 0)))]
               ystep=(yrange(1)-yrange(0))/20
               clr=['000000'x,'0000FF'x,'00FF00'x,'FF0000'x,'FFFF00'x,'00FFFF'x,'FF8800'x,'FF00FF'x]
               for n=0,n_elements(ppcme.tels)-1 do begin
                  z=where(ppcme.tmass(n,*) gt 0,cnt)
                  if cnt ge 1 then begin
                    times=utc2str(tai2utc(ppcme.dates(n,z)))
                    if n eq 0 then utplot,times,ppcme.tmass(n,z),background='FFFFFF'x,color=clr(n),yrange=yrange,timerange=trange else $
                       outplot,times,ppcme.tmass(n,z),color=clr(n)
                    xyouts,30*60,yrange(1)-(ystep+ystep*n),ppcme.tels(n),color=clr(n)
                  endif
               endfor
              device,decompose=0
    	    END


	'DONE' :  BEGIN 	
               wins=[0,1,2,3,20,21]
	       for i=0,n_elements(wins)-1 do wdel,wins(i)
               if ppcme.ofiles(0) eq 1 and ppcme.outfile gt -1 then begin
                   openw,ppcme.outfile,ppcme.htfile,/append
                   for i =0,n_Elements(htarr)-1 do printf,ppcme.outfile,htarr(i)
		   close,ppcme.outfile
	           FREE_LUN,ppcme.outfile
               endif
               if ppcme.ofiles(1) eq 1  and ppcme.outfile1 gt -1 then begin
                   openw,ppcme.outfile1,ppcme.rtfile,/append
                   for i =0,n_Elements(rtarr)-1 do printf,ppcme.outfile1,rtarr(i)
		   close,ppcme.outfile1
	           Free_lun,ppcme.outfile1
               endif
               if ppcme.ofiles(4) eq 1 then begin
                    massfile = ppcme.outdir+'Total_Mass_'+string(utc2yymmdd(tai2utc(ppcme.dates(0,0)),/yyyy),'(i8)')+'.txt'
		    OPENW,tmout,massfile,/get_lun
		    for j=0,n_Elements(ppcme.tels)-1 do begin
                      printf,tmout,'Sensor:  '+ppcme.tels(j)
		      z=where(ppcme.tmass(j,*) gt 0,cnt)
                      for j1=0,cnt-1 do printf,tmout,utc2str(tai2utc(ppcme.dates(j,z(j1)))),' ',string(ppcme.tmass(j,z(j1)),'(e14.7)')
		    endfor
                    close,tmout
		    free_lun,tmout
               endif

               WIDGET_CONTROL, ppcme.cmebase , /DESTROY 
    	    END

	ELSE : BEGIN
                 PRINT, '%%Process CME  Unknown event.'
	     END

    ENDCASE

return
END 


pro process_cme,dir,sellong=sellong,backgrounds=backgrounds

common pprocesscme

device,decompose=0
undefine,htarr
undefine,rtarr

 tels = ['cor2a','cor2b','c2','c3','cor1a','cor1b','hi1a','hi1b']
 SC = ['SECCHI','SECCHI','LASCO','LASCO','SECCHI','SECCHI','SECCHI','SECCHI']

if datatype(ppcme) ne 'UND' then begin
  bminmax=ppcme.bminmax
  fscales=ppcme.fscales
endif else begin
  bminmax=dblarr(n_Elements(tels),2) & bminmax(*,0)=-2.4e+11 & bminmax(*,1)=1.2e+12
  fscales=strarr(n_elements(tels)) + 'Linear'
endelse
if keyword_set(sellong) then begin
   sellong=1 
   bminmax=dblarr(n_Elements(tels),2) & bminmax(*,0)=-8 & bminmax(*,1)=16
endif else sellong=0
if keyword_set(backgrounds) then backgrounds=backgrounds else backgrounds=['','','','','','','','']

ifiles=strarr(n_Elements(tels),500)
dates=dblarr(n_Elements(tels),500)
done_list=strarr(n_Elements(tels),500)
tmass=dblarr(n_Elements(tels),500)
cloudmass=dblarr(n_Elements(tels),500)

if datatype(dir) ne 'STR' then begin
  tmp = PICKFILE( TITLE='Select directory containing MASS FITS FILES',GET_PATH=dir)
		    if strmid(dir,0,1,/reverse_offset) ne '/' then dir=dir+'/'
endif

if datatype(dir) eq 'STR' then begin
  sz=size(dir)
  if sz(0) eq 0 and sellong eq 0 then begin
    outdir=dir
    files=file_search(dir+'/20*[2Q]m???.fts')
    z=where(strpos(files,'2mc2A') gt 0) & if (z(0) gt -1 ) then ifiles(0,0:n_elements(z)-1)=files(z)
    z=where(strpos(files,'2mc2B') gt 0) & if (z(0) gt -1 ) then ifiles(1,0:n_elements(z)-1)=files(z)
    z=where(strpos(files,'mc2L') gt 0)
    if (z(0) gt -1 ) then begin
       ifiles(2,0:n_elements(z)-1)=files(z)
       if strpos(ifiles(3,z(0)),'_Qm') gt -1 then sc(2)=sc(2)+' QUICK LOOK ' 
    endif
    z=where(strpos(files,'mc3L') gt 0)
    if (z(0) gt -1 ) then begin
       ifiles(3,0:n_elements(z)-1)=files(z) 
       if strpos(ifiles(3,z(0)),'_Qm') gt -1 then sc(3)=sc(3)+' QUICK LOOK ' 
    endif
    z=where(strpos(files,'2mc1A') gt 0) & if (z(0) gt -1 ) then ifiles(4,0:n_elements(z)-1)=files(z)
    z=where(strpos(files,'2mc1B') gt 0) & if (z(0) gt -1 ) then ifiles(5,0:n_elements(z)-1)=files(z)
    z=where(strpos(files,'2mh1A') gt 0) & if (z(0) gt -1 ) then ifiles(6,0:n_elements(z)-1)=files(z)
    z=where(strpos(files,'2mh1B') gt 0) & if (z(0) gt -1 ) then ifiles(7,0:n_elements(z)-1)=files(z) 
  endif else ifiles = dir

  for nw =0,n_Elements(tels)-1 do begin
    z=where(ifiles(nw,*) ne '',cnt)
    if nw eq 0 then fcnt=cnt else fcnt=[fcnt,cnt]
    if cnt ge 1 then begin
        for i=0,n_Elements(z)-1 do begin
         im = sccreadfits(ifiles[nw,i],h,/nodata)
         dates[nw,i]= anytim2tai(h.date_obs)
      endfor  
    endif
  endfor
endif else fcnt=[0,0,0]

tcnt=where(fcnt gt 0,cnt)
if cnt lt 1 then begin
  print,'********** NO MASS FILE FOUND IN DIRECTORY '+dir+' *****************'
  return
endif else begin
  lcnt=max(fcnt) 
  tels=tels(tcnt) & sc=sc(tcnt)
  ifiles=ifiles(tcnt,0:lcnt)
  backgrounds=backgrounds(tcnt)
  dates=dates(tcnt,0:lcnt)
  done_list=done_list(tcnt,0:lcnt)
  tmass=tmass(tcnt,0:lcnt)
  cloudmass=cloudmass(tcnt,0:lcnt)
endelse

if sellong eq 0 then outdir=dir else outdir=''

cmebase=widget_base(/COL, TITLE='Process CME')

row1=widget_base(cmebase,/row)
cola=widget_base(row1,/colum)

;  ssel=CW_BGROUP(cola,['Select SECCHI images','Select LASCO images'],BUTTON_UVALUE = ['SSEL','LSEL'],/ROW)
if sellong eq 0 then begin
  rows=widget_base(cola,/row,/frame)
  cmed = CW_FIELD(rows, VALUE=dir, XSIZE=40, YSIZE=1, UVALUE='CMED', /ALL_EVENTS, title='CME Fits Directory:')
  ssel=CW_BGROUP(rows,['Select images'],BUTTON_UVALUE = ['SSEL'],/ROW)
endif else cmed=-1

  scls=['Linear','LOG','SQRT']
  sfiles=intarr(n_Elements(tels))
  bckgrnd=intarr(n_Elements(tels))
  bmint=intarr(n_Elements(tels))
  bmaxt=intarr(n_Elements(tels))
  sclst=intarr(n_Elements(tels))

  row1=widget_base(cola,/row)
 
  for nw=0,n_Elements(tels)-1 do begin
     files=ifiles(nw,*)
     files=files(where(files ne ''))
;     if nw eq 0 or (nw eq 4 and sellong eq 0) then begin
     if nw eq 0 then begin
       colb=widget_base(row1,/colum)
     endif
     cola=widget_base(colb,/colum,/frame)
     tmp=widget_label(cola,value='********************* '+sc(nw)+'-'+tels(nw)+' *********************')
     row1bb=widget_base(cola,/row)
     agroup=CW_BGROUP(row1bb,['Select bytscl'],BUTTON_UVALUE = ['SELMINMAX_'+tels(nw)],/ROW)
     sfiles(nw) = WIDGET_COMBOBOX(row1bb, VALUE=strmid(files,strlen(files(0))-25,26), UVALUE='FSEL_'+tels(nw))
     ;if sellong ne 0 then widget_control,agroup,map=0
     ;if sellong eq 0 then begin
      ; row1bb=widget_base(cola,/row)
      ; bmint(nw) = CW_FIELD(row1bb, VALUE=STRTRIM(bminmax(nw,0),2), XSIZE=6, YSIZE=1, UVALUE='BMINT_'+tels(nw), /ALL_EVENTS, title='BYTSCL Min')
      ; bmaxt(nw) = CW_FIELD(row1bb, VALUE=STRTRIM(bminmax(nw,1),2), XSIZE=6, YSIZE=1, UVALUE='BMAXT_'+tels(nw), /ALL_EVENTS, title='Max')
      ; sclst(nw) = WIDGET_DROPLIST(row1bb, VALUE=scls, UVALUE='SCLS_'+tels(nw))
      ; WIDGET_CONTROL,sclst(nw),SET_DROPLIST_SELECT= where(scls eq fscales(nw))
     ;  agroup=CW_BGROUP(row1bb,['Select bytscl'],BUTTON_UVALUE = ['SELMINMAX_'+tels(nw)],/ROW,xoffset=40)
     ;endif
  endfor


outfiles=['HT','Param','ssim','swire','mass']
if sellong eq 0 then ofiles=[1,1,1,0,1] else ofiles=[0,0,0,0,0]
if sellong eq 0 then begin
  row2=widget_base(cmebase,/row)
  tmp=widget_label(row2,VALUE='Output Files: ')
  outf=CW_BGROUP(row2,outfiles,BUTTON_UVALUE=outfiles,/NONEXCLUSIVE,/row)
  row2=widget_base(cmebase,/row)
  widget_control,outf,set_value=(ofiles)
  odir = CW_FIELD(row2, VALUE=outdir, XSIZE=40, YSIZE=1, UVALUE='ODIR', /ALL_EVENTS, title='Output Dir')
  lsel=CW_BGROUP(row2,['Select Dir'],BUTTON_UVALUE = ['pickfile'],/ROW)
endif else odir=-1

row3=widget_base(cmebase,/row)
tmp=widget_label(row3,value='Use SECCHI - ')
usesv=['cor2','cor1','hi1']
uses = WIDGET_DROPLIST(row3, VALUE=usesv, UVALUE='uses')
tmp=widget_label(row3,value='      LASCO - ')
uselv=['no','c2','c3']
usel = WIDGET_DROPLIST(row3, VALUE=uselv, UVALUE='usel')
row3=widget_base(cmebase,/row)
tmp=widget_label(row3,VALUE='      ')
if sellong eq 0 then $
    lsel=CW_BGROUP(row3,['Run current image','Play Movie','HT plot','Plot Total Mass','EXIT'],BUTTON_UVALUE = ['RUN','PLAY','HTPLOT','PLOTMASS','DONE'],/ROW) else $
      lsel=CW_BGROUP(row3,['Run current image','EXIT'],BUTTON_UVALUE = ['RUN','DONE'],/ROW)
    



ppcme={cmebase:cmebase ,$
      tels:tels, $
      SC: SC ,$
      ifiles:ifiles ,$
      done_list:done_list ,$
      dates:dates ,$
      bminmax:bminmax ,$
      fscales:fscales ,$
      scls:scls ,$
      sfiles:sfiles ,$
      bckgrnd:bckgrnd ,$
      bmint:bmint ,$
      bmaxt:bmaxt ,$
      sclst:sclst ,$
      ofiles:ofiles,$
      outfiles:outfiles ,$
      outfile:-1 ,$
      outfile1:-1 ,$
      htfile:'' ,$
      rtfile:'' ,$
      massfile:'' ,$
      runsel:intarr(n_Elements(tels)) ,$      
      outdir:outdir,$       
      dir:dir,$       
      cmed:cmed,$       
      uselasco:'no',$
      usel:usel,$
      usesecchi:'cor2',$
      uses:uses,$
      uselv:uselv,$
      usesv:usesv,$
      tmass:tmass,$
      cloudmass:cloudmass,$
      sellong:sellong,$
      odir :odir,$
      backgrounds :backgrounds,$
       mpw :-1}

  WIDGET_CONTROL, /REAL, cmebase

  XMANAGER, 'Process CME', cmebase, EVENT_HANDLER='process_cme_event'

end
