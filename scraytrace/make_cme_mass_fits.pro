;+
;$Id: make_cme_mass_fits.pro,v 1.23 2012/07/11 13:08:52 mcnutt Exp $
;
; Project     : SECCHI/LASCO
;                   
; Name        : make_cme_mass_fits
;               
; Purpose     : wigdet tool for input to rtsccguicloud
;               
; Explanation : To select input image for rtsccguicloud
;                     
; Use         : IDL> make_cme_mass_fits
;
; Optional Inputs:
;
; output: selected files cme ht fits, and param file
;
; KEYWORDS: 
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
;$Log: make_cme_mass_fits.pro,v $
;Revision 1.23  2012/07/11 13:08:52  mcnutt
;checkingmake_cme_mass_fits.pro
;
;Revision 1.22  2012/05/17 15:27:21  mcnutt
;corrected background select
;
;Revision 1.21  2012/05/11 18:39:36  mcnutt
;changed image list from droplist to combobox for scroll bars
;
;Revision 1.20  2012/04/30 17:10:52  mcnutt
;corrected file name when calling process_cme
;
;Revision 1.19  2012/04/27 14:07:56  mcnutt
;added option to call process_cme to set the Carrington longitued and latitude before mass is calculated
;
;Revision 1.18  2012/04/13 18:33:08  mcnutt
;removed full res mass fits option
;
;Revision 1.17  2012/04/13 14:52:49  mcnutt
;relabeled select image buttons
;
;Revision 1.16  2012/04/13 14:43:21  mcnutt
;seperated c2 and c3 image selection from the cor2 selections
;
;Revision 1.15  2012/04/11 16:14:04  mcnutt
;correrted reset comment
;
;Revision 1.14  2012/04/10 19:38:26  mcnutt
;change lasco to use reduce_std_size instead of scc_pitin_array
;
;Revision 1.13  2011/03/03 19:21:55  avourlid
;Fixed bug in pcme.good line 303
;
;Revision 1.12  2011/03/02 18:57:43  mcnutt
;added more background selection options
;
;Revision 1.11  2011/02/25 15:51:31  mcnutt
;corected C2 and C3 filenames
;
;Revision 1.10  2011/02/23 18:41:46  mcnutt
;corrected lasco mass file
;
;Revision 1.9  2011/02/23 18:38:29  mcnutt
;corrected lasco mass files and added notification if ql is used
;
;Revision 1.8  2011/02/22 18:41:06  mcnutt
;improved update list background always first -1
;
;Revision 1.7  2011/02/17 20:35:40  mcnutt
;first image in list always background
;
;Revision 1.6  2011/02/17 18:36:32  mcnutt
;modified to include cor1 and hi1 files change common block structure/ complete overhall
;
;Revision 1.5  2010/09/10 12:37:47  avourlid
;removed SPWAWN commands to reduce OS-related errors, AV
;
;Revision 1.4  2010/09/08 15:37:19  mcnutt
;corrected lasco full res mass files
;
;Revision 1.3  2010/09/08 14:34:04  mcnutt
;added resolution selection and end time
;
;Revision 1.2  2010/08/27 16:20:59  mcnutt
;correct outdir
;
;Revision 1.1  2010/08/26 13:25:34  mcnutt
;selects image to create mass fits files
;
;Revision 1.1  2010/08/12 15:06:14  mcnutt
;Beta version
;

pro make_cme_mass_fits_event,ev

common pprocesscme,ppcme,htarr,rtarr,sellon_sgui
common processcme,pcme


COMMON scc_playmv_COMMON, moviev
COMMON selbytscl, selbyt
COMMON reduce_history, cmnver, prev_a, prev_hdr, zblocks0

   if(where(tag_names(ev) eq 'VALUE') gt -1)then input=ev.Value else WIDGET_CONTROL, ev.id, GET_UVALUE=input
   WIDGET_CONTROL, ev.id, GET_UVALUE=test
   IF (DATATYPE(test) EQ 'STR') THEN input = test
   IF strpos(input,'_') ge 0 then begin
     telsel=strmid(input,strpos(input,'_')+1,strlen(input))
     nw=where(pcme.tels eq telsel) & nw=nw(0)
     input=strmid(input,0,strpos(input,'_'))
   endif

    CASE (input) OF


	'SSEL' :  BEGIN 	
                     
                     if strmid(pcme.cmetime2,0,2) ne '20' then tmp=cme_file_query(pcme.cmetime) else $
		     	    	tmp=cme_file_query(pcme.cmetime,pcme.cmetime2,/ahead,/behind)
		     if tmp.sc_b(0) ne '' then begin
                        nw=where(pcme.tels eq 'cor2b') & nw=nw(0)
                        pcme.ifiles(nw,*)=''
                        pcme.backgrounds(nw)=tmp.sc_b(0)
                        pcme.ifiles(nw,0:n_elements(tmp.sc_b)-1)=tmp.sc_b(0:n_elements(tmp.sc_b)-1)
                        pcme.good(nw,where(pcme.ifiles(nw,*) ne '')) =1 & pcme.good(nw,0) =0
			files=pcme.ifiles(nw,where(pcme.good(nw,*) eq 1))
                        files=strmid(files,strlen(files(0))-25,25)
                        widget_control,pcme.bckgrnd(nw),set_value=strmid(pcme.backgrounds(nw),strlen(pcme.backgrounds(nw))-25,25)
                        WIDGET_CONTROL,pcme.sfiles(nw),SET_value=files 
	             endif
		     if tmp.sc_a(0) ne '' then begin
                        nw=where(pcme.tels eq 'cor2a') & nw=nw(0)
                        pcme.ifiles(nw,*)=''
                        pcme.backgrounds(nw)=tmp.sc_a(0)
                        pcme.ifiles(nw,0:n_elements(tmp.sc_a)-1)=tmp.sc_a(0:n_elements(tmp.sc_a)-1)
                        pcme.good(nw,where(pcme.ifiles(nw,*) ne '')) =1 & pcme.good(nw,0) =0
			files=pcme.ifiles(nw,where(pcme.good(nw,*) eq 1))
                        files=strmid(files,strlen(files(0))-25,25)
                        widget_control,pcme.bckgrnd(nw),set_value=strmid(pcme.backgrounds(nw),strlen(pcme.backgrounds(nw))-25,25)
                        WIDGET_CONTROL,pcme.sfiles(nw),SET_value=files 
	             endif
    	    END
	'C2SSEL' :  BEGIN 	
                    if strmid(pcme.cmetime2,0,2) ne '20' then tmp=cme_file_query(pcme.cmetime) else $
		     	    	tmp=cme_file_query(pcme.cmetime,pcme.cmetime2,/c2)
		     if tmp.c2(0) ne '' then begin
                        nw=where(pcme.tels eq 'c2') & nw=nw(0)
                        pcme.ifiles(nw,*)=''
                        pcme.backgrounds(nw)=tmp.c2(0)
                        pcme.ifiles(nw,0:n_elements(tmp.c2)-1)=tmp.c2(0:n_elements(tmp.c2)-1)
                        pcme.good(nw,where(pcme.ifiles(nw,*) ne '')) =1 & pcme.good(nw,0) =0
			files=pcme.ifiles(nw,where(pcme.good(nw,*) eq 1))
                        if strpos(files(0),'/ql/') gt -1 then WIDGET_CONTROL,pcme.lasconote(0),SET_value='!!!QUICK LOOK!!!!!' else $
			    WIDGET_CONTROL,pcme.lasconote(0),SET_value='Level 05' 
                        files=strmid(files,strlen(files(0))-25,25)
                        widget_control,pcme.bckgrnd(nw),set_value=strmid(pcme.backgrounds(nw),strlen(pcme.backgrounds(nw))-25,25)
                        WIDGET_CONTROL,pcme.sfiles(nw),SET_value=files
                     endif
    	    END
	'C3SSEL' :  BEGIN 	
                      if strmid(pcme.cmetime2,0,2) ne '20' then tmp=cme_file_query(pcme.cmetime) else $
		     	    	tmp=cme_file_query(pcme.cmetime,pcme.cmetime2,/c3)
		     if tmp.c3(0) ne '' then begin
                        nw=where(pcme.tels eq 'c3') & nw=nw(0)
                        pcme.ifiles(nw,*)=''
                        pcme.backgrounds(nw)=tmp.c3(0)
                        pcme.ifiles(nw,0:n_elements(tmp.c3)-1)=tmp.c3(0:n_elements(tmp.c3)-1)
                        pcme.good(nw,where(pcme.ifiles(nw,*) ne '')) =1 & pcme.good(nw,0) =0
			files=pcme.ifiles(nw,where(pcme.good(nw,*) eq 1))
                        if strpos(files(0),'/ql/') gt -1 then WIDGET_CONTROL,pcme.lasconote(1),SET_value='!!!QUICK LOOK!!!!!' else $
			    WIDGET_CONTROL,pcme.lasconote(1),SET_value='Level 05' 
                        files=strmid(files,strlen(files(0))-25,25)
                        widget_control,pcme.bckgrnd(nw),set_value=strmid(pcme.backgrounds(nw),strlen(pcme.backgrounds(nw))-25,25)
                        WIDGET_CONTROL,pcme.sfiles(nw),SET_value=files
                     endif

    	    END

 	'COR1SSEL' :  BEGIN 	
                     if strmid(pcme.cmetime2,0,2) ne '20' then tmp=cme_file_query(pcme.cmetime,/cor1) else $
		     	    	tmp=cme_file_query(pcme.cmetime,pcme.cmetime2,/cor1)
		     if tmp.cor1b(0) ne '' then begin
                        nw=where(pcme.tels eq 'cor1b') & nw=nw(0)
                        pcme.ifiles(nw,*)=''
                        pcme.backgrounds(nw)=tmp.cor1b(0)
                        pcme.ifiles(nw,0:n_elements(tmp.cor1b)-1)=tmp.cor1b(0:n_elements(tmp.cor1b)-1)
                        pcme.good(nw,where(pcme.ifiles(nw,*) ne '')) =1 & pcme.good(nw,0) =0
			files=pcme.ifiles(nw,where(pcme.good(nw,*) eq 1))
                        files=strmid(files,strlen(files(0))-25,25)
                        widget_control,pcme.bckgrnd(nw),set_value=strmid(pcme.backgrounds(nw),strlen(pcme.backgrounds(nw))-25,25)
                        WIDGET_CONTROL,pcme.sfiles(nw),SET_value= files
	             endif
		     if tmp.cor1a(0) ne '' then begin
                        nw=where(pcme.tels eq 'cor1a') & nw=nw(0)
                        pcme.ifiles(nw,*)=''
                        pcme.backgrounds(nw)=tmp.cor1a(0)
                        pcme.ifiles(nw,0:n_elements(tmp.cor1a)-1)=tmp.cor1a(0:n_elements(tmp.cor1a)-1)
                        pcme.good(nw,where(pcme.ifiles(nw,*) ne '')) =1 & pcme.good(nw,0) =0
			files=pcme.ifiles(nw,where(pcme.good(nw,*) eq 1))
                        files=strmid(files,strlen(files(0))-25,25)
                        widget_control,pcme.bckgrnd(nw),set_value=strmid(pcme.backgrounds(nw),strlen(pcme.backgrounds(nw))-25,25)
                        WIDGET_CONTROL,pcme.sfiles(nw),SET_value= files
	             endif
 
    	    END

 	'HI1SSEL' :  BEGIN 	
                     
                     if strmid(pcme.cmetime2,0,2) ne '20' then tmp=cme_file_query(pcme.cmetime,/hi1) else $
		     	    	tmp=cme_file_query(pcme.cmetime,pcme.cmetime2,/hi1)
		     if tmp.hi1b(0) ne '' then begin
                        nw=where(pcme.tels eq 'hi1b') & nw=nw(0)
                        pcme.ifiles(nw,*)=''
                        pcme.backgrounds(nw)=tmp.hi1b(0)
                        pcme.ifiles(nw,0:n_elements(tmp.hi1b)-1)=tmp.hi1b(0:n_elements(tmp.hi1b)-1)
                        pcme.good(nw,where(pcme.ifiles(nw,*) ne '')) =1 & pcme.good(nw,0) =0
			files=pcme.ifiles(nw,where(pcme.good(nw,*) eq 1))
                        files=strmid(files,strlen(files(0))-25,25)
                        widget_control,pcme.bckgrnd(nw),set_value=strmid(pcme.backgrounds(nw),strlen(pcme.backgrounds(nw))-25,25)
                        WIDGET_CONTROL,pcme.sfiles(nw),SET_value= files
	             endif
		     if tmp.hi1a(0) ne '' then begin
                        nw=where(pcme.tels eq 'hi1a') & nw=nw(0)
                        pcme.ifiles(nw,*)=''
                        pcme.backgrounds(nw)=tmp.hi1a(0)
                        pcme.ifiles(nw,0:n_elements(tmp.hi1a)-1)=tmp.hi1a(0:n_elements(tmp.hi1a)-1)
                        pcme.good(nw,where(pcme.ifiles(nw,*) ne '')) =1 & pcme.good(nw,0) =0
			files=pcme.ifiles(nw,where(pcme.good(nw,*) eq 1))
                        files=strmid(files,strlen(files(0))-25,25)
                        widget_control,pcme.bckgrnd(nw),set_value=strmid(pcme.backgrounds(nw),strlen(pcme.backgrounds(nw))-25,25)
                        WIDGET_CONTROL,pcme.sfiles(nw),SET_value= files
	             endif
 
    	    END

        'CMET' : pcme.cmetime=ev.value
        'CMET2' : pcme.cmetime2=ev.value


         'MOVIE': begin
            pcme.movie=telsel
            pcme.updated=0
            use=where(pcme.good(nw,*) eq 1)
            if n_Elements(use) gt 1 then begin
              files=[pcme.backgrounds(nw),reform(pcme.ifiles(nw,use))]
              files=files(where(files ne ''))
              if datatype(moviev) eq 'STC' then begin
               widget_control,moviev.base,/destroy,bad_id=error
               widget_control,moviev.winbase,/destroy,bad_id=error
              endif
              if pcme.scale(nw) eq 'Select' then selb=1 else selb=0
              if pcme.scale(nw) eq 'Autoscale' then auto=1 else auto=0
              if pcme.sc(nw) eq 'SECCHI' and strpos(telsel,'co') gt -1 then fmask=1 else fmask=0
              if pcme.sc(nw) eq 'LASCO'then mocc=1 else mocc=0
              if pcme.sc(nw) eq 'LASCO'then outer=1 else outer=0
              if strpos(telsel,'cor1') gt -1 then pb=1 else pb=0
              if strpos(telsel,'hi') gt -1 then dorot=0 else dorot=1
              if strpos(telsel,'hi') gt -1 then pan=0 else pan=512
	      scc_mkmovie,files,pcme.bminmax(nw,0),pcme.bminmax(nw,1),/diff,/times,FMASK=fmask,dorotate=dorot,$
	    	    	    pbser=pb,pan=pan,selectbmin=selb,/nocal,automax=auto,MASK_OCC=mocc,outer=outer
              
              if pcme.scale(nw) eq 'Select' then begin
               pcme.bminmax(nw,*)=[selbyt.bmin,selbyt.bmax]
               pcme.scale(nw)=pcme.scls(0)
               widget_control,pcme.bmint(nw),set_value=pcme.bminmax(nw,0)
               widget_control,pcme.bmaxt(nw),set_value=pcme.bminmax(nw,1)
               WIDGET_CONTROL,pcme.sclst(nw),SET_DROPLIST_SELECT= where(pcme.scls eq pcme.scale(nw))
	      endif
	     endif
	   END
           
         'UPDATE': begin
            if pcme.movie eq telsel and pcme.updated ne 1 and datatype(moviev) eq 'STC' then begin
              pcme.updated=1
              use=where(pcme.good(nw,*) eq 1)
              if n_Elements(use) gt 1 then begin
                files=[pcme.backgrounds(nw),reform(pcme.ifiles(nw,use))]
                files=files(where(files ne ''))
	        pcme.backgrounds(nw)=files(moviev.first)
                files=files(1:n_Elements(files)-1) ;remove backgorund image from list
                good=where(moviev.deleted eq 0) 
                good=good(where(good ge moviev.first and good le moviev.last))
                files=strmid(files,strlen(files(0))-25,25)
                widget_control,pcme.bckgrnd(nw),set_value=strmid(pcme.backgrounds(nw),strlen(pcme.backgrounds(nw))-25,25)
                WIDGET_CONTROL,pcme.sfiles(nw),SET_value= files(good)
                pcme.good(nw,*)=0
                pcme.good(nw,use(good))=1
               endif
             endif else print,'current movie is not '+telsel+' or file list was already been updated'
	   END

        'BKGSEL': begin
                if datatype(moviev) ne 'STC' then begin
		  bkg=widget_info(pcme.sfiles(nw),/COMBOBOX_GETTEXT)
                  use=where(pcme.good(nw,*) eq 1)
                  if n_Elements(use) gt 1 then begin
		   files=reform(pcme.ifiles(nw,use))
		   test=where(strpos(files,bkg) gt -1)
	           pcme.backgrounds(nw)=files(test)
                   pcme.good(nw,*)=0
                   files=strmid(files,strlen(files(0))-25,25)
                   pcme.good(nw,use(test+1:n_Elements(files)-1))=1
                   widget_control,pcme.bckgrnd(nw),set_value=strmid(pcme.backgrounds(nw),strlen(pcme.backgrounds(nw))-25,25)
                   WIDGET_CONTROL,pcme.sfiles(nw),SET_value= files(test+1:n_Elements(files)-1)
                 endif
                endif else print,'Can not select background while scc_playmovie is running'
	   END

         'RESET': begin
                if datatype(moviev) ne 'STC' then begin
		  files=reform(pcme.ifiles(nw,*))
	          pcme.backgrounds(nw)=files(0)
                  files=files(where(files ne ''))
                  if n_Elements(files) gt 1 then begin
                   files=strmid(files,strlen(files(0))-25,25)
                   pcme.good(nw,*)=0
                   pcme.good(nw,1:n_Elements(files)-1)=1
                   widget_control,pcme.bckgrnd(nw),set_value=strmid(pcme.backgrounds(nw),strlen(pcme.backgrounds(nw))-25,25)
                   WIDGET_CONTROL,pcme.sfiles(nw),SET_value= files(where(pcme.good(nw,*) ge 1))
                 endif
               endif else print,'Can not reset image list while scc_playmovie is running'
	   END

        'BMINT' : pcme.bminmax(nw,0)=ev.value
        'BMAXT' : pcme.bminmax(nw,1)=ev.value
        'SCLS':   pcme.scale(nw)=pcme.scls(ev.index)
        'FRES' :  pcme.outres=ev.index
        'ODIR' :  pcme.outdir=ev.value
        'FSEL' :  test=0

        'pickfile' : BEGIN
	       file = PICKFILE( TITLE='Select output directory',GET_PATH=path)
		    if strmid(path,0,1,/reverse_offset) ne '/' then path=path+'/'
                    pcme.outdir=path
                    widget_control,pcme.odir,set_value=pcme.outdir

    	    END

        'MKMASS' : BEGIN
               if N_ELEMENTS(where(pcme.good gt 0)) ne 1 then begin
                tmp=file_search(pcme.outdir)
                if tmp eq '' then begin
                   ;cmd='mkdir '+pcme.outdir
		   ;spawn,cmd
		file_mkdir, pcme.outdir
		endif
                if strmid(pcme.outdir,0,1,/reverse_offset) ne '/' then path=pcme.outdir+'/' else path=pcme.outdir
                outsize=512
		if pcme.outres eq 1 then outsize=1024 
		if pcme.outres eq 2 then outsize=0
                widget_control,pcme.setlon,get_value=lon
                widget_control,pcme.setlat,get_value=lat
		pcme.cmelonlat=[lon,lat]
                if total(pcme.cmelonlat) ne 0. then cmelonlat=pcme.cmelonlat else cmelonlat=0
                for nw =0, n_Elements(pcme.tels)-1 do begin
		  files=reform(pcme.ifiles(nw,*))
                  if strtrim(files(0),2) ne 'Not selected' then begin
                      files=files(where(pcme.good(nw,*) eq 1))
                      if strpos(pcme.tels(nw),'cor1') gt -1 then pb=1 else pb=0
                      if pcme.sc(nw) eq 'SECCHI' then $
		      	secchi_prep,pcme.backgrounds(nw),h0,back,outsize=outsize,pb=pb;,/CALFAC_OFF, /NOCALFAC,/calimg_off,/NOWARP,/BKGIMG_OFF
                      if pcme.sc(nw) eq 'LASCO' then reduce_level_1,pcme.backgrounds(nw),h0,back,/NOFITS
     	              if strpos(pcme.tels(nw),'co') gt -1 then mask=get_smask(h0)
                      for i=0,n_Elements(files)-1 do begin   
                          if pcme.sc(nw) eq 'LASCO' then begin
                            reduce_level_1,files(i),hdr,image,/NOFITS
		            if pcme.outres eq 0 then image=reduce_std_size((image-back),hdr) else image=image-back
                           ; hdr=convert2secchi(thdr)
		           ; if pcme.outres eq 0 then image=scc_putin_array((image-back),hdr,512) else image=image-back
                          endif else begin
			    secchi_prep,files(i),hdr,image,outsize=outsize,pb=pb;,/CALFAC_OFF, /NOCALFAC,/calimg_off,/NOWARP,/BKGIMG_OFF
                            image=image-back
    	                  endelse
			  img=SCC_CALC_CME_MASS(image,hdr,/all,cmelonlat=cmelonlat)
                          if strpos(pcme.tels(nw),'co') gt -1 then img=img*mask
                          if pcme.sc(nw) eq 'LASCO' then begin
                            fname=hdr.date_obs                                         
                            if strpos(files(i),'/ql/') gt -1 then ml='_Qm' else ml='_2m'
			    fname=strmid(fname,0,4)+strmid(fname,5,2)+strmid(fname,8,2)+'_'+strmid(fname,11,2)+strmid(fname,14,2)+strmid(fname,17,2)+ml+pcme.tels(nw)+'L.fts'
    	                  endif else begin
			    fname=hdr.filename
    	                    strput,fname,'2m',strpos(fname,'.fts')-5 
                          endelse
    	                  hdr.filename=fname
                          hdr.bunit='grams'
                          fits_hdr = struct2fitshead(hdr,/allow_crota,/dateunderscore)
		          writefits,path+hdr.filename,img,fits_hdr
                      endfor 
		   endif
		endfor
	       endif

    	    END

        'SELLON' : BEGIN
               if N_ELEMENTS(where(pcme.good gt 0)) ne 1 then begin
                  ufiles=pcme.ifiles
                  for nw =0, n_Elements(pcme.tels)-1 do begin
		    files=reform(pcme.ifiles(nw,*))
                    ufiles(nw,*)=''
                    if strtrim(files(0),2) ne 'Not selected' then begin
                      files=files(where(pcme.good(nw,*) eq 1))
                      ufiles(nw,0:n_elements(files)-1)=files
     	            endif
                  endfor
		  sellon=1
                  process_cme,ufiles,/sellon,backgrounds=pcme.backgrounds
                  if datatype(sellon_sgui) eq 'STC' then begin
                    pcme.cmelonlat=[sellon_sgui.lon*!radeg,sellon_sgui.lat*!radeg]
                    widget_control,pcme.setlon,set_value=pcme.cmelonlat(0)
                    widget_control,pcme.setlat,set_value=pcme.cmelonlat(1)
                  endif
                endif
    	    END

	'DONE' :  BEGIN 	
            if datatype(moviev) eq 'STC' then begin
               widget_control,moviev.base,/destroy,bad_id=error
               widget_control,moviev.winbase,/destroy,bad_id=error
            endif 
               WIDGET_CONTROL, pcme.cmebase , /DESTROY 
    	    END

	ELSE : BEGIN
                 PRINT, '%%Make CME Fits Unknown event.'
	     END

    ENDCASE

return
END 


pro make_cme_mass_fits

common processcme

 tels = ['cor2a','cor2b','c2','c3','cor1a','cor1b','hi1a','hi1b']
 SC = ['SECCHI','SECCHI','LASCO','LASCO','SECCHI','SECCHI','SECCHI','SECCHI']

if datatype(pcme) ne 'UND' then begin
  bminmax=pcme.bminmax
  scale=pcme.scale
  cmetime=pcme.cmetime
  cmetime2=pcme.cmetime2
endif else begin
  bminmax=fltarr(n_Elements(tels),2) & bminmax(*,0)=-8.0 & bminmax(*,1)=16.0
  scale=strarr(n_elements(tels)) + 'Select'
  cmetime='YYYY/MM/DD hh:mm'
  cmetime2='YYYY/MM/DD hh:mm'
endelse


ifiles=strarr(n_Elements(tels),500)
good=intarr(n_Elements(tels),500) +0
backgrounds=strarr(n_Elements(tels)) +'Not selected'


if  datatype(list) eq 'STC' then begin

                     tmp=cme_file_query(pcme.cmetime)
		     if list.sc_a(0) ne '' then begin
                        backgrounds(0)=list.sc_a(0)
                        ifiles(0,*)=list.sc_a(0:n_elements(list.sc_a)-1)
                        good(0,where(ifiles(0,*) ne '')) =1 & good(0,0) =0
 	             endif
		     if list.sc_b(0) ne '' then begin
                        backgrounds(1)=list.sc_b(0)
                        ifiles(1,*)=list.sc_b(0:n_elements(list.sc_b)-1)
                        good(1,where(ifiles(1,*) ne '')) =1 & good(1,0) =0
                     endif
		     if list.c2(0) ne '' then begin
			backgrounds(2)=list.c2(0)
                        ifiles(2,*)=list.c2(0:n_elements(list.c2)-1)
                        good(2,where(ifiles(2,*) ne '')) =1 & good(2,0) =0
                     endif
		     if list.c3(0) ne '' then begin
			 backgrounds(3)=list.c3(0)
                         ifiles(3,*)=list.c3(0:n_elements(list.c3)-1)
                         good(3,where(ifiles(3,*) ne '')) =1 & good(3,0) =0
                     endif
		     if list.cor1a(0) ne '' then begin
			 backgrounds(4)=list.cor1a(0)
                         ifiles(4,*)=list.cor1a(0:n_elements(list.cor1a)-1)
                         good(4,where(ifiles(4,*) ne '')) =1 & good(4,0) =0
                     endif
		     if list.cor1b(0) ne '' then begin
			 backgrounds(5)=list.cor1b(0)
                         ifiles(5,*)=list.cor1b(0:n_elements(list.cor1b)-1)
                         good(5,where(ifiles(5,*) ne '')) =1 & good(5,0) =0
                     endif
		     if list.hi1b(0) ne '' then begin
			 backgrounds(6)=list.hi1a(0)
                         ifiles(6,*)=list.hi1a(0:n_elements(list.hi1a)-1)
                         good(6,where(ifiles(6,*) ne '')) =1 & good(6,0) =0
                     endif
		     if list.hi1b(0) ne '' then begin
			 backgrounds(7)=list.hi1b(0)
                         ifiles(7,*)=list.hi1b(0:n_elements(list.hi1b)-1)
                         good(7,where(ifiles(7,*) ne '')) =1 & good(7,0) =0
                     endif

endif else ifiles(*,0)='Not selected             '

cmebase=widget_base(/COL, TITLE='CME MASS FITS TOOL')

rowa=widget_base(cmebase,/row)
cola=widget_base(rowa,/colum,/frame)

;  ssel=CW_BGROUP(cola,['Select SECCHI images','Select LASCO images'],BUTTON_UVALUE = ['SSEL','LSEL'],/ROW)
  rows=widget_base(cola,/row,/frame)
  cmet = CW_FIELD(rows, VALUE=cmetime, XSIZE=18, YSIZE=1, UVALUE='CMET', /ALL_EVENTS, title='Time Of CME or Start time:')
  cmet = CW_FIELD(rows, VALUE=cmetime2, XSIZE=18, YSIZE=1, UVALUE='CMET2', /ALL_EVENTS, title='END time:')
  tmp=widget_label(rows,value='Select Images:')
  ssel=CW_BGROUP(rows,['COR2 A/B','LASCO C2','LASCO C3','COR1 A/B','HI1 A/B'],BUTTON_UVALUE = ['SSEL','C2SSEL','C3SSEL','COR1SSEL','HI1SSEL'],/ROW)

cd, current=result
outdir = result + '/'

row2d=widget_base(cola,/row)
odir = CW_FIELD(row2d, VALUE=outdir, XSIZE=60, YSIZE=1, UVALUE='ODIR', /ALL_EVENTS, title='Output Dir')
lsel=CW_BGROUP(row2d,['Select Dir'],BUTTON_UVALUE = ['pickfile'],/ROW)
;lsel=CW_BGROUP(row3,['RUN'],BUTTON_UVALUE = ['RUN'],/ROW)
ores=['512','1024'];,'Full Res']
outres=0
tmp=widget_label(row2d,value='Resolution:')
fsize = WIDGET_DROPLIST (row2d, VALUE=ores, UVALUE='FRES')
lsel=CW_BGROUP(row2d,['Make MASS Fits Files','EXIT'],BUTTON_UVALUE = ['MKMASS','DONE'],/ROW)

row2a=widget_base(cola,/row)
cmelonlat=[0.,0.]
setlon = CW_FIELD(row2a, VALUE=string(cmelonlat(0),'f8.3'), XSIZE=8, YSIZE=1, UVALUE='SETLON', /ALL_EVENTS, title='CME CARRINGTON LONGITUDE:')
setlat = CW_FIELD(row2a, VALUE=string(cmelonlat(0),'f8.3'), XSIZE=8, YSIZE=1, UVALUE='SETLAT', /ALL_EVENTS, title='LATITUDE:')
lsel2=CW_BGROUP(row2a,['Run Process CME'],BUTTON_UVALUE = ['SELLON'],/ROW)
;lsel=CW_BGROUP(row3,['RUN'],BUTTON_UVALUE = ['RUN'],/ROW)


  scls=['Use Current','Select','Autoscale']
  sfiles=intarr(n_Elements(tels))
  bckgrnd=intarr(n_Elements(tels))
  bmint=intarr(n_Elements(tels))
  bmaxt=intarr(n_Elements(tels))
  sclst=intarr(n_Elements(tels))
  lasconote=intarr(2)

row1=widget_base(cmebase,/row)

  for nw=0,n_Elements(tels)-1 do begin
     if nw eq 0 or nw eq 4 then begin
       colb=widget_base(row1,/colum,/frame)
     endif
     cola=widget_base(colb,/colum,/frame)
     files=ifiles(nw,*)
     files=files(where(files ne ''))
     files=strmid(files,strlen(files(0))-25,25)
     row2c=widget_base(cola,/row)
     bgroup=CW_BGROUP(row2c,['RESET'],BUTTON_UVALUE = ['RESET_'+tels(nw)],/ROW)
     tmp=widget_label(row2c,value='********************* ')
     tmp=widget_label(row2c,value=sc(nw)+'-'+strupcase(tels(nw)))
     if tels(nw) eq 'c2' then lasconote(0)=widget_text(row2c,value='not set',ysize=1,xsize=15)
     if tels(nw) eq 'c3' then lasconote(1)=widget_text(row2c,value='not set',ysize=1,xsize=15)
     tmp=widget_label(row2c,value=' *********************')

     row1c=widget_base(cola,/row)
     sfiles(nw) = WIDGET_COMBOBOX (row1c, VALUE=files, UVALUE='FSEL_'+tels(nw))
     bgroup=CW_BGROUP(row1c,['Use for BKGRND'],BUTTON_UVALUE = ['BKGSEL_'+tels(nw)],/ROW)
     tmp=widget_label(row1c,value='Background:')
     bckgrnd(nw)=widget_text(row1c,value=strmid(backgrounds(nw),strlen(backgrounds(nw))-25,25),ysize=1,xsize=25)

     row1bb=widget_base(cola,/row)
     bmint(nw) = CW_FIELD(row1bb, VALUE=STRTRIM(bminmax(nw,0),2), XSIZE=6, YSIZE=1, UVALUE='BMINT_'+tels(nw), /ALL_EVENTS, title='BYTSCL Min')
     bmaxt(nw) = CW_FIELD(row1bb, VALUE=STRTRIM(bminmax(nw,1),2), XSIZE=6, YSIZE=1, UVALUE='BMAXT_'+tels(nw), /ALL_EVENTS, title='Max')
     sclst(nw) = WIDGET_DROPLIST (row1bb, VALUE=scls, UVALUE='SCLS_'+tels(nw))
     WIDGET_CONTROL,sclst(nw),SET_DROPLIST_SELECT= where(scls eq scale(nw))
     bgroup=CW_BGROUP(row1bb,['Create Movie','update file list'],BUTTON_UVALUE = ['MOVIE_'+tels(nw),'UPDATE_'+tels(nw)],/ROW)
  endfor

pcme={cmebase:cmebase ,$
      cmetime:cmetime, $
      cmetime2:cmetime2, $
      tels:tels, $
      SC: SC ,$
      ifiles:ifiles ,$
      good:good ,$
      backgrounds:backgrounds ,$
      bminmax:bminmax ,$
      scale:scale ,$
      scls:scls ,$
      sfiles:sfiles ,$
      bckgrnd:bckgrnd ,$
      bmint:bmint ,$
      bmaxt:bmaxt ,$
      sclst:sclst ,$
      outdir:outdir,$       
      ores:ores,$
      outres:outres,$
      setlon:setlon,$
      setlat:setlat,$
      cmelonlat:cmelonlat,$
      lasconote:lasconote,$
      movie:'NA',$       
      updated:0,$ 
      odir :odir}

  WIDGET_CONTROL, /REAL, cmebase

  XMANAGER, 'Process CME', cmebase, EVENT_HANDLER='make_cme_mass_fits_event', /NO_BLOCK

end
