;+
;*************************************************************************
; Francisco A. Iglesias - UTN-FRM/GEAA - franciscoaiglesias@frm.utn.edu.ar
;
; Download the specified data from the corresponding mission repository
;
; History:
;	20180409 - First version
;*************************************************************************
;-
pro download_data_20110317

;Tasks
do_aia=1   ; if set allways downloads the data
do_euvia=1 ; if set downloads only if the files are not there
do_euvib=1 ; if set downloads only if the files are not there

;***SDO/AIA
if do_aia then begin
 ODIR=DATA_PATH+'/sdo/aia/L1/193/20110317' 
 bus=vso_search('2011/03/17 11:25:00',' 2011/03/17 11:35:00', source='SDO', $
 instr='AIA', wave='193', sample=60.*5) ;Gets metadate from VSO
 
 sopa = vso_get(bus, /RICE, out_dir=ODIR) ;Downloads the corresponding data
endif


;***STEREOA/EUVI
if do_euvia then begin
 DATE='20110317'
 FILES=['105530_n4euA','110030_n4euA','110530_n4euA','111530_n4euA','112030_n4euA','113030_n4euA']
 EUVI_REPO='https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0/a/img/euvi'
 EUVI_ODIR=DATA_PATH+'/stereo/secchi/L0/a/img/euvi/'+DATE

 EUVI_SRC=EUVI_REPO+'/'+SDATE+'/'+SDATE+'_'+FILES+'.fts'
 for i=0, n_elements(EUVI_SRC)-1 do begin
  if file_test(EUVI_ODIR+'/'+SDATE+'_'+FILES[i]+'.fts') eq 0 then $
    spawn,'wget '+string(34B)+EUVI_SRC[i]+string(34B)+' -P '+ EUVI_ODIR ; downloads file if it does not exist
 endfor
endif


;***STEREOB/EUVI
if do_euvib then begin
 DATE='20110317'
 FILES=['105530_n4euB','110030_n4euB','110530_n4euB','111530_n4euB','112030_n4euB','113030_n4euB']
 EUVI_REPO='https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0/b/img/euvi'
 EUVI_ODIR=DATA_PATH+'/stereo/secchi/L0/b/img/euvi/'+DATE

 EUVI_SRC=EUVI_REPO+'/'+SDATE+'/'+SDATE+'_'+FILES+'.fts'
 for i=0, n_elements(EUVI_SRC)-1 do begin
  if file_test(EUVI_ODIR+'/'+SDATE+'_'+FILES[i]+'.fts') eq 0 then $
    spawn,'wget '+string(34B)+EUVI_SRC[i]+string(34B)+' -P '+ EUVI_ODIR ; downloads file if it does not exist
 endfor
endif

end
