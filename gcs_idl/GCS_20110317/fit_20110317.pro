;+
;*************************************************************************
; Francisco A. Iglesias - franciscoaiglesias@hotmail.com
;
; Runs all the individual FMwLASCO sripts contained in this directory 
; to fit the GCS modell to different instants of time using data 
; from different instruments
;
; History:
;	20180712 - First version
;*************************************************************************
;-
pro fit_20110317
@gcs_config

;*****CONSTANTS
DATE='20110317'

;*****MAIN
f=file_search(GCSD_PATH+'/GCS_'+DATE+'/','FMw*.pro')
a= strmid(f,strpos(f[0],'FMwLASCO')+16, 2)
b=a
for i=0, n_elements(a)-1 do begin
 a[i]=strsplit(a[i],'.',/extract)	
 b[i]=strjoin(strsplit(a[i],'m',/extract,/PRESERVE_NULL),'-')
endfor
b=sort(float(b))
f=reverse(f[b])
a=reverse(a[b])
print, f
; runs the FMw scripts one at a time
for i=0, n_elements(a)-2 do begin
 print,file_basename(f[i],'.pro')
 bla=execute(file_basename(f[i],'.pro'))
 savf=GCSD_PATH+'/GCS_'+DATE+'/'+strtrim(a[i+1])+'.sav'
 if ~file_search(savf) then begin	
  spawn, 'cp '+GCSD_PATH+'/GCS_'+DATE+'/'+a[i]+'.sav '+savf
  message,'.sav file not found! running...cp '+GCSD_PATH+'/GCS_'+DATE+'/'+a[i]+'.sav '+savf,/informational
 endif
endfor
bla=execute(file_basename(f[i],'.pro'))
while  !D.WINDOW ne -1 do wdelete
bla=execute('tevo_FMwLASCO'+DATE) ; runs the plotting script
message,'FINISHED!!!',/informational
end