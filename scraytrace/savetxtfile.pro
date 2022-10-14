;+
; $Id: savetxtfile.pro,v 1.2 2008/08/21 13:37:35 thernis Exp $
;
; PURPOSE:
;  read a text formated file and place each row in a string array
;
; CATEGORY:
;  io, data handling
;
; INPUTS:
;  filename : file name to save
;
; OUTPUTS:
;  tabtxt : array to save containing one line of text per subscript
;
;-
pro savetxtfile,filename,tabtxt

if n_params() eq 0 then begin
    message,'calling : savetxtfile,filename,tabtxt',/info
    return
endif

openw,lun,filename,/get_lun
for i=0,n_elements(tabtxt)-1 do begin
    printf,lun,tabtxt[i]
endfor
free_lun,lun

return
end
