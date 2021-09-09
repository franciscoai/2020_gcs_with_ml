;+
;
; NAME:
;     READTXTFILE
;
; PURPOSE:
;  read a text formated file and place each row in a string array
;
; CATEGORY:
;  io, data handling
;
; INPUTS:
;  filename : file name to read
;
; OUTPUTS:
;  txtline : array containing the lines of the file
;  nlines : number of lines
;
; $Id: readtxtfile.pro,v 1.1 2008/07/29 14:56:19 thernis Exp $
;-


pro readtxtfile,filename,txtline,nlines


if n_params() eq 0 then begin
    message,'calling sequence: readtxtfile,filename,txtline,nlines',/info
    return
endif



;init of the var
txtline=['']
nlines=0
line=''
;start extraction loop
openr,lun,filename,/get_lun
while (not eof(lun)) do begin
  readf,lun,line
  txtline=[txtline,line]
  nlines = nlines +1
endwhile
if nlines gt 0 then txtline=txtline(1:*)
;print,'nlines=',nlines
free_lun,lun

return
end
