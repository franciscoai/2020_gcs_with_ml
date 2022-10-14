;+
; PURPOSE:
;  get the user's home directory for saving images
; 
; CATEGORY:
;  simulation, raytracing, data handling
;
; OUTPUT:
;  usersavepath : user's directory path
;
; CVS:
;  $Id: rtgetuserdir.pro,v 1.1 2006/09/08 16:03:22 nathan Exp $
;-


pro rtgetuserdir,usersavepath

usersavepath=getenv('HOME')
print,'User''s home directory : '+usersavepath
; ---- check if saveraytrace directory exist. If not, create it
if file_exist(usersavepath+'/saveraytrace') eq 0 then begin
    print,'Creating :'+usersavepath+'/saveraytrace'
    spawn,'mkdir '+usersavepath+'/saveraytrace'
endif
usersavepath=usersavepath+'/saveraytrace'

return
end
