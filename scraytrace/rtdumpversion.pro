;+
;  $Id: rtdumpversion.pro,v 1.1 2006/11/09 19:42:14 thernis Exp $
; 
; PURPOSE: 
;  Print the version of the library
;
; CATEGORY:
;  raytracing, simulation, version management
;
; INPUTS:
;
; OUTPUTS:
;  print the version of the library
;
;-

pro rtdumpversion

rtinitenv
s=call_external(getenv('RT_LIBFILE'),$
                'dumpbuildinfo',/unload)

return
end
