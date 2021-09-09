function getdensity,point,modelid,modparam=modparam,sopath=sopath,quiet=quiet,unload=unload

;+
; PURPOSE:
;  Get the electron density of a model at a point of the space
; 
; CATEGORY:
;  simulation, raytracing, 3d
;
; INPUTS:
;  point : [x,y,z] position in space where you wnat the density
;  modelid : density number
;  modparam : array of parameters to pass to the density model
;  sopath : path where the C progs are located in case a different branch needs to be tested
;  quiet : quiet mode
;  unload : force to unload the shared object 
;
; OUTPUTS:
;  return : the electron density at the requested point and the
;           requested density model
;
; $Id: getdensity.pro,v 1.2 2006/10/30 21:52:40 thernis Exp $
;-


if n_elements(point) ne 3 then point=[0.,0,0] else point=float(point)
if n_elements(modelid) eq 0 then modelid=1L else modelid=long(modelid)
if n_elements(modparam) eq 0 then modparam=0. else modparam=float(modparam)
quiet=keyword_set(quiet)

; -- init output density
neout=0.

rtinitenv ; -- init environment variables


if n_elements(sopath) eq 0 then sopath=getenv('RT_LIBFILE')

; -- call density 
if not quiet then starttime=systime(1)
s=call_external(sopath,$
                'getdensity',$
                modelid,point,modparam,neout,unload=unload)

if not quiet then begin
    print,'Seconds ellapsed :'
    print,systime(1)-starttime
endif


return,neout
end
;
; CVSLOG:
;  $Log: getdensity.pro,v $
;  Revision 1.2  2006/10/30 21:52:40  thernis
;  Change the environment variable that points to the shared object library.
;
;
