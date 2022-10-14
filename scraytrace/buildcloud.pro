;+
;
; PURPOSE:
;  build a density cube
;
; CATEGORY:
;  simulation, raytracing, 3d, io
; 
; INPUTS:
;  modelid : model to generate the density cloud
;  modparam : parameters of the model if needed
;  cubesidenbpix : size of the output density cube in pixel
;  cubesidersun : size of the output density cube in Rsun. Note that
;                 the Sun center is always at the center of the cube
;  cubeoriginrsun : [x0,y0,z0] pos of the first voxel of the density
;                   cube (centered at thecenter of the voxel). By
;                   default = 
;                [x0=-cubesidersun/2.,y0=-cubesidersun/2.,z0=-cubesidersun/2.]
;  outputtype : 0 = outputXX.txt x/y/z/ne, 1= cubeXX.txt i/j/k/ne
;               2 = outputXX.dat sidenbpix,sideRsun,ne.... : binary
;                   formated file 
;               3 = outputXX.dat : binary density list formated file
;  quiet : set to avoid displaying progression messages
;
; OUTPUTS:
;  write a file on the disk outputXX.txt or cubeXX.txt, with XX the model ID
;
; CALL EXEMPLE:
;  buildcloud,14,cubesidenbpix=64L,cubesidersun=60.
;
; CVS:
;  $Id: buildcloud.pro,v 1.4 2009/04/02 20:35:58 thernis Exp $
;
;-



pro buildcloud,modelid,modparam=modparam,cubesidenbpix=cubesidenbpix, $
               cubesidersun=cubesidersun,outputtype=outputtype, $
               N=N,cpix=cpix,csun=csun,cubeoriginrsun=cubeoriginrsun,quiet=quiet

; 'N' is an alias for 'cubesidenbpix', to match other program conventions
; 'cpix' is also an alias for 'cubesidenbpix' to avoid typoes
; 'csun' is an alias for 'cubesidenrsun' to avoid typoes

if n_elements(modelid) eq 0 then modelid=1L

if n_elements(modparam) eq 0 then begin
    getmoddefparam,33,s
    modparam=parsemoddefparam(s,sv)
;    modparam=0. 
end else begin
    modparam=float(modparam)
end

if (n_elements(cpix) ne 0) then cubesidenbpix=long(cpix)
if (n_elements(N) ne 0) then cubesidenbpix=long(N)
if (n_elements(cubesidenbpix) eq 0) then cubesidenbpix=64L

if (n_elements(csun) ne 0) then cubesidersun=csun
if n_elements(cubesidersun) eq 0 then cubesidersun=12. else cubesidersun=float(cubesidersun)
if n_elements(cubeoriginrsun) eq 0 then cubeoriginrsun=-cubesidersun/2.*[1.,1,1] else cubeoriginrsun=float(cubeoriginrsun)

if n_elements(outputtype) eq 0 then outputtype=0
if n_elements(modparam) ne 0 then modparam=float(modparam)

if n_elements(quiet) eq 0 then quiet=0l else quiet=1L

; cast all to longs/floats to avoid CC problems
modelid=long(modelid)
cubesidenbpix=long(cubesidenbpix)
cubesidersun=float(cubesidersun)
outputtype=long(outputtype)

rtinitenv ; init share library filename

;print,modelid,cubesidenbpix,cubesidersun,outputtype,modparam
starttime=systime(1)
psep=path_sep()
rtexec=getenv('RT_LIBFILE')
s=call_external(rtexec,'buildcloud', modelid, $
                cubesidenbpix,cubesidersun,outputtype,modparam,cubeoriginrsun,quiet,/unload)

if quiet eq 0 then begin
;if (outputtype le 1 or outputtype eq 2) then begin
    print,'Seconds ellapsed :'
    print,systime(1)-starttime
;end
endif

return
end
;
; CVSLOG:
;  $Log: buildcloud.pro,v $
;  Revision 1.4  2009/04/02 20:35:58  thernis
;  Implement quiet mode
;
;  Revision 1.3  2009/03/06 22:13:04  thernis
;  Change comments
;
;  Revision 1.2  2006/10/30 21:52:40  thernis
;  Change the environment variable that points to the shared object library.
;
;
