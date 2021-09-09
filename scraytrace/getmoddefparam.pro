pro getmoddefparam,modelid,ss,paramfile,verb=verb,statusflag=statusflag
;+
; PURPOSE:
;  returns the model default parameters
;
; CATEGORY:
;  raytracing
;
; INPUTS:
; modelid : model id number
; verb : set to display processing info
;
; OUTPUTS:
;  statusflag : bit 0 : default parameters undefined. 
;               bit 1 : model with no interest only for testing.
;               bit 2 : no parameters needed.
;
; ss : structure containing the default parameter structure and
;      modelid number.
; paramfile : string array containing the default parameters.
;
;  $Id: getmoddefparam.pro,v 1.2 2006/10/30 21:52:40 thernis Exp $
;-

modelid=long(modelid)
verb=keyword_set(verb)
statusflag=0L 


rtinitenv
starttime=systime(1)
exitstatus=call_external(getenv('RT_LIBFILE'),$
                'getdefaultparameteridl',$
                modelid,statusflag,/unload)

undefflag=statusflag and 1
fortestflag=ishft(statusflag,-1) and 1
noparamflag=ishft(statusflag,-2) and 1
if verb then begin
    print,'Seconds ellapsed :'
    print,systime(1)-starttime
    if undefflag then print,'Parameters undefined.' 
    if fortestflag then print,'Obsolete model or only for testing.'
    if noparamflag then print,'No parameters needed for that model.'
endif



;if undefflag or noparamflag then begin
;    paramfile=['']
;    return
;endif
; ---- get the result from the file
readtxtfile,'defmodparam.txt',paramfile,nbline

; ---- put the result in a structure
if nbline eq 0 then begin
    ; -- set to 1 if the structure is empty
    s=1
endif else begin
    ; -- fill in the structure
    s0={keyword:'',def:'',desc:'',units:''}
    
    nbfield=n_tags(s0)
    nbparam=nbline/nbfield
    
    s=replicate(s0,nbparam)
    
    idx=0L
    for i=0,nbparam-1 do $
      for j=0,nbfield-1 do s[i].(j)=paramfile[idx++]

endelse

ss={s:s,modelid:modelid}


return
end
;
; CVSLOG:
;  $Log: getmoddefparam.pro,v $
;  Revision 1.2  2006/10/30 21:52:40  thernis
;  Change the environment variable that points to the shared object library.
;
;
;
