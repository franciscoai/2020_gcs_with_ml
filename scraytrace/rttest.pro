;+
;
; PURPOSE:
;  Perform some simple test to see if the install went properly
;
; CATEGORY:
;  raytracing, test
;
; INPUTS:
;  none
;
; OUTPUTS:
;  display on screen
;
;  $Id: rttest.pro,v 1.4 2011/08/18 20:38:07 thernis Exp $
;
;-
pro rttest

; ---- init environment variables
rtinitenv

; ---- test 1
;      try simple integration
raytracewl,sbt,imsize=[32,32],modelid=21,losnbp=32,/c3,/usedefault

testflag1=(sbt.im[5,5] gt 0) and (sbt.im[5,5] lt 1e-13) and (sbt.imsize[0] eq 32)

if testflag1 then print,'Test 1 passed !'

; ---- test 2
;      to be implemented...
 


testflag=testflag1 ; and testflag2 and ...
if testflag then begin
    print,"Looks good !"
endif else begin
    print,"Test not passed: there might be something wrong in the installation..."
endelse

return
end
