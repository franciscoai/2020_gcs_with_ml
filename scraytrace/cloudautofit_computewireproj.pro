;+
; PURPOSE: Compute wireframe contour
;
; $Id: cloudautofit_computewireproj.pro,v 1.1 2009/02/05 20:16:44 thernis Exp $
; $Log: cloudautofit_computewireproj.pro,v $
; Revision 1.1  2009/02/05 20:16:44  thernis
; First commit
;
;-
pro cloudautofit_computewireproj,han,hgt,rat,lon,lat,rot,scom,wireprojA,wireprojB,nodisp=nodisp

rebfact=2.^scom.resolution
nbvertaxisp=long(150l * rebfact)
nbvertcirup=long(150l * rebfact)
nbvertshell=long(150l * rebfact)
oc=cmecloud(han,hgt,nbvertaxisp,nbvertcirup,rat,nbvertshell,/distjuncisleadingedge)

neang=[lon,lat,rot]

; ---- compute the wire-frame projection in A and B
; -- wire frame projection in A
imsize=[scom.imsidesize,scom.imsidesize]
rtcloud,oc,sa,imsize=imsize,scchead=scom.hdra,/quiet,neang=neang,/fclip
kerside=5
wireprojA=erode(dilate(sa.im,replicate(1,kerside,kerside)),replicate(1,kerside,kerside))-erode(dilate(sa.im,replicate(1,kerside,kerside)),replicate(1,kerside+2,kerside+2))
;wireprojA=convol(float(wireprojA),scom.kerconv,/center)
wireprojA*=*scom.masksunA
;maxwireprojA=max(wireprojA)
;if maxwireprojA gt 0. then wireprojA/=maxwireprojA
; -- wire frame projection in B
rtcloud,oc,sb,imsize=imsize,scchead=scom.hdrb,/quiet,neang=neang,/fclip
wireprojB=erode(dilate(sb.im,replicate(1,kerside,kerside)),replicate(1,kerside,kerside))-erode(dilate(sb.im,replicate(1,kerside,kerside)),replicate(1,kerside+2,kerside+2))
;wireprojB=convol(float(wireprojB),scom.kerconv,/center)
wireprojB*=*scom.masksunB
;maxwireprojB=max(wireprojB)
;if maxwireprojB gt 0. then wireprojB/=maxwireprojB

if ~keyword_set(nodisp) then begin
	wnd,2,bytscl(wireprojA+ *scom.imdistA)
	wnd,3,bytscl(wireprojB+ *scom.imdistB)
endif

return
end

