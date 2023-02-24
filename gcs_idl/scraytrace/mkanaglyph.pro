;+
;  $Id: mkanaglyph.pro,v 1.1 2006/09/08 16:03:20 nathan Exp $
;
; NAME:
;  mkanaglyph.pro
;
;
; PURPOSE:
;  Build an anaglyph image from two images with two point-of-view2
;
; CATEGORY:
;  visualization, 3d
;
; CALLING SEQUENCE:
;  ima=mkanaglyph(imright,imleft)
;
;
; INPUTS:
;  imright : image seen by the right eye
;  imleft : image seen by the left eye
;
;
; KEYWORD PARAMETERS:
;  nogreen : set green channel to 0 instead of the right image
;
;
; OUTPUTS:
;  ima : anaglyph image [3,sx,sy]
;
;
; EXAMPLE:
;  ima=mkanagliph(imr,iml)
;  tv,ima,true=1
;
;-

function mkanaglyph,imr,iml,nogreen=nogreen

szr=size(imr,/dim)

img=bytarr(3,szr[0],szr[1])

img[0,*,*]=iml
if not keyword_set(nogreen) then img[1,*,*]=imr
img[2,*,*]=imr

return,img
end
