;+
;  $Id: rotnmat.pro,v 1.1 2006/11/01 17:24:04 thernis Exp $
;
; PURPOSE:
;  Compute a rotation matrix useful for the Dragger
;
; CATEGORY:
;  data handling, mathematics
;
; HISTORY:
;originally from Dragger.pro
;Nishant Patel, Russell Howard, Arnaud Thernisien, et. al.
;July 23, 2004
;-
;------------------------------------------------
function ROTNMAT,alpha, beta, gamma
;
;  calculates the rotation matrix
;
;    cosMat = FLTARR(3, 3)


;  Transform the angle in radians.
;

    ralpha = alpha * !DPI / 180.0
    rbeta = beta * !DPI / 180.0
    rgamma = gamma * !DPI / 180.0

    rx=[[1,0,0],$
        [0,cos(ralpha),sin(ralpha)],$
        [0,-sin(ralpha),cos(ralpha)]]
    
    ry=[[cos(rbeta),0,-sin(rbeta)],$
        [0,1,0],$
        [sin(rbeta),0,cos(rbeta)]]

    rz=[[cos(rgamma),sin(rgamma),0],$
        [-sin(rgamma),cos(rgamma),0],$
        [0,0,1]]

    cosMat=ry#rx#rz

    return,cosMat
  
END

